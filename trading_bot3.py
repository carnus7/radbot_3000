import time
import threading
import datetime
import math
import sqlite3

import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from dearpygui.dearpygui import *

##############################################################################
#                           BOT CONFIG & SETUP
##############################################################################

BOT_NAME = "RadBot 3000"

API_KEY = "op6DbKpHWxECyjDf3qzNl7OKIKAFHjoUnkteZUUpNJBzztBu1jFhEmuLCAYtA8Vd"
API_SECRET = "ErAACkas4CkfcF1qtxdl3D3XG2amOHZffNO8lO2dX8Y00lyzemUEX8G59Oz94Wdi"
client = Client(API_KEY, API_SECRET, testnet=True)

DB_NAME = "trading_bot.db"

# Stop-loss ratio (10%)
stop_loss_ratio = 0.1

# The local dictionary of coin holdings purchased by this robot
# ignoring any pre-existing testnet stash.
purchased_holdings = {}  # symbol -> {"quantity": float, "buy_price": float, "cost_usd": float}

# For top altcoins caching
_cached_top_altcoins = []
_last_top_altcoins_fetch_time = 0
_TOP_ALTCOINS_FETCH_INTERVAL = 300

# The local trade capital used across *all* coins
trade_capital = 50.0
initial_trade_capital = 50.0

stop_trading = False

# Toggle to show/hide strategy decisions
show_strategy_decisions = True

##############################################################################
#                      DATABASE SETUP
##############################################################################
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        side TEXT,
        quantity REAL,
        price REAL,
        total_usd REAL,
        timestamp TEXT
    )
''')
c.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy TEXT,
        profit REAL,
        runtime TEXT,
        start_time TEXT,
        end_time TEXT
    )
''')
conn.commit()

##############################################################################
#                        STRATEGIES
##############################################################################

def sma_strategy(data, short_window=5, long_window=20):
    if len(data) < long_window:
        return "HOLD"
    data['SMA_short'] = data['close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['close'].rolling(window=long_window).mean()

    sma_short_now = data['SMA_short'].iloc[-1]
    sma_long_now = data['SMA_long'].iloc[-1]
    sma_short_prev = data['SMA_short'].iloc[-2]
    sma_long_prev = data['SMA_long'].iloc[-2]

    if sma_short_now > sma_long_now and sma_short_prev <= sma_long_prev:
        return "BUY"
    elif sma_short_now < sma_long_now and sma_short_prev >= sma_long_prev:
        return "SELL"
    return "HOLD"

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    if len(data) < period:
        return "HOLD"
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if rsi.iloc[-1] < oversold:
        return "BUY"
    elif rsi.iloc[-1] > overbought:
        return "SELL"
    return "HOLD"

def macd_strategy(data, short_window=12, long_window=26, signal_window=9):
    if len(data) < long_window + signal_window:
        return "HOLD"
    data['EMA_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    macd_now = data['MACD'].iloc[-1]
    macd_prev = data['MACD'].iloc[-2]
    signal_now = data['Signal'].iloc[-1]
    signal_prev = data['Signal'].iloc[-2]

    if macd_now > signal_now and macd_prev <= signal_prev:
        return "BUY"
    elif macd_now < signal_now and macd_prev >= signal_prev:
        return "SELL"
    return "HOLD"

def bollinger_strategy(data, period=20, num_std=2):
    if len(data) < period:
        return "HOLD"

    data['SMA'] = data['close'].rolling(window=period).mean()
    data['STD'] = data['close'].rolling(window=period).std()
    data['Upper'] = data['SMA'] + (num_std * data['STD'])
    data['Lower'] = data['SMA'] - (num_std * data['STD'])

    close_price = data['close'].iloc[-1]
    if close_price < data['Lower'].iloc[-1]:
        return "BUY"
    elif close_price > data['Upper'].iloc[-1]:
        return "SELL"
    return "HOLD"

def stochastic_strategy(data, k_period=14, d_period=3, overbought=80, oversold=20):
    if len(data) < (k_period + d_period):
        return "HOLD"
    data['Lowest_low'] = data['low'].rolling(window=k_period).min()
    data['Highest_high'] = data['high'].rolling(window=k_period).max()
    data['%K'] = ((data['close'] - data['Lowest_low']) /
                  (data['Highest_high'] - data['Lowest_low'])) * 100
    data['%D'] = data['%K'].rolling(window=d_period).mean()

    k_now = data['%K'].iloc[-1]
    k_prev = data['%K'].iloc[-2]
    d_now = data['%D'].iloc[-1]
    d_prev = data['%D'].iloc[-2]

    if k_now > d_now and k_prev <= d_prev and k_now < oversold:
        return "BUY"
    elif k_now < d_now and k_prev >= d_prev and k_now > overbought:
        return "SELL"
    return "HOLD"

def psar_strategy(data, af=0.02, af_max=0.2):
    if len(data) < 2:
        return "HOLD"
    psar = data['close'].copy()
    bull_trend = True
    af_current = af
    ep = data['high'].iloc[0]

    for i in range(1, len(data)):
        prev_psar = psar.iloc[i - 1]
        if bull_trend:
            psar.iloc[i] = prev_psar + af_current * (ep - prev_psar)
            if data['low'].iloc[i] < psar.iloc[i]:
                bull_trend = False
                psar.iloc[i] = ep
                af_current = af
                ep = data['low'].iloc[i]
            else:
                if data['high'].iloc[i] > ep:
                    ep = data['high'].iloc[i]
                    af_current = min(af_current + af, af_max)
        else:
            psar.iloc[i] = prev_psar - af_current * (prev_psar - ep)
            if data['high'].iloc[i] > psar.iloc[i]:
                bull_trend = True
                psar.iloc[i] = ep
                af_current = af
                ep = data['high'].iloc[i]
            else:
                if data['low'].iloc[i] < ep:
                    ep = data['low'].iloc[i]
                    af_current = min(af_current + af, af_max)

    last_close = data['close'].iloc[-1]
    last_psar = psar.iloc[-1]

    if last_psar < last_close:
        return "BUY"
    elif last_psar > last_close:
        return "SELL"
    return "HOLD"

STRATEGIES = {
    "SMA": sma_strategy,
    "RSI": rsi_strategy,
    "MACD": macd_strategy,
    "Bollinger": bollinger_strategy,
    "Stochastic": stochastic_strategy,
    "Parabolic SAR": psar_strategy,
}

##############################################################################
#                         UTILITY & TRADE LOGIC
##############################################################################

def log_message_ui(msg):
    current_logs = get_value("log_multiline")
    new_logs = f"{current_logs}\n{msg}"
    set_value("log_multiline", new_logs)
    print(msg)

def get_current_price(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except:
        return 0.0

def insert_trade_db(symbol, side, quantity, price, total_cost):
    tstamp = datetime.datetime.now().isoformat()
    c.execute(
        "INSERT INTO trades (symbol, side, quantity, price, total_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol, side, quantity, price, total_cost, tstamp)
    )
    conn.commit()

def fetch_symbol_min_max(symbol):
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                return (float(f["minQty"]), float(f["maxQty"]), float(f["stepSize"]))
    except:
        pass
    return (0.000001, 9999999, 0.000001)

def fetch_symbol_min_notional(symbol):
    """Fetch the minimum notional value for a given symbol."""
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "MIN_NOTIONAL":
                return float(f["minNotional"])
    except Exception as e:
        log_message(f"Error fetching MIN_NOTIONAL for {symbol}: {e}")
    return 10.0  # Default fallback value

def round_step_size(quantity, stepSize):
    if stepSize <= 0:
        return quantity
    precision = int(round(-math.log10(stepSize), 0))
    return round(quantity, precision)

def fetch_data_for_strategy(symbol, interval, lookback=50):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
        if not klines:
            log_message_ui(f"No kline data returned for {symbol}. Skipping.")
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.dropna(subset=['timestamp'], inplace=True)
        df['timestamp'] = df['timestamp'].astype(int)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        log_message_ui(f"Error fetch_data_for_strategy {symbol}: {e}")
        return pd.DataFrame()

def execute_buy(symbol, spend):
    global trade_capital
    
    minQty, maxQty, stepSize = fetch_symbol_min_max(symbol)
    minNotional = fetch_symbol_min_notional(symbol)
    price = get_current_price(symbol)

    if spend < minNotional:
        log_message_ui(f"Trade amount ${spend:.2f} is below the minimum notional ${minNotional:.2f}. Set to minimal notional.")
        spend = minNotional

    # Check if we have enough local capital
    if spend > trade_capital:
        log_message_ui(f"Cannot buy {symbol}, spend= {spend:.2f} > trade_capital= {trade_capital:.2f}.")
        return

    

    if price <= 0:
        log_message_ui(f"Invalid price for {symbol}.")
        return

    quantity = spend / price
    quantity = round_step_size(quantity, stepSize)

    if quantity < minQty:
        log_message_ui(f"quantity {quantity} < minQty {minQty}, abort BUY {symbol}")
        return
    if quantity > maxQty:
        quantity = maxQty

    # For brevity, ignoring minQty, etc. but can do similarly as your code
    try:
        order = client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=round(quantity, 8)
        )
        fill_price = float(order['fills'][0]['price'])
        total_cost = fill_price * quantity
        trade_capital -= total_cost

        purchased_holdings[symbol] = {
            "quantity": purchased_holdings.get(symbol, {}).get("quantity", 0.0) + quantity,
            "buy_price": fill_price,
            "cost_usd": purchased_holdings.get(symbol, {}).get("cost_usd", 0.0) + total_cost
        }

        insert_trade_db(symbol, "BUY", quantity, fill_price, total_cost)
        log_message_ui(f"Buy Executed: symbol={symbol}, qty={quantity:.6f}, fill_price={fill_price:.6f}, total={total_cost:.2f}")
        add_coin_trade_history(symbol, "BUY", quantity, fill_price, total_cost)
    except Exception as e:
        log_message_ui(f"Error buying {symbol}: {e}")

def execute_sell(symbol):
    global trade_capital

    info = purchased_holdings.get(symbol)
    if not info or info["quantity"] <= 0:
        log_message_ui(f"No BOT-owned holdings to sell for {symbol}.")
        return

    quantity = info["quantity"]
    price = get_current_price(symbol)
    if price <= 0:
        log_message_ui(f"Invalid price for SELL {symbol}.")
        return

    try:
        order = client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=round(quantity, 8)
        )
        fill_price = float(order['fills'][0]['price'])
        total_revenue = fill_price * quantity
        trade_capital += total_revenue

        insert_trade_db(symbol, "SELL", quantity, fill_price, total_revenue)
        log_message_ui(f"Sell Executed: symbol={symbol}, qty={quantity:.6f}, fill_price={fill_price:.6f}, total={total_revenue:.2f}")
        add_coin_trade_history(symbol, "SELL", quantity, fill_price, total_revenue)

        # remove from purchased_holdings
        del purchased_holdings[symbol]
    except Exception as e:
        log_message_ui(f"Error selling {symbol}: {e}")

##############################################################################
#                         PRICING GRAPH
##############################################################################

# We'll store price history for each coin
coin_price_history = {}  # symbol -> {"x":[], "y":[]}

##############################################################################
#                          UI UPDATE & LAYOUT
##############################################################################

def log_message_coin_tab(symbol, message):
    """Append a line to the coin's tab multiline."""
    text_tag = f"{symbol}_history_multiline"
    old_text = get_value(text_tag)
    new_line = f"{message}\n"
    set_value(text_tag, old_text + new_line)

def update_wallet_display():
    """
    We show the real USDT from the testnet (for reference),
    plus a list of the bot-owned coins with their USDT value, quantity, and profit.
    Also the overall profit in % vs initial capital.
    """
    usdt_balance = 0.0
    try:
        account_data = client.get_account()
        for b in account_data["balances"]:
            if b["asset"] == "USDT":
                usdt_balance = float(b["free"])
                break
    except Exception as e:
        pass

    set_value("wallet_balance_text", f"Binance USDT (Testnet): ${usdt_balance:.2f}")

    # Theoretical total = trade_capital + sum of all purchased coins
    total_value = trade_capital
    wallet_lines = ""
    for sym, info in purchased_holdings.items():
        qty = info["quantity"]
        cost_usd = info["cost_usd"]
        buy_price = info["buy_price"]
        curr_price = get_current_price(sym)
        curr_val = qty * curr_price
        # profit on that asset
        asset_profit = curr_val - cost_usd
        wallet_lines += f"{sym}: qty={qty:.6f}, val=${curr_val:.2f}, profit=${asset_profit:.2f}\n"
        total_value += curr_val

    if not wallet_lines:
        wallet_lines = "No purchased assets."

    set_value("wallet_assets_text", wallet_lines)

    if initial_trade_capital > 0:
        profit_pct = ((total_value - initial_trade_capital) / initial_trade_capital) * 100.0
    else:
        profit_pct = 0.0

    set_value("trade_capital_text", f"Trade Money: ${trade_capital:.2f}")
    set_value("profit_text", f"Profit: {profit_pct:.2f}%")
    set_value("total_asset", f"Assets: ${total_value:.2f}")

def color_line(side, quantity_usd_str):
    if side == "BUY":
        return f"[RED]{quantity_usd_str}[/RED]"
    else:
        return f"[GREEN]{quantity_usd_str}[/GREEN]"

def add_coin_trade_history(symbol, side, quantity, price, total_usd):
    """Log a buy/sell event with color-coded USDT cost or revenue."""
    history_tag = f"{symbol}_history_multiline"
    old_text = get_value(history_tag)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    quantity_usd_str = f"${total_usd:.2f}"
    color_str = color_line(side, quantity_usd_str)

    new_line = f"{timestamp_str} - {side} {quantity:.6f} @ {price:.6f} - {color_str}\n"
    set_value(history_tag, old_text + new_line)

def update_coin_tab(symbol):
    """Update the displayed quantity, price, stop-loss, and update the line plot data."""
    info = purchased_holdings.get(symbol, None)
    qty_bot = info["quantity"] if info else 0.0
    set_value(f"{symbol}_quantity_text", f"Quantity Held (Bot-Owned): {qty_bot:.6f}")

    try:
        ticker_price = get_current_price(symbol)
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        set_value(f"{symbol}_price_text", f"Current Price: {ticker_price:.6f}")
        set_value(f"{symbol}_timestamp_text", f"Last Update: {now_str}")

        # Stop-loss check
        if info and info["quantity"] > 0:
            buy_price = info["buy_price"]
            threshold = buy_price * (1.0 - stop_loss_ratio)
            if ticker_price < threshold:
                log_message_coin_tab(symbol, f"Stop-loss triggered. SELL {symbol} price={ticker_price:.6f}, threshold={threshold:.6f}")
                execute_sell(symbol)

        # Update graph
        ph = coin_price_history.setdefault(symbol, {"x":[], "y":[]})
        ph["x"].append(len(ph["x"]))  # simple x index
        ph["y"].append(ticker_price)
        set_value(f"{symbol}_line_series", [ph["x"], ph["y"]])

    except Exception as e:
        pass  # ignore

def get_top_altcoins(limit=10):
    global _cached_top_altcoins, _last_top_altcoins_fetch_time
    now = time.time()
    if _cached_top_altcoins and (now - _last_top_altcoins_fetch_time < _TOP_ALTCOINS_FETCH_INTERVAL):
        return _cached_top_altcoins
    try:
        tickers = client.get_ticker()
        sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        altcoins = [t['symbol'] for t in sorted_tickers if t['symbol'].endswith('USDT')]
        _cached_top_altcoins = altcoins[:limit]
        _last_top_altcoins_fetch_time = now
        return _cached_top_altcoins
    except:
        return _cached_top_altcoins or []

##############################################################################
#               MAIN MONITOR AND TRADE LOOP
##############################################################################

def monitor_and_trade_top_altcoins(interval, request_freq):
    global stop_trading
    strategy_name = get_value("strategy_dropdown")

    try:
        while not stop_trading:
            update_wallet_display()

            # read or refresh stop_loss_ratio
            try:
                user_sl = float(get_value("stop_loss_input"))
            except:
                user_sl = 10.0
            global stop_loss_ratio
            stop_loss_ratio = user_sl / 100.0

            # gather strategy params if needed (like RSI)
            # for brevity, just do RSI as example
            rsi_period = int(get_value("rsi_period_input"))
            rsi_overbought = float(get_value("rsi_overbought_input"))
            rsi_oversold = float(get_value("rsi_oversold_input"))

            # fetch top altcoins
            top_altcoins = get_top_altcoins()
            log_message_ui(f"Monitoring coins: {top_altcoins}")

            # for each coin
            negative_coins = []
            positive_coins = []
            sum_scores = 0.0

            for symbol in top_altcoins:
                initialize_coin_tab(symbol)
                update_coin_tab(symbol)

                # fetch data
                df = fetch_data_for_strategy(symbol, interval, lookback=50)
                if df.empty or "close" not in df.columns:
                    if show_strategy_decisions:
                        log_message_coin_tab(symbol, f"No data or 'close' missing. HOLD.")
                    continue

                # run the strategy to get a signal => convert to score
                signal = "HOLD"
                if strategy_name == "SMA":
                    signal = sma_strategy(df)
                elif strategy_name == "RSI":
                    signal = rsi_strategy(df, rsi_period, rsi_overbought, rsi_oversold)
                elif strategy_name == "MACD":
                    signal = macd_strategy(df)
                elif strategy_name == "Bollinger":
                    signal = bollinger_strategy(df)
                elif strategy_name == "Stochastic":
                    signal = stochastic_strategy(df)
                elif strategy_name == "Parabolic SAR":
                    signal = psar_strategy(df)

                # Convert to score
                if signal == "BUY":
                    sc = +1.0
                elif signal == "SELL":
                    sc = -1.0
                else:
                    sc = 0.0

                if show_strategy_decisions:
                    reason_msg = f"Strategy: {strategy_name} => {signal}"
                    log_message_coin_tab(symbol, reason_msg)

                if sc < 0:
                    negative_coins.append(symbol)
                elif sc > 0:
                    positive_coins.append(symbol)
                    sum_scores += sc

            # SELL negative coins
            for sym in negative_coins:
                if sym in purchased_holdings:
                    # SELL
                    execute_sell(sym)

            # BUY positive coins
            if sum_scores > 0:
                local_cap = trade_capital
                for sym in positive_coins:
                    # ratio
                    # for simplicity sc=1 for every coin => split evenly
                    # or you can parse sc from the loop above
                    if strategy_name == "SMA": 
                        sc = 1.0
                    else:
                        # just assume 1.0 for all
                        sc = 1.0
                    ratio = sc / sum_scores
                    buy_amount = local_cap * ratio
                    if buy_amount < 1e-6:
                        continue
                    execute_buy(sym, buy_amount)
            else:
                if show_strategy_decisions:
                    log_message_ui("No coin has a strictly positive score => skip buying.")

            time.sleep(request_freq)

    except Exception as e:
        log_message_ui(f"Error in trading loop: {e}")

    # Insert performance record
    c.execute(
        "INSERT INTO performance (strategy, profit, runtime, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
        (strategy_name, 0, "", str(datetime.datetime.now()), str(datetime.datetime.now()))
    )
    conn.commit()
    log_message_ui(f"{BOT_NAME} stopped with strategy: {strategy_name}")

##############################################################################
#                              UI LAYOUT
##############################################################################

def start_trading_thread():
    global stop_trading, trade_capital, initial_trade_capital
    stop_trading = False

    interval = get_value("interval_input")
    request_freq = float(get_value("request_freq_input"))
    try:
        typed_capital = float(get_value("trade_amount_input"))
    except:
        typed_capital = 50.0
    trade_capital = typed_capital
    initial_trade_capital = typed_capital

    t = threading.Thread(
        target=monitor_and_trade_top_altcoins,
        args=(interval, request_freq),
        daemon=True
    )
    t.start()

def stop_trading_thread():
    global stop_trading
    stop_trading = True
    log_message_ui("Stop signal sent. Bot will stop soon...")

def toggle_strategy_decisions(sender, data):
    global show_strategy_decisions
    show_strategy_decisions = not show_strategy_decisions

def setup_ui():
    create_context()
    create_viewport(title=BOT_NAME, width=1280, height=720)

    with window(label=BOT_NAME, width=1280, height=720):

        # # Menu bar for toggles
        # with menu_bar():
        #     with menu(label="Options"):
        #         add_menu_item(label="Show/Hide Strategy Decisions", callback=toggle_strategy_decisions)

        add_text(f"{BOT_NAME} - Version 3 gives u BROKE POWER", color=[255, 255, 0])
        add_separator()

        # Layout:
        #  Left: wallet
        #  then config
        #  then logs, coin tabs

        with child_window(tag="wallet_child", width=400, height=270, border=True):
            add_text("Wallet & Bot Profit", color=[0, 255, 255])
            add_text("Binance USDT (Testnet): $0.00", tag="wallet_balance_text")
            add_text("Trade Money: $0.00", tag="trade_capital_text")
            add_text("Profit: 0.00%", tag="profit_text")
            add_text("Assets: $0.00", tag="total_asset")
            add_separator()
            add_text("Bot-Owned Assets:")
            add_input_text(
                tag="wallet_assets_text",
                multiline=True,
                default_value="No purchased assets.",
                width=380,
                height=100,
                readonly=True
            )

        # Config
        with child_window(tag="config_child", width=380, height=270, border=True, pos=(410, 55)):
            add_text("Trading Configuration", color=[0, 255, 0])

            add_input_text(label="Binance Interval", default_value="1m", tag="interval_input", width=200)
            add_input_text(label="Trade Amount (USD)", default_value="50", tag="trade_amount_input", width=200)
            add_input_text(label="Request Frequency (sec)", default_value="60", tag="request_freq_input", width=200)

            add_combo(
                items=["SMA", "RSI", "MACD", "Bollinger", "Stochastic", "Parabolic SAR"],
                label="Strategy",
                default_value="RSI",
                tag="strategy_dropdown",
                width=200
            )

            # Stop Loss ratio
            add_input_text(label="Stop Loss (%)", default_value="10", tag="stop_loss_input", width=200)

            add_separator()
            add_button(label="Start Trading", callback=lambda s,a: start_trading_thread())
            add_button(label="Stop Trading", callback=lambda s,a: stop_trading_thread())

        # Strategy Fine-tuning
        with child_window(tag="strategy_tune_child", width=350, height=270, border=True, pos=(800, 55)):
            add_text("Strategy Fine-Tune Params", color=[255, 128, 0])

            add_text("RSI Params:")
            add_input_text(label="Period", default_value="14", tag="rsi_period_input", width=60)
            add_input_text(label="Overbought", default_value="70", tag="rsi_overbought_input", width=60)
            add_input_text(label="Oversold", default_value="30", tag="rsi_oversold_input", width=60)

        # coin tabs + logs in the rest
        with child_window(tag="coin_tabs_child", width=640, height=400, border=True, pos=(8, 328)):
            add_text("Coin Tabs (Ignoring Testnet Stash)", color=[255, 255, 0])
            with tab_bar(tag="coin_tab_bar"):
                pass

        # Logs
        with child_window(tag="logs_child", width=620, height=400, border=True, pos=(656, 328)):
            add_text("Logs", color=[255, 0, 255])
            add_input_text(
                tag="log_multiline",
                multiline=True,
                default_value="--- Logs ---",
                width=600,
                height=360,
                readonly=True
            )

    setup_dearpygui()
    show_viewport()

def add_coin_tab(symbol):
    """
    Create a tab for the coin with a small line plot on the right side.
    """
    with tab(label=symbol, parent="coin_tab_bar"):
        add_text(f"Coin: {symbol}", color=[255, 255, 0])
        add_separator()
        add_text("Quantity Held (Bot-Owned): 0.000000", tag=f"{symbol}_quantity_text")
        add_separator()
        add_text("Current Price: ???", tag=f"{symbol}_price_text")
        add_text("Last Update: ???", tag=f"{symbol}_timestamp_text")
        add_separator()

        add_text(f"{symbol} Buy/Sell History:", color=[0, 255, 0])
        add_input_text(
            tag=f"{symbol}_history_multiline",
            multiline=True,
            default_value="",
            width=600,
            height=120,
            readonly=True
        )
        add_separator()

        # Price Plot
        with plot(label="Price Chart", height=200, width=600):
            add_plot_axis(mvXAxis, label="Index", tag=f"{symbol}_x_axis")
            with plot_axis(mvYAxis, label="Price", tag=f"{symbol}_y_axis"):
                add_line_series([], [], label="Price", tag=f"{symbol}_line_series")

def initialize_coin_tab(symbol):
    # if does_item_exist => skip
    if does_item_exist(f"{symbol}_quantity_text"):
        return
    add_coin_tab(symbol)

##############################################################################
#                                MAIN
##############################################################################

def main():
    setup_ui()
    start_dearpygui()
    destroy_context()

if __name__ == "__main__":
    main()
