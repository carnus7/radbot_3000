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

# =================== BOT CONFIG ===================
BOT_NAME = "RadBot 3000"

API_KEY = "op6DbKpHWxECyjDf3qzNl7OKIKAFHjoUnkteZUUpNJBzztBu1jFhEmuLCAYtA8Vd"
API_SECRET = "ErAACkas4CkfcF1qtxdl3D3XG2amOHZffNO8lO2dX8Y00lyzemUEX8G59Oz94Wdi"
client = Client(API_KEY, API_SECRET, testnet=True)

# =================== DATABASE SETUP ===================
conn = sqlite3.connect("trading_bot.db", check_same_thread=False)
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

# =================== GLOBALS ===================
stop_trading = False

# The local dictionary of coin holdings purchased by this robot
# ignoring any pre-existing testnet stash.
purchased_holdings = {}  # symbol -> quantity we bought

# Each coin’s last buy price for stop-loss checks
last_buy_price = {}       # symbol -> float
stop_loss_ratio = 0.1     # default 10%

# For top altcoins caching
_cached_top_altcoins = []
_last_top_altcoins_fetch_time = 0
_TOP_ALTCOINS_FETCH_INTERVAL = 300

# The local trade capital used across *all* coins
trade_capital = 50.0
initial_trade_capital = 50.0

# =================== LOGGING & UI HELPER ===================
def log_message(msg):
    """Append a log message to the multiline widget and print to console."""
    current_logs = get_value("log_multiline")
    new_logs = f"{current_logs}\n{msg}"
    set_value("log_multiline", new_logs)
    print(msg)

def get_usdt_balance():
    """
    Return real USDT balance from Binance account (Testnet),
    but the bot won't directly rely on this if ignoring stash.
    Just for display.
    """
    try:
        account_data = client.get_account()
        for b in account_data["balances"]:
            if b["asset"] == "USDT":
                return float(b["free"])
        return 0.0
    except Exception as e:
        log_message(f"Error fetching USDT balance: {e}")
        return 0.0

def get_current_price(symbol):
    """Helper to get the current ticker price for the symbol."""
    try:
        return float(client.get_symbol_ticker(symbol=symbol)['price'])
    except Exception as e:
        log_message(f"Error fetching current price for {symbol}: {e}")
        return 0.0

def update_wallet_display():
    """
    Show real USDT from Binance,
    plus local trade_capital,
    plus profit in % based on (trade_capital + sum of all purchased_holdings * current_price).
    """
    usdt_balance = get_usdt_balance()  # purely display
    set_value("wallet_balance_text", f"Binance USDT (Testnet): ${usdt_balance:.2f}")

    # Calculate the "theoretical total" = trade_capital + sum of all purchased coins * current price
    # ignoring pre-existing stash
    total_value = trade_capital
    for sym, qty in purchased_holdings.items():
        if qty > 0:
            current_p = get_current_price(sym)
            total_value += (qty * current_p)

    if initial_trade_capital > 0:
        profit_pct = ((total_value - initial_trade_capital) / initial_trade_capital) * 100.0
    else:
        profit_pct = 0.0

    set_value("trade_capital_text", f"Trade Money: ${trade_capital:.2f}")
    set_value("profit_text", f"Profit: {profit_pct:.2f}%")
    set_value("total_asset", f"Assets: ${total_value:.2f}")

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
    except Exception as e:
        log_message(f"Error fetching top altcoins: {e}")
        return _cached_top_altcoins or []

def initialize_coin_tab(symbol):
    # create a tab if not exist
    if does_item_exist(f"{symbol}_quantity_text"):
        return

    with tab(label=symbol, parent="coin_tab_bar"):
        add_text(f"Coin: {symbol}", color=[255, 255, 0])
        add_separator()
        # We'll display only the quantity this bot purchased (purchased_holdings)
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
            width=-1,
            height=120,
            readonly=True
        )
        add_separator()

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
    """
    Show how many coins we *the bot* purchased,
    display current price,
    check stop-loss for the portion we bought if we have last_buy_price[symbol].
    """
    # quantity from purchased_holdings
    qty_bot = purchased_holdings.get(symbol, 0.0)
    configure_item(f"{symbol}_quantity_text", default_value=f"Quantity Held (Bot-Owned): {qty_bot:.6f}")

    try:
        ticker_price = get_current_price(symbol)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        configure_item(f"{symbol}_price_text", default_value=f"Current Price: {ticker_price:.6f}")
        configure_item(f"{symbol}_timestamp_text", default_value=f"Last Update: {now_str}")

        # Stop-loss check only if we have last_buy_price
        if symbol in last_buy_price and qty_bot > 0:
            buy_price = last_buy_price[symbol]
            threshold = buy_price * (1.0 - stop_loss_ratio)
            if ticker_price < threshold:
                log_message(f"Stop-loss triggered for {symbol}. price={ticker_price:.6f}, threshold={threshold:.6f}")
                execute_trade(symbol, SIDE_SELL)
                del last_buy_price[symbol]

    except Exception as e:
        log_message(f"Error updating coin tab for {symbol}: {e}")

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
    """Fetch some klines for the chosen strategy logic."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
        if not klines or klines is None:
            # No data returned
            log_message(f"No kline data returned for {symbol}. Skipping.")
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = df['timestamp'].astype(int)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        log_message(f"Error fetch_data_for_strategy {symbol}: {e}")
        return pd.DataFrame()

def execute_trade(symbol, side, SIDE_BUY_with_custom_amount=None):
    """
    We'll unify the buy/sell logic, but if side=BUY,
    we might pass an explicit 'SIDE_BUY_with_custom_amount' to spend
    or else we do the default entire 'trade_capital'.
    For SELL => we only sell from purchased_holdings[symbol], ignoring testnet stash.
    """
    global trade_capital

    minQty, maxQty, stepSize = fetch_symbol_min_max(symbol)
    minNotional = fetch_symbol_min_notional(symbol)
    current_price = get_current_price(symbol)

    try:
        if side == SIDE_BUY:
            if SIDE_BUY_with_custom_amount is not None:
                spend = SIDE_BUY_with_custom_amount
            else:
                spend = trade_capital

            # We do not check testnet USDT stash, ignoring any leftover
            if trade_capital <= 0:
                log_message(f"No capital left for the bot to buy {symbol}.")
                return

            if spend < minNotional:
                log_message(f"Trade amount ${spend:.2f} is below the minimum notional ${minNotional:.2f}. Set to minimal notional.")
                spend = minNotional
                
            if spend > trade_capital:
                log_message(f"Trade amount ${spend:.2f} to buy {symbol} exeeds trade capital ${trade_capital:.2f}. Abort buy.")
                return

            quantity = spend / current_price
            quantity = round_step_size(quantity, stepSize)

            if quantity < minQty:
                log_message(f"quantity {quantity} < minQty {minQty}, abort BUY {symbol}")
                return
            if quantity > maxQty:
                quantity = maxQty

            # place real order on testnet
            order = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            fill_price = float(order['fills'][0]['price'])
            total_cost = quantity * fill_price

            trade_capital -= total_cost
            if trade_capital < 0:
                trade_capital = 0

            last_buy_price[symbol] = fill_price

            # Update purchased_holdings
            purchased_holdings[symbol] = purchased_holdings.get(symbol, 0.0) + quantity

            tstamp = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO trades (symbol, side, quantity, price, total_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, side, quantity, fill_price, total_cost, tstamp)
            )
            conn.commit()

            update_wallet_display()
            add_coin_trade_history(symbol, "BUY", quantity, fill_price, total_cost)
            log_message(f"Buy Executed: {order}")

        elif side == SIDE_SELL:
            # We only SELL from purchased_holdings, ignoring testnet stash
            current_bot_qty = purchased_holdings.get(symbol, 0.0)
            if current_bot_qty <= 0:
                log_message(f"No BOT-owned holdings to sell for {symbol}.")
                return

            quantity = round_step_size(current_bot_qty, stepSize)
            if quantity < minQty:
                log_message(f"quantity {quantity} < minQty {minQty}, abort SELL {symbol}")
                return
            if quantity > maxQty:
                quantity = maxQty

            order = client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )

            fill_price = float(order['fills'][0]['price'])
            total_revenue = quantity * fill_price
            trade_capital += total_revenue

            # reduce purchased_holdings
            purchased_holdings[symbol] = max(0.0, current_bot_qty - quantity)

            if symbol in last_buy_price:
                del last_buy_price[symbol]

            tstamp = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO trades (symbol, side, quantity, price, total_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, side, quantity, fill_price, total_revenue, tstamp)
            )
            conn.commit()

            update_wallet_display()
            add_coin_trade_history(symbol, "SELL", quantity, fill_price, total_revenue)
            log_message(f"Sell Executed: {order}")

    except Exception as e:
        log_message(f"Error in execute_trade for {symbol} side={side}: {e}")

# ========== Strategy Functions ==========
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
    ep = data['high'].iloc[0]
    af_current = af
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

def predict_profit(symbol, data, strategy_name, params):
    """
    We'll run the selected strategy to see if it says BUY/SELL/HOLD.
    Then we convert that to a numerical score:
      - If strategy says BUY => +some positive
      - If strategy says SELL => negative
      - If HOLD => 0 or small positive
    We can refine it further by factoring in e.g. how close RSI is to oversold/overbought, etc.
    """
    # Run the strategy
    if strategy_name == "SMA":
        signal = sma_strategy(data)
    elif strategy_name == "RSI":
        period = int(params.get("RSI_period", 14))
        overbought = float(params.get("RSI_overbought", 70))
        oversold = float(params.get("RSI_oversold", 30))
        signal = rsi_strategy(data, period=period, overbought=overbought, oversold=oversold)
    elif strategy_name == "MACD":
        short_w = params.get("MACD_short", 12)
        long_w = params.get("MACD_long", 26)
        sig_w = params.get("MACD_signal", 9)
        signal = macd_strategy(data, short_window=short_w, long_window=long_w, signal_window=sig_w)
    elif strategy_name == "Bollinger":
        signal = bollinger_strategy(data)
    elif strategy_name == "Stochastic":
        signal = stochastic_strategy(data)
    elif strategy_name == "Parabolic SAR":
        signal = psar_strategy(data)
    else:
        signal = "HOLD"

    # Convert to score
    if signal == "BUY":
        score = +1.0
    elif signal == "SELL":
        score = -1.0
    else:
        score = 0.0

    return score

def monitor_and_trade_top_altcoins(interval, request_freq):
    """
    Main loop: 
    - read strategy + params
    - fetch top altcoins
    - compute score (predict_profit)
    - if negative => SELL, if positive => BUY (split capital), if 0 => hold
    - update tab (which also runs stop-loss check)
    - display profit by summing purchased_holdings + trade_capital
    """
    global stop_trading

    strategy_name = get_value("strategy_dropdown")
    log_message(f"Starting {BOT_NAME} with multi-coin approach, strategy={strategy_name}")

    try:
        while not stop_trading:
            update_wallet_display()

            # read or refresh user-defined stop_loss_ratio
            try:
                user_sl = float(get_value("stop_loss_input"))
            except:
                user_sl = 10.0
            global stop_loss_ratio
            stop_loss_ratio = (user_sl / 100.0)

            # read strategy param fields
            params = {
                "RSI_period": int(get_value("rsi_period_input")),
                "RSI_oversold": float(get_value("rsi_oversold_input")),
                "RSI_overbought": float(get_value("rsi_overbought_input")),
                "MACD_short": int(get_value("macd_short_input")),
                "MACD_long": int(get_value("macd_long_input")),
                "MACD_signal": int(get_value("macd_signal_input")),
            }

            # fetch top altcoins
            top_altcoins = get_top_altcoins()
            log_message(f"Coins to monitor: {top_altcoins}")

            coin_scores = []
            for symbol in top_altcoins:
                initialize_coin_tab(symbol)
                update_coin_tab(symbol)  # triggers stop-loss if needed

                df = fetch_data_for_strategy(symbol, interval)
                if df.empty or "close" not in df.columns:
                    continue

                sc = predict_profit(symbol, df, strategy_name, params)
                coin_scores.append((symbol, sc))

            coin_scores.sort(key=lambda x: x[1], reverse=True)

            if not coin_scores:
                time.sleep(request_freq)
                continue

            # SELL for negative
            negative_coins = [c for c in coin_scores if c[1] < 0]
            for (sym, score) in negative_coins:
                execute_trade(sym, SIDE_SELL)

            # BUY for positive
            positive_coins = [c for c in coin_scores if c[1] > 0]
            sum_scores = sum([p[1] for p in positive_coins])
            if sum_scores <= 0:
                log_message("No coin has a strictly positive prediction. Skipping buy.")
            else:
                global trade_capital
                local_cap = trade_capital
                if local_cap > 1e-6:
                    ratio_list = []
                    for (sym, sc) in positive_coins:
                        ratio = sc / sum_scores
                        ratio_list.append((sym, ratio, sc))
                    for (sym, ratio, sc) in ratio_list:
                        buy_amount = local_cap * ratio
                        if buy_amount < 1e-6:
                            continue
                        execute_trade(sym, SIDE_BUY, SIDE_BUY_with_custom_amount=buy_amount)

            time.sleep(request_freq)

    except Exception as e:
        log_message(f"Error in main trading loop: {e}")

    log_message(f"Stopped {BOT_NAME}.")
    c.execute(
        "INSERT INTO performance (strategy, profit, runtime, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
        (get_value("strategy_dropdown"), 0, "", str(datetime.datetime.now()), str(datetime.datetime.now()))
    )
    conn.commit()

# ========== UI & STARTUP ==========

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
    log_message("Stop signal sent. Bot will stop soon...")

def start_trading_callback(sender, data):
    start_trading_thread()

def stop_trading_callback(sender, data):
    stop_trading_thread()

def show_strategy_descriptions():
    with window(label="Strategy Descriptions", modal=True, no_close=True, popup=True) as desc_window:
        add_text("Strategies: SMA, RSI, MACD, etc.\n"
                 "SELL if strategy < 0 or stop-loss triggers.\n"
                 "Only purchased holdings are used (testnet stash ignored)."
                 "\nProfit calculates the value of purchased coins + trade capital.")
        add_button(label="Close", callback=lambda s,a: delete_item(desc_window))

create_context()
create_viewport(title=BOT_NAME, width=1500, height=900)

with window(label=BOT_NAME, width=1500, height=900):
    with menu_bar():
        with menu(label="Help"):
            add_menu_item(label="Strategy Descriptions", callback=show_strategy_descriptions)

    add_text(f"{BOT_NAME} - Ignores Testnet Stash, Tracks Only Bot-Owned Coins", color=[255, 255, 0])
    add_separator()

    # Config child
    with child_window(tag="config_child", width=400, height=200, border=True):
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

        add_button(label="Start Trading", callback=start_trading_callback)
        add_button(label="Stop Trading", callback=stop_trading_callback)

    # Strategy Fine-tuning
    with child_window(tag="strategy_tune_child", width=400, height=300, border=True, show=False):
        add_text("Strategy Fine-Tune Params", color=[255, 128, 0])

        # RSI
        add_text("RSI Params:")
        add_input_text(label="Period", default_value="14", tag="rsi_period_input", width=60)
        add_input_text(label="Overbought", default_value="70", tag="rsi_overbought_input", width=60)
        add_input_text(label="Oversold", default_value="30", tag="rsi_oversold_input", width=60)
        add_separator()

        # MACD
        add_text("MACD Params:")
        add_input_text(label="Short Window", default_value="12", tag="macd_short_input", width=60)
        add_input_text(label="Long Window", default_value="26", tag="macd_long_input", width=60)
        add_input_text(label="Signal Window", default_value="9", tag="macd_signal_input", width=60) 

    # Wallet area
    with child_window(tag="wallet_child", width=400, height=130, border=True):
        add_text("Wallet & Bot Profit", color=[0, 255, 255])
        add_text("Binance USDT (Testnet): $0.00", tag="wallet_balance_text")
        add_text("Trade Money: $0.00", tag="trade_capital_text")
        add_text("Profit: 0.00%", tag="profit_text")
        add_text("Assets: $0.00", tag="total_asset")

    # coin tabs
    with child_window(tag="coin_tabs_child", width=-1, height=300, border=True):
        add_text("Coin Tabs (Ignoring Testnet Stash)", color=[255, 255, 0])
        with tab_bar(tag="coin_tab_bar"):
            pass

    # logs
    with child_window(tag="logs_child", width=-1, height=-1, border=True):
        add_text("Logs", color=[255, 0, 255])
        add_input_text(
            tag="log_multiline",
            multiline=True,
            default_value="--- Logs ---",
            width=-1,
            height=-1,
            readonly=True
        )

setup_dearpygui()
show_viewport()
start_dearpygui()
destroy_context()
