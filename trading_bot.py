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

API_KEY = "6cGga0XFJingehbxNNOmuqyFn0YJtuJKZcrkxNAsFLzLGr0wjmE89h0l3EcYgz8O"
API_SECRET = "IKEsFHGnQmUvzPUhGiWQAWZTzGrHBY3ixvic74s0n3d6jlzauja6beR0b5nOSpzS"
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

# =================== GLOBAL VARIABLES ===================
stop_trading = False           # Flag to stop trading thread

# Each symbol -> how many base tokens we hold (for logic/tracking)
holdings = {}

# For showing coin tabs
coin_tabs = {}

# We'll track these for top altcoins
_cached_top_altcoins = []
_last_top_altcoins_fetch_time = 0
_TOP_ALTCOINS_FETCH_INTERVAL = 300  # 5 minutes

# local capital for the bot to spend across all trades (set from UI)
trade_capital = 50.0
initial_trade_capital = 50.0  # so we can measure profit
profit_percent = 0.0

# =================== HELPER FUNCTIONS ===================
def log_message(msg):
    current_logs = get_value("log_multiline")
    new_logs = f"{current_logs}\n{msg}"
    set_value("log_multiline", new_logs)
    print(msg)

def get_usdt_balance():
    """
    Return real USDT balance from your Binance account.
    If not found or error, return 0.0
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

def update_wallet_display():
    """
    Show real USDT from Binance, plus our local trade capital's profit in percent.
    """
    usdt_balance = get_usdt_balance()
    set_value("wallet_balance_text", f"Wallet Balance (actual USDT): ${usdt_balance:.2f}")

    # Calculate profit % relative to initial_trade_capital
    if initial_trade_capital > 0:
        current_profit = ((trade_capital - initial_trade_capital) / initial_trade_capital) * 100.0
    else:
        current_profit = 0.0
    set_value("profit_text", f"Profit: {current_profit:.2f}%")

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
    if symbol in coin_tabs:
        return
    with tab(label=symbol, parent="coin_tab_bar") as tab_id:
        coin_tabs[symbol] = tab_id

        add_text(f"Coin: {symbol}", color=[255, 255, 0])
        add_separator()
        add_text(f"Quantity Held: 0", tag=f"{symbol}_quantity_text")
        add_separator()

        add_text(f"{symbol} Current Price:", color=[0, 255, 0])
        add_text("Price: ???", tag=f"{symbol}_price_text")
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

def update_coin_tab(symbol):
    """
    Display the coin's quantity from `holdings`,
    the latest price from Binance, and the last update time.
    """
    if symbol not in coin_tabs:
        return

    qty_text = f"Quantity Held: {holdings.get(symbol, 0.0):.6f}"
    configure_item(f"{symbol}_quantity_text", default_value=qty_text)

    try:
        ticker_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        configure_item(f"{symbol}_price_text", default_value=f"Price: {ticker_price:.6f}")
        configure_item(f"{symbol}_timestamp_text", default_value=f"Last Update: {now_str}")
    except Exception as e:
        log_message(f"Error fetching current price for {symbol}: {e}")

def add_coin_trade_history(symbol, side, quantity, price):
    history_tag = f"{symbol}_history_multiline"
    old_text = get_value(history_tag)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_line = f"{timestamp_str} - {side} {quantity:.6f} @ {price:.6f}\n"
    set_value(history_tag, old_text + new_line)

def fetch_symbol_min_max(symbol):
    """
    Return (minQty, maxQty, stepSize) from the symbol's LOT_SIZE filter.
    """
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                return (float(f["minQty"]), float(f["maxQty"]), float(f["stepSize"]))
    except Exception as e:
        log_message(f"Error fetching symbol info for {symbol}: {e}")
    # fallback
    return (0.000001, 999999999, 0.000001)

def round_step_size(quantity, stepSize):
    if stepSize <= 0:
        return quantity
    precision = int(round(-math.log10(stepSize), 0))
    return round(quantity, precision)

def execute_trade(symbol, side):
    """
    We'll read the global trade_capital as the total capital we have left for trading.
    For a BUY:
      - We won't exceed `trade_capital`.
      - Actually use binance wallet for real trades.
    For a SELL:
      - We SELL all local holdings for that symbol.
    """
    global trade_capital

    initialize_coin_tab(symbol)

    minQty, maxQty, stepSize = fetch_symbol_min_max(symbol)

    try:
        ticker_price = float(client.get_symbol_ticker(symbol=symbol)['price'])

        if side == SIDE_BUY:
            # We only can spend up to 'trade_capital', but also must check real USDT wallet
            usdt_balance = get_usdt_balance()
            actual_spend = min(trade_capital, usdt_balance)
            if actual_spend <= 0:
                log_message(f"No capital or USDT to buy {symbol}.")
                return

            quantity = actual_spend / ticker_price
            quantity = round_step_size(quantity, stepSize)

            if quantity < minQty:
                log_message(f"Calc quantity {quantity} < minQty {minQty} for {symbol}. Abort BUY.")
                return
            if quantity > maxQty:
                quantity = maxQty

            order = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )

            fill_price = float(order['fills'][0]['price'])
            total_cost = quantity * fill_price

            # Decrease trade_capital by the cost
            trade_capital -= total_cost
            if trade_capital < 0:
                trade_capital = 0  # can't go negative

            # Update local holdings
            holdings[symbol] = holdings.get(symbol, 0.0) + quantity

            # DB logging
            tstamp = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO trades (symbol, side, quantity, price, total_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, side, quantity, fill_price, total_cost, tstamp)
            )
            conn.commit()

            update_wallet_display()
            add_coin_trade_history(symbol, "BUY", quantity, fill_price)
            log_message(f"Buy Executed: {order}")

        elif side == SIDE_SELL:
            current_qty = holdings.get(symbol, 0.0)
            if current_qty <= 0:
                log_message(f"No holdings to sell for {symbol}.")
                return

            quantity = round_step_size(current_qty, stepSize)
            if quantity < minQty:
                log_message(f"Calc quantity {quantity} < minQty {minQty} for {symbol}. Abort SELL.")
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

            # Increase trade_capital by the revenue
            trade_capital += total_revenue

            holdings[symbol] = 0.0

            # DB logging
            tstamp = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO trades (symbol, side, quantity, price, total_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, side, quantity, fill_price, total_revenue, tstamp)
            )
            conn.commit()

            update_wallet_display()
            add_coin_trade_history(symbol, "SELL", quantity, fill_price)
            log_message(f"Sell Executed: {order}")

    except Exception as e:
        log_message(f"Error in execute_trade for {symbol} side={side}: {e}")

# =================== TRADING STRATEGIES ===================
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
    if len(data) < k_period + d_period:
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

def monitor_and_trade_top_altcoins(interval, strategy_name, request_freq):
    """
    Main loop that fetches top altcoins, updates prices, runs strategies, does buys/sells.
    'trade_capital' is the total money (in USDT) for the bot to spend across all trades.
    Gains/losses from each trade are added/subtracted.
    """
    global stop_trading
    log_message(f"Starting {BOT_NAME} with strategy: {strategy_name}")

    if strategy_name not in STRATEGIES:
        log_message(f"Strategy {strategy_name} not found.")
        return

    strategy_func = STRATEGIES[strategy_name]

    try:
        while not stop_trading:
            # 1) Update display of real USDT wallet & profit
            update_wallet_display()

            # 2) Fetch top altcoins
            top_altcoins = get_top_altcoins()
            log_message(f"Monitoring: {top_altcoins}")

            # 3) For each symbol, update tab + run strategy
            for symbol in top_altcoins:
                initialize_coin_tab(symbol)
                update_coin_tab(symbol)

                # We fetch a bit of data for strategy
                df = fetch_data_for_strategy(symbol, interval)
                if df.empty or "close" not in df.columns:
                    continue

                # Evaluate strategy
                signal = strategy_func(df)
                if signal == "BUY":
                    execute_trade(symbol, SIDE_BUY)
                    log_message(f"{datetime.datetime.now()} - {symbol} - BUY signal")
                elif signal == "SELL":
                    if holdings.get(symbol, 0.0) > 0:
                        execute_trade(symbol, SIDE_SELL)
                        log_message(f"{datetime.datetime.now()} - {symbol} - SELL signal")
                    else:
                        log_message(f"SELL signal but no holdings for {symbol}")
                else:
                    pass  # HOLD

            time.sleep(request_freq)

    except Exception as e:
        log_message(f"Error in trading loop: {e}")

    # Insert performance record (optional)
    c.execute(
        "INSERT INTO performance (strategy, profit, runtime, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
        (strategy_name, 0, "", str(datetime.datetime.now()), str(datetime.datetime.now()))
    )
    conn.commit()

    log_message(f"{BOT_NAME} stopped with strategy: {strategy_name}")

def fetch_data_for_strategy(symbol, interval, lookback=50):
    """
    Minimal helper to get some klines for the strategy logic.
    We won't store them in coin_data. Just ephemeral for strategy calc.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
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

def start_trading_thread():
    global stop_trading, trade_capital, initial_trade_capital
    stop_trading = False

    interval = get_value("interval_input")
    strategy_name = get_value("strategy_dropdown")
    request_freq = float(get_value("request_freq_input"))

    # read the initial trade capital from UI
    try:
        typed_capital = float(get_value("trade_amount_input"))
    except:
        typed_capital = 50.0

    trade_capital = typed_capital
    initial_trade_capital = typed_capital

    t = threading.Thread(
        target=monitor_and_trade_top_altcoins,
        args=(interval, strategy_name, request_freq),
        daemon=True
    )
    t.start()

def stop_trading_thread():
    global stop_trading
    stop_trading = True
    log_message(f"Stop signal sent. {BOT_NAME} will stop soon...")

def start_trading_callback(sender, app_data):
    start_trading_thread()

def stop_trading_callback(sender, app_data):
    stop_trading_thread()

def show_strategy_descriptions():
    with window(label="Strategy Descriptions", modal=True, no_close=True, popup=True) as desc_window:
        add_text("SMA: Simple Moving Average crossover.\n"
                 "RSI: Buys if RSI < 30, sells if RSI > 70.\n"
                 "MACD: Buys if MACD crosses above signal, sells if below.\n"
                 "Bollinger: Buys if close < lower band, sells if close > upper band.\n"
                 "Stochastic: Looks for %K crossing %D in oversold/overbought.\n"
                 "Parabolic SAR: Buys if PSAR flips below price, sells if flips above.\n")
        add_button(label="Close", callback=lambda s,a: delete_item(desc_window))

# Create the UI
create_context()
create_viewport(title=BOT_NAME, width=1500, height=900)

with window(label=BOT_NAME, width=1500, height=900):
    with menu_bar():
        with menu(label="Help"):
            add_menu_item(label="Strategy Descriptions", callback=show_strategy_descriptions)

    add_text(f"{BOT_NAME} - Single-Capital Bot (No Candle Charts)", color=[255, 255, 0])
    add_separator()

    with child_window(tag="config_child", width=400, height=220, border=True):
        add_text("Trading Configuration", color=[0, 255, 0])
        add_input_text(label="Binance Kline Interval (1m, 5m, etc.)", default_value="1m",
                       tag="interval_input", width=200)
        # This is the total money the bot can spend across all trades
        add_input_text(label="Initial Trade Amount (USD)", default_value="50",
                       tag="trade_amount_input", width=200)
        add_input_text(label="Request Frequency (sec)", default_value="60",
                       tag="request_freq_input", width=200)
        add_combo(
            items=["SMA", "RSI", "MACD", "Bollinger", "Stochastic", "Parabolic SAR"],
            label="Strategy",
            default_value="SMA",
            tag="strategy_dropdown",
            width=200
        )
        add_button(label="Start Trading", callback=start_trading_callback)
        add_button(label="Stop Trading", callback=stop_trading_callback)

    # Show actual binance USDT wallet & the profit %
    with child_window(tag="wallet_child", width=400, height=120, border=True):
        add_text("Wallet & Bot Overview", color=[0, 255, 255])
        add_text("Wallet Balance (actual USDT): $0.00", tag="wallet_balance_text")
        add_text("Profit: 0.00%", tag="profit_text")

    # coin tabs
    with child_window(tag="coin_tabs_child", width=-1, height=350, border=True):
        add_text("Coin Tabs (Price + Last Update + Buy/Sell History)", color=[255, 255, 0])
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
