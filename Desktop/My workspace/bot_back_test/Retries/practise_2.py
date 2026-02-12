import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Bot Parameters ---------
symbols = ["XAUUSD", "XAUEUR"]
risk_per_trade = 0.20  # 2% risk per trade
stop_loss_pips = 80
take_profit_pips = 100
rsi_period = 14
ma_period = 50
check_interval = 60  # Check for signals every 60 seconds
timeframe = mt5.TIMEFRAME_M5


# --------- Helper Functions ---------
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    series = pd.Series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_ma(prices, period= 50):
    """Calculate Moving Average"""
    return pd.Series(prices).rolling(window=period).mean().iloc[-1]


def get_current_data(symbol, timeframe, bars=100):
    """Fetch recent data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_lot_size(symbol, risk_pct, sl_pips):
    """Calculate position size based on risk"""
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    
    balance = account_info.balance
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        return 0.01
    
    # For gold: 1 standard lot = $10 per pip
    pip_value_per_standard_lot = 10.0
    
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value_per_standard_lot)
    
    # Apply constraints
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    
    return round(lot, 2)


def get_open_positions(symbol=None):
    """Get all open positions, optionally filtered by symbol"""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    
    if positions is None:
        return []
    
    return list(positions)


def close_position(position):
    """Close an open position"""
    symbol = position.symbol
    ticket = position.ticket
    lot = position.volume
    position_type = position.type
    
    # Determine closing order type
    if position_type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Bot close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    return result


def open_position(symbol, order_type, lot_size, sl_pips, tp_pips):
    """Open a new position"""
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        return None
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            return None
    
    point = symbol_info.point
    
    if order_type == "buy":
        price = mt5.symbol_info_tick(symbol).ask
        sl = price - sl_pips * point
        tp = price + tp_pips * point
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    else:
        price = mt5.symbol_info_tick(symbol).bid
        sl = price + sl_pips * point
        tp = price - tp_pips * point
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type_mt5,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "RSI-MA Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return None
    
    return result


def check_signal(symbol):
    """Check for buy/sell signals"""
    df = get_current_data(symbol, timeframe, bars=max(rsi_period, ma_period) + 20)
    
    if df is None or len(df) < max(rsi_period, ma_period):
        return None
    
    # Calculate indicators
    rsi = calculate_rsi(df['close'].values, rsi_period)
    ma = calculate_ma(df['close'].values, ma_period)
    current_price = df['close'].iloc[-1]
    
    # Check for signals
    if rsi > 50 and current_price > ma:
        return "buy"
    elif rsi < 50 and current_price < ma:
        return "sell"
    
    return None


def run_bot():
    """Main bot loop"""
    print("=" * 70)
    print("RSI-MA TRADING BOT - LIVE MODE")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"RSI Period: {rsi_period} | MA Period: {ma_period}")
    print(f"Check Interval: {check_interval} seconds")
    print("=" * 70)
    print("\nBot started. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                time.sleep(check_interval)
                continue
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
            
            for symbol in symbols:
                # Check if position already exists for this symbol
                positions = get_open_positions(symbol)
                
                if len(positions) > 0:
                    print(f"  {symbol}: Position already open (Skip)")
                    continue
                
                # Check for signals
                signal = check_signal(symbol)
                
                if signal:
                    lot_size = calculate_lot_size(symbol, risk_per_trade, stop_loss_pips)
                    
                    print(f"  {symbol}: {signal.upper()} signal detected - Opening {lot_size} lots")
                    
                    result = open_position(symbol, signal, lot_size, stop_loss_pips, take_profit_pips)
                    
                    if result:
                        print(f"    Order executed: Ticket #{result.order}")
                    else:
                        print(f"    Order failed")
                else:
                    print(f"  {symbol}: No signal")
            
            print()
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        mt5.shutdown()
        print("MT5 connection closed")


if __name__ == "__main__":
    run_bot()