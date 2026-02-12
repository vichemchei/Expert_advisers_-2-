import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\XM Global MT5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Bot Parameters ---------
symbols = ["GOLD#", "XAUEUR#"]
risk_per_trade = 0.02  # 2% risk per trade
stop_loss_pips = 80
take_profit_pips = 100
rsi_period = 14
ema_period = 200
stoch_k_period = 14
stoch_d_period = 3
stoch_slowing = 3
check_interval = 60  # Check for signals every 60 seconds
timeframe = mt5.TIMEFRAME_H1

# --------- Trailing Stop Parameters ---------
ENABLE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_PIPS = 30  # Start trailing after 30 pips profit
TRAILING_STOP_DISTANCE_PIPS = 20    # Keep stop 20 pips behind current price
TRAILING_CHECK_INTERVAL = 10        # Check trailing stop every 10 seconds


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


def calculate_ema(prices, period):
    """Calculate EMA"""
    series = pd.Series(prices)
    ema = series.ewm(span=period, adjust=False).mean()
    return ema.iloc[-1]


def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3):
    """Calculate Stochastic Oscillator"""
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)
    close_series = pd.Series(closes)
    
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    k_percent_slowed = k_percent.rolling(window=slowing).mean()
    d_percent = k_percent_slowed.rolling(window=d_period).mean()
    
    return k_percent_slowed.iloc[-1], d_percent.iloc[-1]


def get_current_data(symbol, timeframe, bars=250):
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
    
    # Get pip value
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    pip_value_per_lot = tick_value * (0.1 / tick_size)
    
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    
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


def modify_position_sl_tp(position, new_sl, new_tp=None):
    """Modify stop loss and take profit of an open position"""
    symbol = position.symbol
    ticket = position.ticket
    
    # If new_tp is None, keep the existing TP
    if new_tp is None:
        new_tp = position.tp
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": new_tp,
        "magic": 234000,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"    Failed to modify SL: {result.comment}")
        return False
    
    return True


def update_trailing_stop(position):
    """Update trailing stop for a position"""
    if not ENABLE_TRAILING_STOP:
        return False
    
    symbol = position.symbol
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        return False
    
    point = symbol_info.point
    tick = mt5.symbol_info_tick(symbol)
    
    if tick is None:
        return False
    
    # Get current price based on position type
    if position.type == mt5.ORDER_TYPE_BUY:
        current_price = tick.bid
        entry_price = position.price_open
        current_sl = position.sl
        
        # Calculate profit in pips
        profit_pips = (current_price - entry_price) / point
        
        # Check if trailing should be activated
        if profit_pips >= TRAILING_STOP_ACTIVATION_PIPS:
            # Calculate new stop loss
            new_sl = current_price - (TRAILING_STOP_DISTANCE_PIPS * point)
            
            # Only update if new SL is higher than current SL
            if new_sl > current_sl:
                success = modify_position_sl_tp(position, new_sl)
                if success:
                    print(f"    [{symbol}] Trailing SL updated: {current_sl:.5f} -> {new_sl:.5f} (Profit: {profit_pips:.1f} pips)")
                    return True
    
    elif position.type == mt5.ORDER_TYPE_SELL:
        current_price = tick.ask
        entry_price = position.price_open
        current_sl = position.sl
        
        # Calculate profit in pips
        profit_pips = (entry_price - current_price) / point
        
        # Check if trailing should be activated
        if profit_pips >= TRAILING_STOP_ACTIVATION_PIPS:
            # Calculate new stop loss
            new_sl = current_price + (TRAILING_STOP_DISTANCE_PIPS * point)
            
            # Only update if new SL is lower than current SL
            if new_sl < current_sl:
                success = modify_position_sl_tp(position, new_sl)
                if success:
                    print(f"    [{symbol}] Trailing SL updated: {current_sl:.5f} -> {new_sl:.5f} (Profit: {profit_pips:.1f} pips)")
                    return True
    
    return False


def manage_trailing_stops():
    """Check and update trailing stops for all open positions"""
    positions = get_open_positions()
    
    if not positions:
        return
    
    for position in positions:
        update_trailing_stop(position)


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
        "comment": "RSI-Stoch-EMA Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return None
    
    return result


def check_signal(symbol):
    """Check for buy/sell signals using RSI, Stochastic, and EMA"""
    data_needed = max(ema_period, stoch_k_period, rsi_period) + 50
    df = get_current_data(symbol, timeframe, bars=data_needed)
    
    if df is None or len(df) < data_needed:
        return None
    
    # Calculate indicators
    rsi = calculate_rsi(df['close'].values, rsi_period)
    ema = calculate_ema(df['close'].values, ema_period)
    stoch_k, stoch_d = calculate_stochastic(
        df['high'].values, df['low'].values, df['close'].values,
        stoch_k_period, stoch_d_period, stoch_slowing
    )
    current_price = df['close'].iloc[-1]
    
    # BUY Signal: Stoch %K < 20 AND RSI > 50 AND Price > EMA200
    if stoch_k < 20 and rsi > 50 and current_price > ema:
        return "buy"
    
    # SELL Signal: Stoch %K > 80 AND RSI < 50 AND Price < EMA200
    elif stoch_k > 80 and rsi < 50 and current_price < ema:
        return "sell"
    
    return None


def display_positions_status():
    """Display current positions with profit information"""
    positions = get_open_positions()
    
    if not positions:
        return
    
    print("\n  --- Open Positions ---")
    for position in positions:
        symbol = position.symbol
        position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        entry_price = position.price_open
        current_sl = position.sl
        current_tp = position.tp
        profit = position.profit
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                point = symbol_info.point
                if position.type == mt5.ORDER_TYPE_BUY:
                    pips = (current_price - entry_price) / point
                else:
                    pips = (entry_price - current_price) / point
                
                print(f"  [{symbol}] {position_type} | Entry: {entry_price:.5f} | Current: {current_price:.5f} | "
                      f"Profit: ${profit:.2f} ({pips:.1f} pips) | SL: {current_sl:.5f} | TP: {current_tp:.5f}")
    print()


def run_bot():
    """Main bot loop"""
    print("=" * 70)
    print("RSI-STOCHASTIC-EMA TRADING BOT - LIVE MODE")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"Timeframe: H1 (1-hour charts)")
    print(f"\nIndicators:")
    print(f"  - RSI Period: {rsi_period}")
    print(f"  - EMA Period: {ema_period}")
    print(f"  - Stochastic: K={stoch_k_period}, D={stoch_d_period}, Slowing={stoch_slowing}")
    print(f"\nStrategy Logic:")
    print(f"  BUY: Stoch %K < 20 AND RSI > 50 AND Price > EMA{ema_period}")
    print(f"  SELL: Stoch %K > 80 AND RSI < 50 AND Price < EMA{ema_period}")
    print(f"\nTrailing Stop Settings:")
    print(f"  - Enabled: {ENABLE_TRAILING_STOP}")
    print(f"  - Activation: {TRAILING_STOP_ACTIVATION_PIPS} pips profit")
    print(f"  - Distance: {TRAILING_STOP_DISTANCE_PIPS} pips behind price")
    print(f"  - Check Interval: {TRAILING_CHECK_INTERVAL} seconds")
    print(f"\nSignal Check Interval: {check_interval} seconds")
    print("=" * 70)
    print("\nBot started. Press Ctrl+C to stop.\n")
    
    last_signal_check = time.time()
    last_trailing_check = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Check and update trailing stops more frequently
            if current_time - last_trailing_check >= TRAILING_CHECK_INTERVAL:
                manage_trailing_stops()
                last_trailing_check = current_time
            
            # Check for new signals at regular intervals
            if current_time - last_signal_check >= check_interval:
                account_info = mt5.account_info()
                if account_info is None:
                    print("Failed to get account info")
                    last_signal_check = current_time
                    time.sleep(5)
                    continue
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
                
                # Display current positions
                display_positions_status()
                
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
                last_signal_check = current_time
    
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print("MT5 connection closed")


if __name__ == "__main__":
    run_bot()