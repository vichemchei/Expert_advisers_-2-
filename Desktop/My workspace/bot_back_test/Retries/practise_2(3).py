import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import gc


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Bot Parameters ---------
symbols = ["XAUUSD", "XAUEUR"]
risk_per_trade = 0.20  # 20% risk per trade
stop_loss_pips = 100
take_profit_pips = 120
rsi_period = 14
ma_period = 10
check_interval = 15  # Check for signals every 60 seconds
timeframe = mt5.TIMEFRAME_M5

# --------- Risk Control Variables ---------
loss_streak = 0
max_loss_streak = 4
cooldown_period = 900  #  15 minutes in seconds
cooldown_until = None
bot_start_time = None

# Simple tracking
open_tickets = set()  # Currently open position tickets
closed_trades = []  # List of closed trades with profit
last_balance = None  # Track balance between cycles

# File to persist trade history
TRADES_FILE = "bot_trades.json"


# --------- Helper Functions ---------
def load_trade_history():
    """Load trade history from file"""
    global closed_trades, loss_streak
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                data = json.load(f)
                closed_trades = data.get('closed_trades', [])
            print(f"[INIT] Loaded {len(closed_trades)} previous trades")
            calculate_loss_streak()
        except Exception as e:
            print(f"[INIT] Error loading trade history: {e}")
            closed_trades = []
        finally:
            gc.collect()


def save_trade_history():
    """Save trade history to file"""
    try:
        with open(TRADES_FILE, 'w') as f:
            json.dump({'closed_trades': closed_trades}, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save trade history: {e}")


def delete_trade_history():
    """Delete the trade history file"""
    global closed_trades, loss_streak
    closed_trades = []
    loss_streak = 0
    gc.collect()
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(TRADES_FILE):
                time.sleep(0.2 * (attempt + 1))
                os.remove(TRADES_FILE)
                print(f"[CLEANUP] Trade history file deleted - Loss streak reset to 0")
                return
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"[ERROR] Failed to delete trade history after {max_attempts} attempts: {e}")
            else:
                time.sleep(0.3)


def calculate_rsi(prices, period=14):
    """RSI calculation matching TradingView (Wilder's method)"""
    prices = pd.Series(prices)
    delta = prices.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def calculate_ma(prices, period=10):
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
    
    pip_value_per_standard_lot = 10.0
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value_per_standard_lot)
    
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    
    return round(lot, 2)


def get_open_positions(symbol=None):
    """Get all open positions"""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return list(positions) if positions else []


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
    bars_needed = max(rsi_period, ma_period) + 50
    df = get_current_data(symbol, timeframe, bars=bars_needed)
    
    if df is None or len(df) < max(rsi_period, ma_period) + 1:
        return None
    
    rsi = calculate_rsi(df['close'].values, rsi_period)
    ma = calculate_ma(df['close'].values, ma_period)
    current_price = df['close'].iloc[-1]
    
    if rsi > 50 and current_price > ma:
        return "buy"
    elif rsi < 50 and current_price < ma:
        return "sell"
    
    return None


def track_closed_positions():
    """Check for closed positions and record their profit/loss"""
    global open_tickets, closed_trades, last_balance
    
    account_info = mt5.account_info()
    if account_info is None:
        return
    
    current_balance = account_info.balance
    current_positions = get_open_positions()
    current_tickets = {pos.ticket for pos in current_positions}
    
    new_tickets = current_tickets - open_tickets
    for ticket in new_tickets:
        print(f"[TRACK] New position opened: #{ticket}")
    
    closed_tickets = open_tickets - current_tickets
    
    if closed_tickets and last_balance is not None:
        balance_change = current_balance - last_balance
        
        for ticket in closed_tickets:
            if len(closed_tickets) == 1:
                profit = balance_change
            else:
                profit = balance_change / len(closed_tickets)
            
            trade = {
                'ticket': ticket,
                'profit': profit,
                'close_time': datetime.now().isoformat(),
                'balance_after': current_balance
            }
            
            closed_trades.append(trade)
            
            result_str = "WIN" if profit > 0 else "LOSS"
            print(f"[CLOSED] Position #{ticket} - {result_str}: ${profit:.2f}")
        
        calculate_loss_streak()
        save_trade_history()
    
    open_tickets = current_tickets.copy()
    last_balance = current_balance


def calculate_loss_streak():
    """Calculate consecutive losses from closed trades"""
    global loss_streak, cooldown_until
    
    if len(closed_trades) == 0:
        loss_streak = 0
        return
    
    streak = 0
    for trade in reversed(closed_trades):
        if trade['profit'] < 0:
            streak += 1
        else:
            break
    
    old_streak = loss_streak
    loss_streak = streak
    
    if old_streak != loss_streak:
        print(f"[STREAK] Loss streak: {old_streak} -> {loss_streak}")
    
    if loss_streak >= max_loss_streak:
        if cooldown_until is None:
            cooldown_until = datetime.now().timestamp() + cooldown_period
            print(f"\n{'='*70}")
            print(f"COOLDOWN ACTIVATED: {loss_streak} consecutive losses")
            print(f"Trading paused until {datetime.fromtimestamp(cooldown_until).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")
            delete_trade_history()
    else:
        if cooldown_until is not None:
            print(f"\n{'='*70}")
            print(f"COOLDOWN ENDED: Loss streak broken ({loss_streak} losses)")
            print(f"{'='*70}\n")
            cooldown_until = None


def is_trading_allowed():
    """Check if trading is allowed"""
    global cooldown_until, loss_streak
    
    if cooldown_until is None:
        return True
    
    current_time = datetime.now().timestamp()
    
    if current_time >= cooldown_until:
        print(f"\n{'='*70}")
        print(f"COOLDOWN PERIOD ENDED")
        print(f"Loss streak reset from {loss_streak} to 0 - Resuming trading")
        print(f"{'='*70}\n")
        cooldown_until = None
        loss_streak = 0
        return True
    
    return False


def run_bot():
    """Main bot loop"""
    global bot_start_time, last_balance, open_tickets
    
    bot_start_time = datetime.now()
    load_trade_history()
    
    account_info = mt5.account_info()
    if account_info:
        last_balance = account_info.balance
        
    current_positions = get_open_positions()
    open_tickets = {pos.ticket for pos in current_positions}
    
    print("=" * 70)
    print("RSI-MA TRADING BOT - LIVE MODE")
    print("=" * 70)
    print(f"Bot Start Time: {bot_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"RSI Period: {rsi_period} | MA Period: {ma_period}")
    print(f"Check Interval: {check_interval} seconds")
    print(f"Max Loss Streak: {max_loss_streak} trades (Cooldown: {cooldown_period/60:.0f} minutes)")
    print(f"Starting Balance: ${last_balance:.2f}")
    print(f"Current Loss Streak: {loss_streak}")
    print(f"Open Positions: {len(open_tickets)}")
    print("=" * 70)
    print("\nBot started. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                time.sleep(check_interval)
                continue
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f} | Streak: {loss_streak} | Open: {len(open_tickets)}")
            
            track_closed_positions()
            
            if not is_trading_allowed():
                if cooldown_until:
                    remaining = int(cooldown_until - datetime.now().timestamp())
                    print(f"  COOLDOWN: {remaining//60}m {remaining%60}s remaining\n")
                time.sleep(check_interval)
                continue
            
            for symbol in symbols:
                positions = get_open_positions(symbol)
                
                if len(positions) > 0:
                    print(f"  {symbol}: Position open")
                    continue
                
                signal = check_signal(symbol)
                
                if signal:
                    lot_size = calculate_lot_size(symbol, risk_per_trade, stop_loss_pips)
                    print(f"  {symbol}: {signal.upper()} signal - Opening {lot_size} lots")
                    
                    result = open_position(symbol, signal, lot_size, stop_loss_pips, take_profit_pips)
                    
                    if result:
                        print(f"    SUCCESS: Ticket #{result.order}")
                        open_tickets.add(result.order)
                    else:
                        print(f"    FAILED")
                else:
                    print(f"  {symbol}: No signal")
            
            print()
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
        save_trade_history()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print("MT5 connection closed")


if __name__ == "__main__":
    run_bot()
