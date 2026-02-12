import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Bot Parameters ---------
symbols = ["XAUUSD", "XAUEUR","BTCUSD","AUDUSD","GBPUSD"]
risk_per_trade = 0.10  # 10% risk per trade
stop_loss_pips = 80
take_profit_pips = 100
rsi_period = 14
ma_period = 10
check_interval = 10  # Check for signals every 30 seconds
timeframe = mt5.TIMEFRAME_M5

# --------- Risk Control Variables ---------
loss_streak = 0
max_loss_streak = 5
cooldown_period = 1200  # 20 minutes in seconds
cooldown_until = None
bot_start_time = None

# Track open positions and their outcomes
tracked_positions = {}  # {ticket: {'symbol': str, 'type': str, 'open_time': datetime, 'open_price': float}}
closed_trades = []  # List of {'ticket': int, 'profit': float, 'close_time': datetime}

# File to persist trade history
TRADES_FILE = "bot_trades.json"


# --------- Helper Functions ---------
def load_trade_history():
    """Load trade history from file"""
    global closed_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                data = json.load(f)
                closed_trades = data.get('closed_trades', [])
                print(f"  [INIT] Loaded {len(closed_trades)} trades from history file")
        except Exception as e:
            print(f"  [INIT] Error loading trade history: {e}")
            closed_trades = []


def save_trade_history():
    """Save trade history to file"""
    try:
        with open(TRADES_FILE, 'w') as f:
            json.dump({'closed_trades': closed_trades}, f, indent=2)
    except Exception as e:
        print(f"  [ERROR] Failed to save trade history: {e}")


def calculate_rsi(prices, period=14):
    """Calculate RSI using Wilder's smoothing (matches TradingView)"""
    if len(prices) < period + 1:
        return 50.0
    
    prices = pd.Series(prices)
    deltas = prices.diff()
    
    # Separate gains and losses
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    
    # First average - simple average
    avg_gain = gains.iloc[1:period+1].mean()
    avg_loss = losses.iloc[1:period+1].mean()
    
    # Subsequent averages - Wilder's smoothing
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


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
    # Get more bars to ensure accurate RSI calculation
    bars_needed = max(rsi_period, ma_period) + 50
    df = get_current_data(symbol, timeframe, bars=bars_needed)
    
    if df is None or len(df) < max(rsi_period, ma_period) + 1:
        return None
    
    # Calculate indicators
    rsi = calculate_rsi(df['close'].values, rsi_period)
    ma = calculate_ma(df['close'].values, ma_period)
    current_price = df['close'].iloc[-1]
    
    # Debug output
    print(f"    RSI: {rsi:.2f} | MA: {ma:.2f} | Price: {current_price:.2f}")
    
    # Check for signals
    if rsi > 50 and current_price > ma:
        return "buy"
    elif rsi < 50 and current_price < ma:
        return "sell"
    
    return None


def track_positions():
    """Track currently open positions and detect closed ones"""
    global tracked_positions, closed_trades, loss_streak
    
    # Get current open positions
    current_positions = get_open_positions()
    current_tickets = {pos.ticket for pos in current_positions}
    
    # Add new positions to tracking
    for pos in current_positions:
        if pos.ticket not in tracked_positions:
            tracked_positions[pos.ticket] = {
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'open_time': datetime.fromtimestamp(pos.time),
                'open_price': pos.price_open,
                'volume': pos.volume
            }
            print(f"  [TRACK] New position tracked: Ticket #{pos.ticket} ({tracked_positions[pos.ticket]['type']} {pos.symbol})")
    
    # Detect closed positions
    tracked_tickets = set(tracked_positions.keys())
    closed_tickets = tracked_tickets - current_tickets
    
    if closed_tickets:
        for ticket in closed_tickets:
            # Try to get the profit from account history
            # Since history doesn't work, we'll estimate based on current balance change
            # or mark as unknown
            closed_trade = {
                'ticket': ticket,
                'symbol': tracked_positions[ticket]['symbol'],
                'type': tracked_positions[ticket]['type'],
                'close_time': datetime.now().isoformat(),
                'profit': None  # Will be determined by balance tracking
            }
            
            # Try to get actual profit from history one more time
            history = mt5.history_deals_get(
                tracked_positions[ticket]['open_time'] - timedelta(minutes=1),
                datetime.now()
            )
            
            if history:
                for deal in history:
                    if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                        closed_trade['profit'] = deal.profit
                        break
            
            # If we still don't have profit, we'll track it as "pending verification"
            if closed_trade['profit'] is None:
                print(f"  [WARN] Position #{ticket} closed but profit unknown - will verify on next cycle")
            else:
                closed_trades.append(closed_trade)
                print(f"  [CLOSED] Position #{ticket} closed with profit: ${closed_trade['profit']:.2f}")
                
                # Update loss streak
                update_loss_streak_from_trades()
                save_trade_history()
            
            # Remove from tracked positions
            del tracked_positions[ticket]


def update_loss_streak_from_trades():
    """Calculate loss streak from tracked closed trades"""
    global loss_streak, cooldown_until
    
    if len(closed_trades) == 0:
        loss_streak = 0
        return
    
    # Filter out trades with unknown profit
    verified_trades = [t for t in closed_trades if t['profit'] is not None]
    
    if len(verified_trades) == 0:
        loss_streak = 0
        return
    
    # Count consecutive losses from most recent
    streak = 0
    for trade in reversed(verified_trades):
        if trade['profit'] < 0:
            streak += 1
        else:
            break
    
    old_streak = loss_streak
    loss_streak = streak
    
    if old_streak != loss_streak:
        print(f"  [STREAK] Loss streak updated: {old_streak} -> {loss_streak}")
    
    # Activate cooldown if needed
    if loss_streak >= max_loss_streak:
        if cooldown_until is None:
            cooldown_until = datetime.now().timestamp() + cooldown_period
            print(f"\n{'='*70}")
            print(f"COOLDOWN ACTIVATED: {loss_streak} consecutive losses detected.")
            print(f"Trading paused until {datetime.fromtimestamp(cooldown_until).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")
    else:
        if cooldown_until is not None:
            print(f"\n{'='*70}")
            print(f"COOLDOWN ENDED: Loss streak broken (now {loss_streak} losses).")
            print(f"Trading resumed.")
            print(f"{'='*70}\n")
            cooldown_until = None


def is_trading_allowed():
    """Check if trading is allowed (not in cooldown)"""
    global cooldown_until, loss_streak
    
    if cooldown_until is None:
        return True
    
    current_time = datetime.now().timestamp()
    
    if current_time >= cooldown_until:
        if loss_streak >= max_loss_streak:
            cooldown_until = current_time + cooldown_period
            print(f"\n{'='*70}")
            print(f"COOLDOWN EXTENDED: Still {loss_streak} consecutive losses.")
            print(f"Trading paused until {datetime.fromtimestamp(cooldown_until).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")
            return False
        else:
            print(f"\n{'='*70}")
            print(f"COOLDOWN CLEARED: Loss streak is now {loss_streak}. Trading resumed.")
            print(f"{'='*70}\n")
            cooldown_until = None
            return True
    
    return False


def run_bot():
    """Main bot loop"""
    global bot_start_time
    
    # Set bot start time
    bot_start_time = datetime.now()
    
    # Load previous trade history
    load_trade_history()
    
    print("=" * 70)
    print("RSI-MA TRADING BOT - LIVE MODE (Position Tracking)")
    print("=" * 70)
    print(f"Bot Start Time: {bot_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"RSI Period: {rsi_period} | MA Period: {ma_period}")
    print(f"Check Interval: {check_interval} seconds")
    print(f"Max Loss Streak: {max_loss_streak} trades (Cooldown: {cooldown_period/60:.0f} minutes)")
    print(f"Previous closed trades: {len(closed_trades)}")
    print("=" * 70)
    print("\nBot started. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                time.sleep(check_interval)
                continue
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f} | Loss Streak: {loss_streak} | Tracked: {len(tracked_positions)}")
            
            # Track positions and detect closures
            track_positions()
            
            # Check if trading is allowed
            if not is_trading_allowed():
                if cooldown_until:
                    remaining = int(cooldown_until - datetime.now().timestamp())
                    print(f"  COOLDOWN ACTIVE: Trading blocked for {remaining//60}m {remaining%60}s")
                print()
                time.sleep(check_interval)
                continue
            
            # Trading is allowed - check for signals
            print("  [STATUS] Trading is ALLOWED - checking for signals...")
            
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