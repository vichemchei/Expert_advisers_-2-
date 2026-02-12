"""
Enhanced RSI-MA-ATR Trading Bot - ALL ISSUES FIXED
Fixed: Trailing stops, position tracking, win/loss tracking
"""
import os
import time
import csv
from datetime import datetime, timedelta
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import joblib

# ============ CONFIGURATION ============
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
symbols = ["XAUUSD", "XAUEUR","AUDUSD","USDJPY"]
magic_number = 234000
comment_text = "RSI-MA-ATR-ML-Bot-v2"

# Technical indicators
rsi_period = 14
ma_short = 20
ma_long = 50
atr_period = 14

# Risk management
default_lot = 0.01
risk_per_trade = 0.01  # 1%
max_positions_per_symbol = 2
max_total_positions = 4
max_daily_loss_pct = 0.03  # 3% daily loss limit
max_total_risk_pct = 0.05  # 5% total exposure limit

# ML settings
MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clf_signal.pkl")
LOT_REG_PATH = os.path.join(MODEL_DIR, "reg_lot.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)
ML_PROB_THRESHOLD = 0.60

# Trading hours filter
TRADING_HOURS_EAT = [
    (9, 15),
    (15, 18)
]

# Daily performance tracking
daily_stats = {
    'start_balance': None,
    'trades_today': 0,
    'wins_today': 0,
    'losses_today': 0,
    'date': datetime.now().date(),
    'closed_tickets': set()  # Track closed tickets to avoid double counting
}

if not mt5.initialize(path=MT5_PATH):
    print("MT5 init failed:", mt5.last_error())

# ============ SIMPLIFIED TRAILING STOP SYSTEM (NO ML) ============

class SimplifiedTrailingStop:
    """
    Simplified trailing stop - ATR-based only, no ML dependency
    Rules:
    1. Initial SL at 1.5 ATR
    2. Move to break-even after 1 ATR profit
    3. Trail by 1 ATR after 2 ATR profit
    4. Close if held > 4 hours with < 0.5 ATR profit
    """
    
    def __init__(self, ticket, symbol, entry_price, pos_type, atr, entry_time):
        self.ticket = ticket
        self.symbol = symbol
        self.entry_price = entry_price
        self.pos_type = pos_type
        self.atr = atr
        self.entry_time = entry_time
        self.be_triggered = False
        self.trailing_triggered = False
        self.highest_profit_atr = 0.0
        
    def update(self, current_price, current_time):
        """Calculate new stop level"""
        is_buy = (self.pos_type == mt5.ORDER_TYPE_BUY)
        
        # Calculate profit in ATR
        if is_buy:
            profit_atr = (current_price - self.entry_price) / self.atr
        else:
            profit_atr = (self.entry_price - current_price) / self.atr
        
        self.highest_profit_atr = max(self.highest_profit_atr, profit_atr)
        
        # Time-based exit
        hours_held = (current_time - self.entry_time).total_seconds() / 3600
        if hours_held >= 4 and profit_atr < 0.5:
            return "CLOSE", f"Time exit ({hours_held:.1f}h)"
        
        # Trailing stop logic
        if profit_atr >= 2.0 and not self.trailing_triggered:
            self.trailing_triggered = True
            trail_distance = 1.0 * self.atr
            new_sl = current_price - trail_distance if is_buy else current_price + trail_distance
            return new_sl, f"Trailing (profit={profit_atr:.2f}ATR)"
        
        # Break-even stop
        if profit_atr >= 1.0 and not self.be_triggered:
            self.be_triggered = True
            be_sl = self.entry_price + (0.2 * self.atr if is_buy else -0.2 * self.atr)
            return be_sl, f"Break-even (profit={profit_atr:.2f}ATR)"
        
        # Continue trailing if already triggered
        if self.trailing_triggered and profit_atr >= 1.5:
            trail_distance = 1.0 * self.atr
            new_sl = current_price - trail_distance if is_buy else current_price + trail_distance
            return new_sl, f"Trailing (profit={profit_atr:.2f}ATR)"
        
        # Initial stop
        initial_sl = self.entry_price - (1.5 * self.atr if is_buy else -1.5 * self.atr)
        return initial_sl, f"Initial SL"


# ============ POSITION TRACKER ============

class PositionTracker:
    """Track all open positions and their states"""
    
    def __init__(self):
        self.positions = {}  # ticket -> position info
        self.last_update = datetime.now()
    
    def update(self):
        """Update position tracking from MT5"""
        mt5_positions = mt5.positions_get()
        current_tickets = set()
        
        if mt5_positions:
            for pos in mt5_positions:
                comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
                if pos.magic == magic_number and comment == comment_text:
                    ticket = pos.ticket
                    current_tickets.add(ticket)
                    
                    # Add new position
                    if ticket not in self.positions:
                        self.positions[ticket] = {
                            'symbol': pos.symbol,
                            'type': pos.type,
                            'volume': pos.volume,
                            'entry_price': pos.price_open,
                            'entry_time': datetime.fromtimestamp(pos.time),
                            'sl': pos.sl,
                            'tp': pos.tp
                        }
                        print(f"📌 New position tracked: {pos.symbol} ticket {ticket}")
                    else:
                        # Update existing
                        self.positions[ticket]['sl'] = pos.sl
                        self.positions[ticket]['tp'] = pos.tp
        
        # Detect closed positions
        closed_tickets = set(self.positions.keys()) - current_tickets
        for ticket in closed_tickets:
            pos_info = self.positions[ticket]
            print(f"🔴 Position closed detected: {pos_info['symbol']} ticket {ticket}")
            
            # Get final result
            if ticket not in daily_stats['closed_tickets']:
                deals = mt5.history_deals_get(ticket=ticket)
                profit = 0
                if deals and len(deals) > 0:
                    profit = sum(deal.profit for deal in deals)
                
                # Update win/loss stats
                if profit > 0:
                    daily_stats['wins_today'] += 1
                    print(f"  ✅ WIN: ${profit:.2f}")
                elif profit < 0:
                    daily_stats['losses_today'] += 1
                    print(f"  ❌ LOSS: ${profit:.2f}")
                
                daily_stats['closed_tickets'].add(ticket)
                trade_logger.log_trade_close(ticket, 0, profit, "Detected closed")
            
            del self.positions[ticket]
        
        self.last_update = datetime.now()
    
    def get_count(self, symbol=None):
        """Get position count"""
        if symbol:
            return sum(1 for p in self.positions.values() if p['symbol'] == symbol)
        return len(self.positions)
    
    def get_position(self, ticket):
        """Get position info"""
        return self.positions.get(ticket)


# ============ TRADE LOGGING SYSTEM ============

class TradeLogger:
    """Comprehensive trade logging for analysis"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.trades_log = os.path.join(log_dir, "trades.csv")
        self.signals_log = os.path.join(log_dir, "signals.csv")
        self.errors_log = os.path.join(log_dir, "errors.csv")
        self.performance_log = os.path.join(log_dir, "daily_performance.csv")
        
        self._init_log_files()
    
    def _init_log_files(self):
        """Create log files with headers"""
        if not os.path.exists(self.trades_log):
            with open(self.trades_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'ticket', 'entry_price', 
                    'lot_size', 'sl', 'tp', 'atr', 'exit_time', 'exit_price', 
                    'profit', 'hold_time_hours', 'exit_reason'
                ])
        
        if not os.path.exists(self.signals_log):
            with open(self.signals_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'signal_type', 'ml_prob', 
                    'rsi', 'ema_diff', 'atr', 'executed', 'rejection_reason'
                ])
        
        if not os.path.exists(self.errors_log):
            with open(self.errors_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'error_type', 'symbol', 'message'])
        
        if not os.path.exists(self.performance_log):
            with open(self.performance_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'start_balance', 'end_balance', 'daily_pnl', 
                    'daily_pnl_pct', 'trades', 'wins', 'losses', 'win_rate'
                ])
    
    def log_trade_open(self, symbol, action, ticket, entry_price, lot_size, sl, tp, atr):
        try:
            with open(self.trades_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), symbol, action.upper(), ticket,
                    f"{entry_price:.5f}", lot_size,
                    f"{sl:.5f}" if sl else "None",
                    f"{tp:.5f}" if tp else "None",
                    f"{atr:.5f}", "", "", "", "", ""
                ])
        except Exception as e:
            print(f"Error logging trade open: {e}")
    
    def log_trade_close(self, ticket, exit_price, profit, exit_reason):
        try:
            with open(self.trades_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), "CLOSE", "", ticket, "",
                    "", "", "", "", datetime.now().isoformat(),
                    f"{exit_price:.5f}", f"{profit:.2f}", "", exit_reason
                ])
        except Exception as e:
            print(f"Error logging trade close: {e}")
    
    def log_signal(self, symbol, signal_type, ml_prob, rsi, ema_diff, atr, executed, rejection_reason=""):
        try:
            with open(self.signals_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), symbol, signal_type,
                    f"{ml_prob:.3f}" if ml_prob else "N/A",
                    f"{rsi:.2f}", f"{ema_diff:.5f}", f"{atr:.5f}",
                    "YES" if executed else "NO", rejection_reason
                ])
        except Exception as e:
            print(f"Error logging signal: {e}")
    
    def log_error(self, error_type, symbol, message):
        try:
            with open(self.errors_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), error_type, symbol, message])
        except Exception as e:
            print(f"Error logging error: {e}")
    
    def log_daily_performance(self, start_balance, end_balance, trades, wins, losses):
        try:
            daily_pnl = end_balance - start_balance
            daily_pnl_pct = (daily_pnl / start_balance * 100) if start_balance > 0 else 0
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            with open(self.performance_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().date().isoformat(),
                    f"{start_balance:.2f}", f"{end_balance:.2f}",
                    f"{daily_pnl:.2f}", f"{daily_pnl_pct:.2f}",
                    trades, wins, losses, f"{win_rate:.1f}"
                ])
        except Exception as e:
            print(f"Error logging daily performance: {e}")


# ============ GLOBAL OBJECTS ============
trailing_stops = {}  # ticket -> SimplifiedTrailingStop
position_tracker = PositionTracker()
trade_logger = TradeLogger()

# ============ HELPER FUNCTIONS ============

def is_trading_hours():
    """Check if current time is within allowed trading hours"""
    current_hour = datetime.now().hour
    for start, end in TRADING_HOURS_EAT:
        if start <= current_hour < end:
            return True
    return False

def check_daily_loss_limit():
    """Check daily loss limit with proper reset"""
    acc_info = mt5.account_info()
    if acc_info is None:
        return True
    
    current_date = datetime.now().date()
    
    # Reset on new day
    if daily_stats['date'] != current_date:
        # Log previous day
        if daily_stats['start_balance'] is not None:
            trade_logger.log_daily_performance(
                daily_stats['start_balance'], acc_info.balance,
                daily_stats['trades_today'], daily_stats['wins_today'],
                daily_stats['losses_today']
            )
        
        daily_stats['date'] = current_date
        daily_stats['start_balance'] = acc_info.balance
        daily_stats['trades_today'] = 0
        daily_stats['wins_today'] = 0
        daily_stats['losses_today'] = 0
        daily_stats['closed_tickets'] = set()
        print(f"\n{'='*60}")
        print(f"🌅 NEW DAY: Reset stats. Starting balance: ${acc_info.balance:.2f}")
        print(f"{'='*60}\n")
        return True
    
    # Initialize if first run
    if daily_stats['start_balance'] is None:
        daily_stats['start_balance'] = acc_info.balance
        return True
    
    # Calculate P&L
    daily_pnl = acc_info.balance - daily_stats['start_balance']
    daily_loss_pct = abs(daily_pnl) / daily_stats['start_balance'] if daily_stats['start_balance'] > 0 else 0
    
    # Only count losses
    if daily_pnl < 0 and daily_loss_pct >= max_daily_loss_pct:
        print(f"❌ DAILY LOSS LIMIT: {daily_loss_pct*100:.2f}% (${daily_pnl:.2f})")
        return False
    
    return True

def get_current_exposure():
    """Calculate total risk exposure"""
    positions = mt5.positions_get()
    if positions is None:
        return 0.0
    
    total_risk = 0.0
    acc_info = mt5.account_info()
    if acc_info is None:
        return 0.0
    
    balance = acc_info.balance
    
    for pos in positions:
        if pos.magic != magic_number:
            continue
        
        if pos.sl > 0:
            risk_distance = abs(pos.price_open - pos.sl)
            
            # Get correct contract size
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info:
                contract_size = symbol_info.trade_contract_size
            else:
                contract_size = 100 if 'XAU' in pos.symbol else 100000
            
            risk_amount = risk_distance * pos.volume * contract_size
            total_risk += risk_amount
    
    return total_risk / balance if balance > 0 else 0.0

def rebuild_trailing_stops():
    """Rebuild trailing stops on restart"""
    global trailing_stops
    trailing_stops = {}
    
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        print("No existing positions to rebuild")
        return
    
    print(f"\n{'='*60}")
    print("REBUILDING TRAILING STOPS")
    print('='*60)
    
    rebuilt = 0
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        ticket = pos.ticket
        
        df = get_ohlcv(symbol, 200)
        if df is None:
            continue
        
        atr_series = calc_atr(df, atr_period)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            continue
        
        atr_val = float(atr_series.iloc[-1])
        entry_time = datetime.fromtimestamp(pos.time)
        
        trailing_stops[ticket] = SimplifiedTrailingStop(
            ticket, symbol, pos.price_open, pos.type, atr_val, entry_time
        )
        
        rebuilt += 1
        hours_ago = (datetime.now() - entry_time).total_seconds() / 3600
        print(f"✓ Rebuilt: {symbol} ticket {ticket} (opened {hours_ago:.1f}h ago)")
    
    print(f"{'='*60}")
    print(f"Successfully rebuilt {rebuilt} trailing stops")
    print(f"{'='*60}\n")

# ============ TECHNICAL INDICATORS ============

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    """Calculate ATR with proper NaN handling"""
    if df is None or len(df) < 2:
        return None
    
    d = df.copy()
    d['h-l'] = d['high'] - d['low']
    d['h-pc'] = (d['high'] - d['close'].shift(1)).abs()
    d['l-pc'] = (d['low'] - d['close'].shift(1)).abs()
    d['tr'] = d[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    if len(d) < period:
        d['atr'] = d['tr'].expanding(min_periods=2).mean()
    else:
        d['atr'] = d['tr'].rolling(window=period, min_periods=period).mean()
    
    # Forward fill
    d['atr'] = d['atr'].bfill().ffill()
    
    return d['atr']

def get_ohlcv(symbol, num_bars=500, timeframe=mt5.TIMEFRAME_M5):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ FEATURE ENGINEERING ============

def build_features(df):
    """Build features with proper NaN handling"""
    d = df.copy()
    
    # Base features
    d['ema_short'] = ema(d['close'], ma_short)
    d['ema_long'] = ema(d['close'], ma_long)
    d['rsi'] = rsi(d['close'], rsi_period)
    d['atr'] = calc_atr(d, atr_period)
    
    # Derived features
    d['ema_diff'] = d['ema_short'] - d['ema_long']
    d['ret_1'] = d['close'].pct_change(1)
    d['ret_5'] = d['close'].pct_change(5)
    d['vol_10'] = d['tick_volume'].rolling(10, min_periods=1).mean()
    d['volatility_10'] = d['close'].pct_change().rolling(10, min_periods=1).std()
    d['close_minus_ema_short'] = d['close'] - d['ema_short']
    d['close_minus_ema_long'] = d['close'] - d['ema_long']
    
    # Enhanced features
    d['rsi_momentum'] = d['rsi'].diff(3).fillna(0)
    d['ema_diff_pct'] = d['ema_diff'] / d['close'].replace(0, np.nan)
    d['atr_pct'] = d['atr'] / d['close'].replace(0, np.nan)
    d['volume_ratio'] = d['tick_volume'] / d['vol_10'].replace(0, np.nan)
    d['price_distance_short'] = (d['close'] - d['ema_short']) / d['atr'].replace(0, np.nan)
    d['price_distance_long'] = (d['close'] - d['ema_long']) / d['atr'].replace(0, np.nan)
    d['trend_strength'] = d['ema_diff'].abs() / d['atr'].replace(0, np.nan)
    
    vol_mean_50 = d['volatility_10'].rolling(50, min_periods=10).mean()
    d['volatility_regime'] = d['volatility_10'] / vol_mean_50.replace(0, np.nan)
    
    # Fill NaNs
    d = d.fillna(0)
    
    return d

def label_future(df, lookahead=20, tp_atr_mult=2.5, sl_atr_mult=1.5):
    """Label data for training"""
    d = df.copy()
    d['future_high'] = d['high'].shift(-1).rolling(window=lookahead, min_periods=1).max()
    d['future_low'] = d['low'].shift(-1).rolling(window=lookahead, min_periods=1).min()
    
    labels = []
    for i in range(len(d)):
        if i + 1 >= len(d):
            labels.append(0)
            continue
        
        entry = d['close'].iloc[i]
        atr = d['atr'].iloc[i]
        
        if pd.isna(atr):
            labels.append(0)
            continue
        
        tp_buy = entry + tp_atr_mult * atr
        sl_buy = entry - sl_atr_mult * atr
        fut_high = d['future_high'].iloc[i]
        fut_low = d['future_low'].iloc[i]
        
        tp_sell = entry - tp_atr_mult * atr
        sl_sell = entry + sl_atr_mult * atr
        
        buy_wins = fut_high >= tp_buy and fut_low > sl_buy
        sell_wins = fut_low <= tp_sell and fut_high < sl_sell
        
        if buy_wins and not sell_wins:
            labels.append(1)
        elif sell_wins and not buy_wins:
            labels.append(-1)
        else:
            labels.append(0)
    
    d['label'] = labels
    return d

# ============ DATA COLLECTION ============

def collect_and_save(symbol, out_csv, num_bars=2000, lookahead=20):
    print(f"Collecting {symbol}...")
    df = get_ohlcv(symbol, num_bars)
    if df is None:
        print("No data for", symbol)
        return
    
    feat = build_features(df)
    labeled = label_future(feat, lookahead=lookahead)
    
    keep_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                 'ema_short', 'ema_long', 'rsi', 'atr', 'ema_diff', 'ret_1', 'ret_5', 
                 'vol_10', 'volatility_10', 'close_minus_ema_short', 'close_minus_ema_long',
                 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 'volume_ratio', 
                 'price_distance_short', 'price_distance_long', 'trend_strength', 
                 'volatility_regime', 'label']
    
    out = labeled[keep_cols].dropna()
    
    if os.path.exists(out_csv):
        out.to_csv(out_csv, mode='a', header=False, index=False)
    else:
        out.to_csv(out_csv, index=False)
    
    print(f"Saved {len(out)} rows to {out_csv}")

# ============ MODEL TRAINING ============

def train_models(csv_path, test_size=0.2, random_state=42):
    """Train ML models"""
    df = pd.read_csv(csv_path)
    
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                   'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                   'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
    
    df = df.dropna(subset=feature_cols + ['label'])
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    X = df[feature_cols].values
    y = df['label'].values
    y_bin = np.where(y == 1, 1, 0)
    y_sell = np.where(y == -1, 1, 0)
    
    # Train buy classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )
    
    clf_buy = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=20,
        min_samples_leaf=10, random_state=random_state, class_weight='balanced'
    )
    clf_buy.fit(X_train, y_train)
    
    print(f"\nBUY Classifier Test Performance:")
    y_pred = clf_buy.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No-Buy', 'Buy']))
    
    # Train sell classifier
    X_train_sell, X_test_sell, y_train_sell, y_test_sell = train_test_split(
        X, y_sell, test_size=test_size, random_state=random_state, stratify=y_sell
    )
    
    clf_sell = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=20,
        min_samples_leaf=10, random_state=random_state, class_weight='balanced'
    )
    clf_sell.fit(X_train_sell, y_train_sell)
    
    print("\nSELL Classifier Test Performance:")
    y_pred_sell = clf_sell.predict(X_test_sell)
    print(classification_report(y_test_sell, y_pred_sell, target_names=['No-Sell', 'Sell']))
    
    joblib.dump({'buy': clf_buy, 'sell': clf_sell}, CLASSIFIER_PATH)
    print(f"\nSaved classifiers -> {CLASSIFIER_PATH}")
    
    # Train lot regressor
    df_profitable = df[df['label'].abs() == 1].copy()
    
    if len(df_profitable) >= 100:
        target_lot = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_lot = (target_lot / target_lot.max()) * 0.5
        
        Xl = df_profitable[feature_cols].values
        yl = target_lot.values
        
        reg_lot = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        reg_lot.fit(Xl, yl)
        joblib.dump(reg_lot, LOT_REG_PATH)
        print(f"Saved lot regressor -> {LOT_REG_PATH}")

# ============ LOAD MODELS ============

def load_models():
    clf = None
    reg_lot = None
    
    if os.path.exists(CLASSIFIER_PATH):
        try:
            clf = joblib.load(CLASSIFIER_PATH)
            print('✓ Loaded classifiers (buy & sell)')
        except Exception as e:
            print('Failed to load classifier', e)
    
    if os.path.exists(LOT_REG_PATH):
        try:
            reg_lot = joblib.load(LOT_REG_PATH)
            print('✓ Loaded lot regressor')
        except Exception as e:
            print('Failed to load lot regressor', e)
    
    return clf, reg_lot

# ============ SIGNAL GENERATION ============

def rule_signal_from_df(df):
    """Basic rule-based signal logic"""
    if df is None or len(df) < max(ma_long, rsi_period) + 2:
        return None
    
    df = df.copy()
    df['ema_short'] = ema(df['close'], ma_short)
    df['ema_long'] = ema(df['close'], ma_long)
    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = calc_atr(df, atr_period)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    trend_strength = abs(last['ema_short'] - last['ema_long']) / last['atr']
    
    if trend_strength < 0.5:
        return None
    
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    
    if (trend_up and 
        prev['low'] <= prev['ema_short'] and 
        last['close'] > last['ema_short'] and 
        35 < last['rsi'] < 70):
        return 'buy'
    
    elif (trend_down and 
          prev['high'] >= prev['ema_short'] and 
          last['close'] < last['ema_short'] and 
          30 < last['rsi'] < 65):
        return 'sell'
    
    return None

def ml_filtered_signal(symbol, clf, df):
    """Apply ML filter to signals"""
    if df is None or len(df) < 30:
        return None
    
    feat = build_features(df).iloc[-1]
    
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                   'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                   'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
    
    if feat[feature_cols].isna().any():
        print(f"Warning: NaN features detected for {symbol}, skipping signal")
        return None
    
    X = feat[feature_cols].values.reshape(1, -1)
    rule = rule_signal_from_df(df)
    
    if clf is None:
        return rule
    
    try:
        # Handle dictionary format
        if isinstance(clf, dict):
            clf_buy = clf.get('buy')
            clf_sell = clf.get('sell')
        else:
            clf_buy = clf
            clf_sell = None
        
        # Buy signal
        if rule == 'buy' and clf_buy is not None:
            proba = clf_buy.predict_proba(X)[0]
            prob_buy = proba[1] if len(proba) > 1 else proba[0]
            
            trade_logger.log_signal(
                symbol, 'BUY', prob_buy, 
                feat['rsi'], feat['ema_diff'], feat['atr'],
                prob_buy >= ML_PROB_THRESHOLD,
                "" if prob_buy >= ML_PROB_THRESHOLD else f"ML prob too low: {prob_buy:.3f}"
            )
            
            if prob_buy >= ML_PROB_THRESHOLD:
                print(f"✓ ML-filtered BUY signal for {symbol} (prob={prob_buy:.3f})")
                return 'buy'
            else:
                print(f"✗ BUY signal rejected by ML for {symbol} (prob={prob_buy:.3f})")
                return None
        
        # Sell signal
        if rule == 'sell':
            if clf_sell is not None:
                proba = clf_sell.predict_proba(X)[0]
                prob_sell = proba[1] if len(proba) > 1 else proba[0]
                
                trade_logger.log_signal(
                    symbol, 'SELL', prob_sell,
                    feat['rsi'], feat['ema_diff'], feat['atr'],
                    prob_sell >= ML_PROB_THRESHOLD,
                    "" if prob_sell >= ML_PROB_THRESHOLD else f"ML prob too low: {prob_sell:.3f}"
                )
                
                if prob_sell >= ML_PROB_THRESHOLD:
                    print(f"✓ ML-filtered SELL signal for {symbol} (prob={prob_sell:.3f})")
                    return 'sell'
                else:
                    print(f"✗ SELL signal rejected by ML for {symbol} (prob={prob_sell:.3f})")
                    return None
            else:
                # Fallback
                trade_logger.log_signal(
                    symbol, 'SELL', None,
                    feat['rsi'], feat['ema_diff'], feat['atr'],
                    True, "No ML model - using rule-based"
                )
                return 'sell'
        
        return None
        
    except Exception as e:
        print(f"ML prediction error for {symbol}: {e}")
        trade_logger.log_error("ML_PREDICTION", symbol, str(e))
        return None

# ============ POSITION SIZING ============

def calculate_lot_ml(symbol, reg_lot, sl_price_distance, current_features=None):
    """Conservative lot sizing with ML adjustment"""
    try:
        acc_info = mt5.account_info()
        if acc_info is None or sl_price_distance is None or sl_price_distance <= 0:
            return default_lot
        
        balance = acc_info.balance
        risk_amount = balance * risk_per_trade
        
        pip_value_per_lot = 10.0
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point if symbol_info and symbol_info.point else 1e-5
        sl_pips = sl_price_distance / point
        
        if sl_pips <= 0:
            return default_lot
        
        base_lot = risk_amount / (sl_pips * pip_value_per_lot)
        base_lot = max(round(base_lot, 2), 0.01)
        
    except Exception as e:
        print(f'calculate_lot base error: {e}')
        return default_lot
    
    if reg_lot is None:
        return base_lot
    
    try:
        if current_features is None:
            df = get_ohlcv(symbol, 200)
            if df is None or len(df) < 20:
                return base_lot
            
            feat = build_features(df).iloc[-1]
            feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                           'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                           'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
            
            if feat[feature_cols].isna().any():
                return base_lot
            
            X = feat[feature_cols].values.reshape(1, -1)
        else:
            X = current_features.reshape(1, -1)
        
        pred = reg_lot.predict(X)[0]
        multiplier = 0.7 + (pred * 0.8)
        multiplier = max(0.5, min(multiplier, 1.5))
        
        final_lot = base_lot * multiplier
        final_lot = max(0.01, round(final_lot, 2))
        
        max_lot = balance * 0.05 / 1000
        final_lot = min(final_lot, max_lot)
        
        return final_lot
        
    except Exception as e:
        print(f'lot regressor error: {e}')
        return base_lot

# ============ PRICE NORMALIZATION ============

def get_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, 1e-5, 1e-5
    point = getattr(symbol_info, "point", 1e-5)
    trade_tick_size = getattr(symbol_info, "trade_tick_size", point)
    if trade_tick_size is None or trade_tick_size <= 0:
        trade_tick_size = point
    return symbol_info, float(point), float(trade_tick_size)

def get_min_stop_distance(symbol):
    symbol_info, point, tick_size = get_symbol_info(symbol)
    if symbol_info is None:
        return 10 * point
    
    possible_attrs = ["trade_stops_level", "stops_level", "stop_level", "stoplevel"]
    for attr in possible_attrs:
        val = getattr(symbol_info, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            if val < 1000:
                return float(val) * point
            return float(val)
    
    return max(10 * point, tick_size * 2)

def normalize_price(price, tick_size):
    if tick_size <= 0:
        return float(price)
    normalized = round(price / tick_size) * tick_size
    return float(normalized)

def validate_sl_distance(current_price, new_sl, min_stop_dist, pos_type):
    if new_sl is None or new_sl == 0:
        return False
    
    actual_distance = abs(current_price - new_sl)
    if actual_distance < min_stop_dist:
        return False
    
    if pos_type == mt5.ORDER_TYPE_BUY and new_sl >= current_price:
        return False
    if pos_type == mt5.ORDER_TYPE_SELL and new_sl <= current_price:
        return False
    
    return True

# ============ MANAGE TRAILING POSITIONS ============

def manage_trailing_positions():
    """FIXED: Simplified trailing stop management - NO ML dependency"""
    positions = mt5.positions_get()
    if positions is None:
        return
    
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        ticket = pos.ticket
        pos_type = pos.type
        
        symbol_info, point, tick_size = get_symbol_info(symbol)
        if symbol_info is None:
            continue
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        current_price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
        
        # Get ATR
        df = get_ohlcv(symbol, 200)
        if df is None or len(df) < 30:
            continue
        
        atr_series = calc_atr(df, atr_period)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            continue
        
        atr_val = float(atr_series.iloc[-1])
        min_stop_dist = get_min_stop_distance(symbol)
        
        # Initialize trailing stop if doesn't exist
        if ticket not in trailing_stops:
            entry_time = datetime.fromtimestamp(pos.time)
            trailing_stops[ticket] = SimplifiedTrailingStop(
                ticket, symbol, pos.price_open, pos_type, atr_val, entry_time
            )
        
        # Get stop recommendation
        trail_stop = trailing_stops[ticket]
        new_sl, reason = trail_stop.update(current_price, datetime.now())
        
        # Handle close signal
        if new_sl == "CLOSE":
            filling_mode = get_filling_mode(symbol)
            close_req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": int(ticket),
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "deviation": 20,
                "magic": magic_number,
                "comment": f"Trail: {reason}",
                "type_filling": filling_mode
            }
            result = mt5.order_send(close_req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                profit = pos.profit
                print(f"🔴 CLOSED {symbol} ticket {ticket}: {reason} | Profit: ${profit:.2f}")
                trade_logger.log_trade_close(ticket, current_price, profit, reason)
                
                # Track win/loss
                if ticket not in daily_stats['closed_tickets']:
                    if profit > 0:
                        daily_stats['wins_today'] += 1
                    else:
                        daily_stats['losses_today'] += 1
                    daily_stats['closed_tickets'].add(ticket)
                
                del trailing_stops[ticket]
            continue
        
        # Validate and normalize stop
        new_sl = normalize_price(new_sl, tick_size)
        
        if not validate_sl_distance(current_price, new_sl, min_stop_dist, pos_type):
            continue
        
        # Only modify if different
        current_sl = pos.sl if pos.sl != 0 else None
        if current_sl is None or abs(new_sl - current_sl) > tick_size:
            modify_req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": float(pos.tp) if pos.tp != 0 else 0.0,
                "magic": magic_number,
                "comment": comment_text
            }
            
            result = mt5.order_send(modify_req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"📊 Modified {symbol} ticket {ticket}: SL={new_sl:.5f} ({reason})")

# ============ ORDER FILLING MODE ============

def get_filling_mode(symbol):
    """Determine the correct filling mode for a symbol"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_FOK
    
    filling_mode = symbol_info.filling_mode
    
    if filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

# ============ OPEN TRADE ============

def open_trade(symbol, action, clf, reg_lot):
    """Open trade with enhanced tracking"""
    
    # Check position limits using tracker
    if position_tracker.get_count(symbol) >= max_positions_per_symbol:
        print(f"⚠ Max positions for {symbol} reached ({max_positions_per_symbol})")
        return
    
    if position_tracker.get_count() >= max_total_positions:
        print(f"⚠ Max total positions reached ({max_total_positions})")
        return
    
    # Check exposure
    current_exposure = get_current_exposure()
    if current_exposure >= max_total_risk_pct:
        print(f"⚠ Max exposure reached: {current_exposure*100:.2f}%")
        return
    
    # Check daily loss limit
    if not check_daily_loss_limit():
        return
    
    # Check trading hours
    if not is_trading_hours():
        print(f"⏰ Outside trading hours, skipping {symbol} {action}")
        return
    
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Symbol {symbol} not available")
        return
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ No tick for {symbol}")
        return
    
    symbol_info, point, tick_size = get_symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Failed to get symbol info for {symbol}")
        return
    
    price = tick.ask if action == "buy" else tick.bid
    
    df = get_ohlcv(symbol, 200)
    if df is None or len(df) < 20:
        return
    
    atr_series = calc_atr(df, atr_period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return
    
    atr_val = float(atr_series.iloc[-1])
    
    atr_mult_sl = 1.5
    atr_mult_tp = 2.5
    
    if action == "buy":
        candidate_sl = price - (atr_mult_sl * atr_val)
        candidate_tp = price + (atr_mult_tp * atr_val)
    else:
        candidate_sl = price + (atr_mult_sl * atr_val)
        candidate_tp = price - (atr_mult_tp * atr_val)
    
    candidate_sl = normalize_price(candidate_sl, tick_size)
    candidate_tp = normalize_price(candidate_tp, tick_size)
    
    min_stop_dist = get_min_stop_distance(symbol)
    sl_distance = abs(price - candidate_sl)
    use_stops = True
    
    if sl_distance < min_stop_dist:
        if action == "buy":
            candidate_sl = price - min_stop_dist
        else:
            candidate_sl = price + min_stop_dist
        
        candidate_sl = normalize_price(candidate_sl, tick_size)
        sl_distance = abs(price - candidate_sl)
        
        if sl_distance < min_stop_dist:
            use_stops = False
            print(f"⚠ Warning: Cannot set stops for {symbol}")
    
    lot = calculate_lot_ml(symbol, reg_lot, sl_distance if use_stops else None)
    if lot < 0.01:
        lot = default_lot
    
    filling_mode = get_filling_mode(symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "magic": magic_number,
        "comment": comment_text,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    
    if use_stops:
        request["sl"] = candidate_sl
        request["tp"] = candidate_tp
    
    result = mt5.order_send(request)
    
    if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
        print(f"❌ {symbol} {action} failed: {getattr(result, 'comment', 'N/A')}")
        trade_logger.log_error("ORDER_FAILED", symbol, f"{action} order failed")
    else:
        daily_stats['trades_today'] += 1
        ticket = getattr(result, 'order', 0)
        print(f"✅ {symbol} {action.upper()} opened at {price} | SL={candidate_sl:.5f}, TP={candidate_tp:.5f}, ATR={atr_val:.5f}, LOT={lot}")
        
        # Log trade
        trade_logger.log_trade_open(
            symbol, action, ticket, price, lot,
            candidate_sl if use_stops else None,
            candidate_tp if use_stops else None,
            atr_val
        )
        
        # Initialize trailing stop
        trailing_stops[ticket] = SimplifiedTrailingStop(
            ticket, symbol, price, 
            mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            atr_val, datetime.now()
        )

# ============ MAIN LIVE LOOP ============

def run_live(symbols, mode='live', data_csv='training_data.csv'):
    clf, reg_lot = load_models()
    
    if mode == 'collect':
        for s in symbols:
            collect_and_save(s, data_csv, num_bars=2000)
        return
    
    if mode == 'train':
        train_models(data_csv)
        return
    
    # REBUILD ON START
    if mode == 'live':
        rebuild_trailing_stops()
        position_tracker.update()
    
    print('\n' + '='*60)
    print('🚀 LIVE TRADING - ALL ISSUES FIXED')
    print('='*60)
    print(f"Symbols: {symbols}")
    print(f"Magic: {magic_number}")
    print(f"Risk/trade: {risk_per_trade*100:.1f}%")
    print(f"Max positions: {max_total_positions}")
    print(f"Daily loss limit: {max_daily_loss_pct*100:.1f}%")
    print(f"✅ Trailing stops: SIMPLIFIED (NO ML)")
    print(f"✅ Position tracking: ACTIVE")
    print(f"✅ Win/Loss tracking: ACTIVE")
    print('='*60 + '\n')
    
    # Initialize
    acc_info = mt5.account_info()
    if acc_info:
        daily_stats['start_balance'] = acc_info.balance
        print(f"Starting balance: ${acc_info.balance:.2f}\n")
    
    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*60)
            
            # Update position tracker FIRST
            position_tracker.update()
            
            # Check daily loss limit
            if not check_daily_loss_limit():
                print("⏸ Daily loss limit exceeded. Sleeping...")
                print("💤 Bot will sleep until next day...")
                time.sleep(3600)
                continue
            
            # Display status
            acc_info = mt5.account_info()
            if acc_info:
                daily_pnl = acc_info.balance - daily_stats.get('start_balance', acc_info.balance)
                daily_pnl_pct = (daily_pnl / daily_stats['start_balance'] * 100) if daily_stats.get('start_balance') else 0
                print(f"💰 Balance: ${acc_info.balance:.2f} | Equity: ${acc_info.equity:.2f}")
                print(f"📊 Daily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%)")
                print(f"📈 Open positions: {position_tracker.get_count()}/{max_total_positions}")
                print(f"📋 Trades: {daily_stats['trades_today']} | W:{daily_stats['wins_today']} L:{daily_stats['losses_today']}")
                
                if daily_stats['trades_today'] > 0:
                    win_rate = (daily_stats['wins_today'] / daily_stats['trades_today'] * 100)
                    print(f"📊 Win Rate: {win_rate:.1f}%")
            
            # Process symbols
            for symbol in symbols:
                try:
                    print(f"\n🔍 Analyzing {symbol}...")
                    
                    df = get_ohlcv(symbol, 300)
                    if df is None:
                        print(f"⚠ No data for {symbol}")
                        continue
                    
                    sig = ml_filtered_signal(symbol, clf, df)
                    
                    if sig:
                        print(f"🎯 Signal: {symbol} - {sig.upper()}")
                        open_trade(symbol, sig, clf, reg_lot)
                    else:
                        print(f"⭕ No signal for {symbol}")
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    trade_logger.log_error("PROCESSING", symbol, str(e))
            
            # Manage trailing stops
            try:
                print(f"\n🎯 Managing trailing stops...")
                manage_trailing_positions()
            except Exception as e:
                print(f"❌ Trailing stop error: {e}")
                trade_logger.log_error("TRAILING", "ALL", str(e))
            
            print(f"\n{'='*60}")
            print(f"✅ Cycle #{cycle} complete. Sleeping 1 minute...")
            print('='*60 + '\n')
            
            time.sleep(60)
    
    except KeyboardInterrupt:
        print('\n🛑 STOPPING (KeyboardInterrupt)')
    
    except Exception as e:
        print(f'\n💥 FATAL ERROR: {e}')
        trade_logger.log_error("FATAL", "SYSTEM", str(e))
        import traceback
        traceback.print_exc()
    
    finally:
        if mt5.initialize():
            mt5.shutdown()
        print("\n🔌 MT5 closed")
        print("👋 Goodbye!\n")

# ============ MAIN ENTRY POINT ============

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Trading Bot - All Issues Resolved')
    parser.add_argument('--mode', choices=['collect', 'train', 'live'], 
                       default='live', 
                       help='Operation mode')
    parser.add_argument('--data', default='training_data.csv', 
                       help='CSV file for training')
    parser.add_argument('--symbols', nargs='*', default=symbols, 
                       help='Trading symbols')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 TRADING BOT - ALL ISSUES FIXED")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print("\n✅ FIXES APPLIED:")
    print("  1. ✅ Simplified Trailing Stop (NO ML dependency)")
    print("  2. ✅ Real-time Position Tracking")
    print("  3. ✅ Accurate Win/Loss Tracking")
    print("  4. ✅ Closed Position Detection")
    print("  5. ✅ Daily Stats with Closed Tickets Set")
    print("="*60 + "\n")
    
    run_live(args.symbols, mode=args.mode, data_csv=args.data)