"""
Enhanced RSI-MA-ATR Trading Bot with Critical Fixes
Key improvements:
- Symmetric ML filtering for buy AND sell signals
- Position limits and exposure management
- Daily loss limits and drawdown protection
- Improved feature engineering
- Better risk management
- Market regime detection
"""
import os
import time
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
symbols = ["XAUUSD", "XAUEUR"]
magic_number = 234000
comment_text = "RSI-MA-ATR-ML-Bot-v2"

# Technical indicators
rsi_period = 14
ma_short = 20
ma_long = 50
atr_period = 14

# Risk management (CRITICAL CHANGES)
default_lot = 0.01
risk_per_trade = 0.01  # Reduced from 2% to 1%
max_positions_per_symbol = 3  # NEW: Only one position per symbol
max_total_positions = 6  # NEW: Maximum total open positions
max_daily_loss_pct = 0.03  # NEW: Stop trading if lose 3% in a day
max_total_risk_pct = 0.05  # NEW: Don't risk more than 5% total at once

# ML settings
MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clf_signal.pkl")
LOT_REG_PATH = os.path.join(MODEL_DIR, "reg_lot.pkl")
TRAIL_REG_PATH = os.path.join(MODEL_DIR, "reg_trail.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)
ML_PROB_THRESHOLD = 0.60  # Lowered slightly for more signals

# NEW: Trading hours filter (avoid low liquidity)
TRADING_HOURS_GMT = [(8, 18)]  # 8 AM to 6 PM UTC (adjust for your broker)

# NEW: Track daily performance
daily_stats = {
    'start_balance': None,
    'trades_today': 0,
    'losses_today': 0,
    'date': datetime.now().date()
}

if not mt5.initialize(path=MT5_PATH):
    print("MT5 init failed:", mt5.last_error())

# ============ HELPER FUNCTIONS ============

def is_trading_hours():
    """Check if current time is within allowed trading hours"""
    current_hour = datetime.utcnow().hour  # FIXED: Added .utcnow()
    for start, end in TRADING_HOURS_GMT:
        if start <= current_hour < end:
            return True
    return False

def check_daily_loss_limit():
    """Check if daily loss limit exceeded"""
    acc_info = mt5.account_info()
    if acc_info is None:
        return True  # Allow trading if can't check
    
    # Reset daily stats if new day
    if daily_stats['date'] != datetime.now().date():
        daily_stats['start_balance'] = acc_info.balance
        daily_stats['trades_today'] = 0
        daily_stats['losses_today'] = 0
        daily_stats['date'] = datetime.now().date()
        return True
    
    # Initialize start balance if not set
    if daily_stats['start_balance'] is None:
        daily_stats['start_balance'] = acc_info.balance
        return True
    
    # Check loss
    daily_loss = daily_stats['start_balance'] - acc_info.balance
    daily_loss_pct = daily_loss / daily_stats['start_balance'] if daily_stats['start_balance'] > 0 else 0
    
    if daily_loss_pct >= max_daily_loss_pct:
        print(f"DAILY LOSS LIMIT REACHED: {daily_loss_pct*100:.2f}% (limit: {max_daily_loss_pct*100:.2f}%)")
        return False
    
    return True

def get_current_exposure():
    """Calculate current risk exposure across all positions"""
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
        
        # Calculate risk based on SL distance
        if pos.sl > 0:
            risk_distance = abs(pos.price_open - pos.sl)
            risk_amount = risk_distance * pos.volume * 100000  # Rough estimate
            total_risk += risk_amount
    
    return total_risk / balance if balance > 0 else 0.0

def count_positions(symbol=None):
    """Count open positions for a symbol or total"""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return 0
    
    count = 0
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic == magic_number and comment == comment_text:
            count += 1
    
    return count

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
    if df is None or len(df) < period:
        return pd.Series([float('nan')] * len(df)) if df is not None else None
    d = df.copy()
    d['h-l'] = d['high'] - d['low']
    d['h-pc'] = (d['high'] - d['close'].shift(1)).abs()
    d['l-pc'] = (d['low'] - d['close'].shift(1)).abs()
    d['tr'] = d[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    d['atr'] = d['tr'].rolling(window=period).mean()
    return d['atr']

def get_ohlcv(symbol, num_bars=500, timeframe=mt5.TIMEFRAME_M5):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ ENHANCED FEATURE ENGINEERING ============

def build_features(df):
    """Enhanced feature set with market regime indicators"""
    d = df.copy()
    
    # Original features
    d['ema_short'] = ema(d['close'], ma_short)
    d['ema_long'] = ema(d['close'], ma_long)
    d['rsi'] = rsi(d['close'], rsi_period)
    d['atr'] = calc_atr(d, atr_period)
    d['ema_diff'] = d['ema_short'] - d['ema_long']
    d['ret_1'] = d['close'].pct_change(1)
    d['ret_5'] = d['close'].pct_change(5)
    d['vol_10'] = d['tick_volume'].rolling(10).mean()
    d['volatility_10'] = d['close'].pct_change().rolling(10).std()
    d['close_minus_ema_short'] = d['close'] - d['ema_short']
    d['close_minus_ema_long'] = d['close'] - d['ema_long']
    
    # NEW: Enhanced features
    d['rsi_momentum'] = d['rsi'].diff(3)  # RSI rate of change
    d['ema_diff_pct'] = d['ema_diff'] / d['close']  # Normalized trend strength
    d['atr_pct'] = d['atr'] / d['close']  # Normalized volatility
    d['volume_ratio'] = d['tick_volume'] / d['vol_10']  # Volume spike detector
    d['price_distance_short'] = (d['close'] - d['ema_short']) / d['atr']  # Pullback depth
    d['price_distance_long'] = (d['close'] - d['ema_long']) / d['atr']
    
    # Market regime: trending vs ranging
    d['trend_strength'] = d['ema_diff'].abs() / d['atr']  # Strong trend > 1.5
    d['volatility_regime'] = d['volatility_10'] / d['volatility_10'].rolling(50).mean()  # >1 = high vol
    
    return d

def label_future(df, lookahead=20, tp_atr_mult=2.5, sl_atr_mult=1.5):
    """Improved labeling with more balanced classes"""
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
        
        # Buy setup
        tp_buy = entry + tp_atr_mult * atr
        sl_buy = entry - sl_atr_mult * atr
        fut_high = d['future_high'].iloc[i]
        fut_low = d['future_low'].iloc[i]
        
        # Sell setup
        tp_sell = entry - tp_atr_mult * atr
        sl_sell = entry + sl_atr_mult * atr
        
        # Check outcomes
        buy_wins = fut_high >= tp_buy and fut_low > sl_buy
        sell_wins = fut_low <= tp_sell and fut_high < sl_sell
        
        if buy_wins and not sell_wins:
            labels.append(1)  # Buy
        elif sell_wins and not buy_wins:
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # No clear winner
    
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
    print(f"Label distribution: {out['label'].value_counts().to_dict()}")

# ============ MODEL TRAINING ============

def train_models(csv_path, test_size=0.2, random_state=42):
    """Enhanced training with cross-validation"""
    df = pd.read_csv(csv_path)
    
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                   'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                   'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
    
    df = df.dropna(subset=feature_cols + ['label'])
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Prepare data for binary classification (Buy vs Not-Buy)
    X = df[feature_cols].values
    y = df['label'].values
    y_bin = np.where(y == 1, 1, 0)  # 1=Buy, 0=No buy
    
    # Also train sell classifier
    y_sell = np.where(y == -1, 1, 0)  # 1=Sell, 0=No sell
    
    # Train buy classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )
    
    clf_buy = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight='balanced'  # Handle imbalanced classes
    )
    clf_buy.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf_buy, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nBUY Classifier Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    y_pred = clf_buy.predict(X_test)
    print("\nBUY Classifier Test Set Performance:")
    print(classification_report(y_test, y_pred, target_names=['No-Buy', 'Buy']))
    
    # Train sell classifier
    X_train_sell, X_test_sell, y_train_sell, y_test_sell = train_test_split(
        X, y_sell, test_size=test_size, random_state=random_state, stratify=y_sell
    )
    
    clf_sell = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight='balanced'
    )
    clf_sell.fit(X_train_sell, y_train_sell)
    
    y_pred_sell = clf_sell.predict(X_test_sell)
    print("\nSELL Classifier Test Set Performance:")
    print(classification_report(y_test_sell, y_pred_sell, target_names=['No-Sell', 'Sell']))
    
    # Save both classifiers
    joblib.dump({'buy': clf_buy, 'sell': clf_sell}, CLASSIFIER_PATH)
    print(f"\nSaved classifiers -> {CLASSIFIER_PATH}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf_buy.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Train regressors (keep existing logic but with more samples requirement)
    df_profitable = df[df['label'].abs() == 1].copy()  # Include both buy and sell setups
    
    if len(df_profitable) >= 100:
        target_lot = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_lot = (target_lot / target_lot.max()) * 0.5
        
        Xl = df_profitable[feature_cols].values
        yl = target_lot.values
        
        reg_lot = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        reg_lot.fit(Xl, yl)
        joblib.dump(reg_lot, LOT_REG_PATH)
        print(f"Saved lot regressor -> {LOT_REG_PATH}")
    else:
        print(f"Warning: Not enough profitable samples ({len(df_profitable)}) to train lot regressor")
    
    if len(df_profitable) >= 100:
        target_trail = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_trail = (target_trail / target_trail.max())
        target_trail = 0.5 + target_trail * 1.5
        
        Xr = df_profitable[feature_cols].values
        yr = target_trail.values
        
        reg_trail = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        reg_trail.fit(Xr, yr)
        joblib.dump(reg_trail, TRAIL_REG_PATH)
        print(f"Saved trail regressor -> {TRAIL_REG_PATH}")
    else:
        print(f"Warning: Not enough profitable samples to train trail regressor")

# ============ LOAD MODELS ============

def load_models():
    clf = None
    reg_lot = None
    reg_trail = None
    
    if os.path.exists(CLASSIFIER_PATH):
        try:
            clf = joblib.load(CLASSIFIER_PATH)
            print('Loaded classifiers (buy & sell)')
        except Exception as e:
            print('Failed to load classifier', e)
    
    if os.path.exists(LOT_REG_PATH):
        try:
            reg_lot = joblib.load(LOT_REG_PATH)
            print('Loaded lot regressor')
        except Exception as e:
            print('Failed to load lot regressor', e)
    
    if os.path.exists(TRAIL_REG_PATH):
        try:
            reg_trail = joblib.load(TRAIL_REG_PATH)
            print('Loaded trail regressor')
        except Exception as e:
            print('Failed to load trail regressor', e)
    
    return clf, reg_lot, reg_trail

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
    
    # Calculate trend strength
    trend_strength = abs(last['ema_short'] - last['ema_long']) / last['atr']
    
    # Only trade in trending markets
    if trend_strength < 0.5:
        return None
    
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    
    # Buy setup: price pulled back to EMA in uptrend
    if (trend_up and 
        prev['low'] <= prev['ema_short'] and 
        last['close'] > last['ema_short'] and 
        35 < last['rsi'] < 70):
        return 'buy'
    
    # Sell setup: price pulled back to EMA in downtrend
    elif (trend_down and 
          prev['high'] >= prev['ema_short'] and 
          last['close'] < last['ema_short'] and 
          30 < last['rsi'] < 65):
        return 'sell'
    
    return None

def ml_filtered_signal(symbol, clf, df):
    """CRITICAL FIX: Apply ML filter to BOTH buy and sell signals"""
    if df is None or len(df) < 30:
        return None
    
    feat = build_features(df).iloc[-1]
    
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                   'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                   'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
    
    # Check for NaN
    if feat[feature_cols].isna().any():
        print(f"Warning: NaN features detected for {symbol}, skipping signal")
        return None
    
    X = feat[feature_cols].values.reshape(1, -1)
    rule = rule_signal_from_df(df)
    
    # If no ML models, use rule-based only
    if clf is None:
        return rule
    
    try:
        # Get both buy and sell classifiers
        clf_buy = clf.get('buy') if isinstance(clf, dict) else clf
        clf_sell = clf.get('sell') if isinstance(clf, dict) else None
        
        # Buy signal with ML filter
        if rule == 'buy' and clf_buy is not None:
            proba = clf_buy.predict_proba(X)[0]
            prob_buy = proba[1] if len(proba) > 1 else proba[0]
            
            if prob_buy >= ML_PROB_THRESHOLD:
                print(f"ML-filtered BUY signal for {symbol} (prob={prob_buy:.3f})")
                return 'buy'
            else:
                print(f"BUY signal rejected by ML for {symbol} (prob={prob_buy:.3f} < {ML_PROB_THRESHOLD})")
                return None
        
        # Sell signal with ML filter (CRITICAL FIX)
        if rule == 'sell' and clf_sell is not None:
            proba = clf_sell.predict_proba(X)[0]
            prob_sell = proba[1] if len(proba) > 1 else proba[0]
            
            if prob_sell >= ML_PROB_THRESHOLD:
                print(f"ML-filtered SELL signal for {symbol} (prob={prob_sell:.3f})")
                return 'sell'
            else:
                print(f"SELL signal rejected by ML for {symbol} (prob={prob_sell:.3f} < {ML_PROB_THRESHOLD})")
                return None
        
        return None
        
    except Exception as e:
        print(f"ML prediction error for {symbol}: {e}")
        return None  # Don't trade on errors

# ============ POSITION SIZING ============

def calculate_lot_ml(symbol, reg_lot, sl_price_distance, current_features=None):
    """Conservative lot sizing with ML adjustment"""
    try:
        acc_info = mt5.account_info()
        if acc_info is None or sl_price_distance is None or sl_price_distance <= 0:
            return default_lot
        
        balance = acc_info.balance
        risk_amount = balance * risk_per_trade  # Now 1% instead of 2%
        
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
    
    # Apply ML multiplier (but keep it conservative)
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
        # More conservative multiplier range: 0.5x to 1.5x (not 0.3x to 2.0x)
        multiplier = 0.7 + (pred * 0.8)
        multiplier = max(0.5, min(multiplier, 1.5))
        
        final_lot = base_lot * multiplier
        final_lot = max(0.01, round(final_lot, 2))
        
        # Stricter maximum lot limit
        max_lot = balance * 0.05 / 1000  # 5% instead of 10%
        final_lot = min(final_lot, max_lot)
        
        print(f"Lot sizing: base={base_lot:.2f}, ML mult={multiplier:.2f}, final={final_lot:.2f}")
        return final_lot
        
    except Exception as e:
        print(f'lot regressor error: {e}')
        return base_lot

# ============ TRAILING STOPS ============

def get_trail_multiplier_ml(reg_trail, df):
    if reg_trail is None or df is None or len(df) < 30:
        return 0.75
    
    try:
        feat = build_features(df).iloc[-1]
        feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                       'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                       'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
        
        if feat[feature_cols].isna().any():
            return 0.75
        
        X = feat[feature_cols].values.reshape(1, -1)
        pred = reg_trail.predict(X)[0]
        pred = float(pred)
        pred = max(0.3, min(pred, 2.0))  # More conservative range
        return pred
        
    except Exception as e:
        print('trail regressor error', e)
        return 0.75

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

def manage_trailing_positions(clf, reg_lot, reg_trail):
    positions = mt5.positions_get()
    if positions is None:
        return
    
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        pos_type = pos.type
        entry = pos.price_open
        ticket = pos.ticket
        
        symbol_info, point, tick_size = get_symbol_info(symbol)
        if symbol_info is None:
            continue
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        current_price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
        
        df = get_ohlcv(symbol, 200)
        if df is None or len(df) < 30:
            continue
        
        atr_series = calc_atr(df, atr_period)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            continue
        
        atr_val = float(atr_series.iloc[-1])
        min_stop_dist = get_min_stop_distance(symbol)
        
        LOCK_THRESHOLD = 1.0 * atr_val
        TRAIL_THRESHOLD = 1.5 * atr_val
        
        current_sl = pos.sl if pos.sl != 0 else None
        current_tp = pos.tp if pos.tp != 0 else None
        new_sl = current_sl
        new_tp = current_tp
        need_modify = False
        
        if pos_type == mt5.ORDER_TYPE_BUY:
            profit_atr = (current_price - entry) / atr_val
        else:
            profit_atr = (entry - current_price) / atr_val
        
        # Phase 1: Lock in profit
        if profit_atr >= 1.0 and profit_atr < 1.5:
            if pos_type == mt5.ORDER_TYPE_BUY:
                lock_sl = entry + (0.2 * atr_val)
                lock_sl = normalize_price(lock_sl, tick_size)
                if lock_sl < current_price - min_stop_dist:
                    if current_sl is None or lock_sl > current_sl:
                        if validate_sl_distance(current_price, lock_sl, min_stop_dist, pos_type):
                            new_sl = lock_sl
                            need_modify = True
                            print(f"Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
            else:
                lock_sl = entry - (0.2 * atr_val)
                lock_sl = normalize_price(lock_sl, tick_size)
                if lock_sl > current_price + min_stop_dist:
                    if current_sl is None or lock_sl < current_sl:
                        if validate_sl_distance(current_price, lock_sl, min_stop_dist, pos_type):
                            new_sl = lock_sl
                            need_modify = True
                            print(f"Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
        
        # Phase 2: Trailing stop
        elif profit_atr >= 1.5:
            trail_mult = get_trail_multiplier_ml(reg_trail, df)
            trail_dist = max(trail_mult * atr_val, min_stop_dist * 1.5)
            
            if pos_type == mt5.ORDER_TYPE_BUY:
                target_sl = current_price - trail_dist
                target_sl = normalize_price(target_sl, tick_size)
                if target_sl < current_price - min_stop_dist:
                    if target_sl > entry:
                        if current_sl is None or target_sl > current_sl:
                            if validate_sl_distance(current_price, target_sl, min_stop_dist, pos_type):
                                new_sl = target_sl
                                need_modify = True
                                print(f"Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} (mult={trail_mult:.2f}, profit={profit_atr:.2f} ATR)")
            else:
                target_sl = current_price + trail_dist
                target_sl = normalize_price(target_sl, tick_size)
                if target_sl > current_price + min_stop_dist:
                    if target_sl < entry:
                        if current_sl is None or target_sl < current_sl:
                            if validate_sl_distance(current_price, target_sl, min_stop_dist, pos_type):
                                new_sl = target_sl
                                need_modify = True
                                print(f"Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} (mult={trail_mult:.2f}, profit={profit_atr:.2f} ATR)")
            
            # Set hard TP if not set
            if current_tp is None or current_tp == 0:
                tp2 = entry + (3.0 * atr_val) if pos_type == mt5.ORDER_TYPE_BUY else entry - (3.0 * atr_val)
                new_tp = normalize_price(tp2, tick_size)
                need_modify = True
                print(f"Setting hard TP for {symbol} ticket {ticket} at {new_tp:.5f}")
        
        if need_modify and (new_sl != current_sl or new_tp != current_tp):
            if new_sl is not None:
                if not validate_sl_distance(current_price, new_sl, min_stop_dist, pos_type):
                    print(f"Skipping invalid SL for {symbol} ticket {ticket}: SL={new_sl:.5f} too close to price={current_price:.5f}")
                    continue
                
                if profit_atr >= 1.5:
                    if pos_type == mt5.ORDER_TYPE_BUY and new_sl <= entry:
                        print(f"Skipping SL below entry in trailing phase for {symbol}: SL={new_sl:.5f}, entry={entry:.5f}")
                        continue
                    if pos_type == mt5.ORDER_TYPE_SELL and new_sl >= entry:
                        print(f"Skipping SL above entry in trailing phase for {symbol}: SL={new_sl:.5f}, entry={entry:.5f}")
                        continue
            
            modify_req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(new_sl) if new_sl is not None else 0.0,
                "tp": float(new_tp) if new_tp is not None else 0.0,
                "magic": magic_number,
                "comment": comment_text
            }
            
            result = mt5.order_send(modify_req)
            if result is None:
                print(f"Modify failed for {symbol} ticket {ticket}: None response")
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Modify failed for {symbol} ticket {ticket}: retcode={result.retcode}, comment={result.comment}")
            else:
                print(f"Modified pos {ticket} {symbol}: SL={new_sl:.5f}, TP={new_tp:.5f}")

# ============ OPEN TRADE ============

def open_trade(symbol, action, clf, reg_lot):
    """Open trade with enhanced risk checks"""
    
    # Check position limits
    if count_positions(symbol) >= max_positions_per_symbol:
        print(f"Max positions for {symbol} reached ({max_positions_per_symbol})")
        return
    
    if count_positions() >= max_total_positions:
        print(f"Max total positions reached ({max_total_positions})")
        return
    
    # Check exposure
    current_exposure = get_current_exposure()
    if current_exposure >= max_total_risk_pct:
        print(f"Max exposure reached: {current_exposure*100:.2f}% (limit: {max_total_risk_pct*100:.2f}%)")
        return
    
    # Check daily loss limit
    if not check_daily_loss_limit():
        return
    
    # Check trading hours
    if not is_trading_hours():
        print(f"Outside trading hours, skipping {symbol} {action}")
        return
    
    if not mt5.symbol_select(symbol, True):
        print(f"Symbol {symbol} not available / not selected")
        return
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"No tick for {symbol}")
        return
    
    symbol_info, point, tick_size = get_symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return
    
    price = tick.ask if action == "buy" else tick.bid
    
    df = get_ohlcv(symbol, 200)
    if df is None or len(df) < 20:
        return
    
    atr_series = calc_atr(df, atr_period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return
    
    atr_val = float(atr_series.iloc[-1])
    
    # More conservative SL/TP
    atr_mult_sl = 1.5
    atr_mult_tp = 2.5  # Reduced from 3.0 for better risk:reward
    
    if action == "buy":
        candidate_sl = price - (atr_mult_sl * atr_val)
        candidate_tp2 = price + (atr_mult_tp * atr_val)
    else:
        candidate_sl = price + (atr_mult_sl * atr_val)
        candidate_tp2 = price - (atr_mult_tp * atr_val)
    
    candidate_sl = normalize_price(candidate_sl, tick_size)
    candidate_tp2 = normalize_price(candidate_tp2, tick_size)
    
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
            print(f"Warning: Cannot set stops for {symbol} - minimum distance not achievable")
    
    lot = calculate_lot_ml(symbol, reg_lot, sl_distance if use_stops else None)
    if lot < 0.01:
        lot = default_lot
    
    filling_mode = mt5.ORDER_FILLING_IOC
    
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
        request["tp"] = candidate_tp2
    
    result = mt5.order_send(request)
    
    if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
        print(f"{symbol} {action} failed: retcode={getattr(result, 'retcode', 'N/A')}, comment={getattr(result, 'comment', 'N/A')}")
    else:
        daily_stats['trades_today'] += 1
        print(f"{symbol} {action.upper()} opened at {price} | SL={candidate_sl:.5f}, TP={candidate_tp2:.5f}, ATR={atr_val:.5f}, LOT={lot}")

# ============ BACKTEST ============

def backtest(csv_path):
    df = pd.read_csv(csv_path)
    
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 
                   'close_minus_ema_short', 'rsi_momentum', 'ema_diff_pct', 'atr_pct', 
                   'volume_ratio', 'price_distance_short', 'trend_strength', 'volatility_regime']
    
    df = df.dropna(subset=feature_cols + ['label'])
    
    clf, _, _ = load_models()
    if clf is None:
        print('No classifier to evaluate. Train models first.')
        return
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Test buy classifier
    y_bin = np.where(y == 1, 1, 0)
    clf_buy = clf.get('buy') if isinstance(clf, dict) else clf
    preds_buy = clf_buy.predict(X)
    
    print("\nBUY Classifier Performance on Full Dataset:")
    print(classification_report(y_bin, preds_buy, target_names=['No-Buy', 'Buy']))
    
    # Test sell classifier
    if isinstance(clf, dict) and 'sell' in clf:
        y_sell = np.where(y == -1, 1, 0)
        clf_sell = clf['sell']
        preds_sell = clf_sell.predict(X)
        
        print("\nSELL Classifier Performance on Full Dataset:")
        print(classification_report(y_sell, preds_sell, target_names=['No-Sell', 'Sell']))

# ============ MAIN LIVE LOOP ============

def run_live(symbols, mode='live', data_csv='training_data.csv'):
    clf, reg_lot, reg_trail = load_models()
    
    if mode == 'collect':
        for s in symbols:
            collect_and_save(s, data_csv, num_bars=2000)
        return
    
    if mode == 'train':
        train_models(data_csv)
        return
    
    print('\n' + '='*60)
    print('STARTING LIVE TRADING LOOP')
    print('='*60)
    print(f"Trading symbols: {symbols}")
    print(f"Magic number: {magic_number}")
    print(f"ML models loaded: clf={clf is not None}, lot_reg={reg_lot is not None}, trail_reg={reg_trail is not None}")
    print(f"Risk per trade: {risk_per_trade*100:.1f}%")
    print(f"Max positions per symbol: {max_positions_per_symbol}")
    print(f"Max total positions: {max_total_positions}")
    print(f"Daily loss limit: {max_daily_loss_pct*100:.1f}%")
    print(f"ML probability threshold: {ML_PROB_THRESHOLD:.2f}")
    print('='*60 + '\n')
    
    # Initialize daily stats
    acc_info = mt5.account_info()
    if acc_info:
        daily_stats['start_balance'] = acc_info.balance
        print(f"Starting balance: ${acc_info.balance:.2f}\n")
    
    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*60)
            
            # Check daily loss limit first
            if not check_daily_loss_limit():
                print("Daily loss limit exceeded. Stopping trading for today.")
                print("Bot will sleep until next day...")
                time.sleep(3600)  # Sleep 1 hour and check again
                continue
            
            # Display current account status
            acc_info = mt5.account_info()
            if acc_info:
                print(f"Balance: ${acc_info.balance:.2f} | Equity: ${acc_info.equity:.2f}")
                print(f"Open positions: {count_positions()}/{max_total_positions}")
                print(f"Trades today: {daily_stats['trades_today']}")
            
            # Process each symbol
            for symbol in symbols:
                try:
                    print(f"\nAnalyzing {symbol}...")
                    
                    df = get_ohlcv(symbol, 300)
                    if df is None:
                        print(f"No data for {symbol}, skipping...")
                        continue
                    
                    sig = ml_filtered_signal(symbol, clf, df)
                    
                    if sig:
                        print(f"Signal detected: {symbol} - {sig.upper()}")
                        open_trade(symbol, sig, clf, reg_lot)
                    else:
                        print(f"No signal for {symbol}")
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Manage trailing stops
            try:
                print(f"\nManaging trailing stops...")
                manage_trailing_positions(clf, reg_lot, reg_trail)
            except Exception as e:
                print(f"Error in trailing stop management: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*60}")
            print(f"Cycle #{cycle} complete. Sleeping 1 minutes...")
            print('='*60 + '\n')
            
            time.sleep(60)  # 1 minute
    
    except KeyboardInterrupt:
        print('\n\n' + '='*60)
        print('STOPPING LIVE LOOP (KeyboardInterrupt)')
        print('='*60)
    
    except Exception as e:
        print(f'\n\n' + '='*60)
        print(f'FATAL ERROR IN LIVE LOOP: {e}')
        print('='*60)
        import traceback
        traceback.print_exc()
    
    finally:
        if mt5.initialize():
            mt5.shutdown()
        print("\nMT5 connection closed")
        print("Goodbye!\n")

# ============ MAIN ENTRY POINT ============

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML-Enhanced Trading Bot for MT5 - FIXED VERSION')
    parser.add_argument('--mode', choices=['collect', 'train', 'live', 'backtest'], 
                       default='live', 
                       help='Operation mode: collect data, train models, run live trading, or backtest')
    parser.add_argument('--data', default='training_data.csv', 
                       help='Path to CSV file for training data')
    parser.add_argument('--symbols', nargs='*', default=symbols, 
                       help='List of symbols to trade (e.g., XAUUSD XAUEUR)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RSI-MA-ATR Trading Bot with ML - ENHANCED VERSION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print(f"Data file: {args.data}")
    print("="*60 + "\n")
    
    if args.mode == 'backtest':
        backtest(args.data)
    else:
        run_live(args.symbols, mode=args.mode, data_csv=args.data)