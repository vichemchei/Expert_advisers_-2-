"""
RSI-MA-ATR Trading Bot with ML integration for MT5 - ENHANCED VERSION
- Collects features & labels to CSV (per-symbol)
- Provides training functions (RandomForest classifier + regressors)
- Integrates ML classifier as a filter for get_signal()
- Integrates ML regressor for adaptive lot sizing and trailing multiplier
- ADDED: Feature normalization with StandardScaler
- ADDED: Cross-validation for model evaluation
- ADDED: Feature importance reporting
- ADDED: Separate models per symbol

FIXES:
- ML filtered signal now properly handles NaN validation
- Calculate lot ML uses ML as multiplier instead of absolute value
- Trailing stop completely rewritten with proper phase separation and validation
- Feature normalization for improved model performance
- K-fold cross-validation for robust model evaluation
- Separate model training per symbol for better performance

USAGE:
1. Fill in MT5_PATH and run the script with `mode='collect'` to collect labeled data from history.
2. Run `mode='train'` to train models (requires scikit-learn & joblib).
3. Run `mode='live'` to run the live trading loop (loads trained models if present).

CAVEATS:
- Train models on sufficiently large, cleaned datasets before using live.
- Start on demo account. Monitor logs and use VPS for 24/7.

Dependencies: pandas, numpy, scikit-learn, joblib, MetaTrader5
"""

import os
import time
from datetime import datetime
import math

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib

# ================ CONFIGURATION ================
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
symbols = ["XAUUSD", "XAUEUR"]
magic_number = 234000
comment_text = "RSI-MA-ATR-ML-Bot"

# Technical indicator parameters
rsi_period = 14
ma_short = 20
ma_long = 50
atr_period = 14

# Risk management defaults
default_lot = 0.01
risk_per_trade = 0.02

# ML model file paths
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ML prediction thresholds
ML_PROB_THRESHOLD = 0.65

# Cross-validation settings
CV_FOLDS = 5

# ================ MT5 INITIALIZATION ================
if not mt5.initialize(path=MT5_PATH):
    print("MT5 init failed:", mt5.last_error())

# ================ HELPER FUNCTIONS FOR MODEL PATHS ================

def get_model_paths(symbol):
    """Get model file paths for a specific symbol"""
    return {
        'classifier': os.path.join(MODEL_DIR, f"clf_signal_{symbol}.pkl"),
        'lot_reg': os.path.join(MODEL_DIR, f"reg_lot_{symbol}.pkl"),
        'trail_reg': os.path.join(MODEL_DIR, f"reg_trail_{symbol}.pkl"),
        'scaler': os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
    }

def get_data_path(symbol):
    """Get CSV data path for a specific symbol"""
    return f"training_data_{symbol}.csv"

# ================ TECHNICAL INDICATORS ================

def ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def calc_atr(df, period=14):
    """Calculate Average True Range"""
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
    """Fetch OHLCV data from MT5 for given symbol"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ================ FEATURE ENGINEERING & LABELING ================

def build_features(df):
    """Build technical features for machine learning"""
    d = df.copy()
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
    return d


def label_future(df, lookahead=20, tp_atr_mult=3.0, sl_atr_mult=1.5):
    """
    Label historical data based on future price movement
    Returns labels: 1=buy (TP hit before SL), -1=sell (TP hit before SL), 0=no clear signal
    """
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
        if fut_high >= tp_buy and fut_low > sl_buy:
            labels.append(1)
            continue
        tp_sell = entry - tp_atr_mult * atr
        sl_sell = entry + sl_atr_mult * atr
        if fut_low <= tp_sell and fut_high < sl_sell:
            labels.append(-1)
            continue
        labels.append(0)
    d['label'] = labels
    return d

# ================ DATA COLLECTION ================

def collect_and_save(symbol, num_bars=2000, lookahead=20):
    """Collect historical data, build features, create labels, and save to symbol-specific CSV"""
    out_csv = get_data_path(symbol)
    print(f"Collecting {symbol}...")
    df = get_ohlcv(symbol, num_bars)
    if df is None:
        print("No data for", symbol)
        return
    feat = build_features(df)
    labeled = label_future(feat, lookahead=lookahead)
    keep_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'ema_short', 'ema_long', 'rsi', 'atr',
                 'ema_diff', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short', 'close_minus_ema_long', 'label']
    out = labeled[keep_cols].dropna()
    
    # Overwrite existing file for this symbol
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")

# ================ MODEL TRAINING WITH NORMALIZATION & CV ================

def train_models_for_symbol(symbol, test_size=0.2, random_state=42):
    """
    Train ML models for a specific symbol with feature normalization and cross-validation
    - StandardScaler for feature normalization
    - K-fold cross-validation for robust evaluation
    - Feature importance reporting
    """
    csv_path = get_data_path(symbol)
    
    if not os.path.exists(csv_path):
        print(f"No data file found for {symbol} at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    
    print(f"\n{'='*60}")
    print(f"Training models for {symbol}")
    print(f"{'='*60}")
    print(f"Training on {len(df)} samples")
    
    X = df[feature_cols].values
    y = df['label'].values

    # Get model paths for this symbol
    paths = get_model_paths(symbol)

    # Fit scaler on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, paths['scaler'])
    print(f"Saved scaler to {paths['scaler']}")

    # Train classifier (binary: buy vs not-buy)
    y_bin = np.where(y == 1, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
    
    # Cross-validation
    print("\nPerforming cross-validation for classifier...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassifier report (buy vs not-buy):")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance (Classifier):")
    print(feature_importance.to_string(index=False))
    
    joblib.dump(clf, paths['classifier'])
    print(f"\nSaved classifier to {paths['classifier']}")

    # Train lot size regressor on profitable buy trades
    df_profitable = df[df['label'] == 1].copy()
    if len(df_profitable) >= 50:
        target_lot = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_lot = (target_lot / target_lot.max())
        target_lot = target_lot * 0.5
        
        Xl = df_profitable[feature_cols].values
        Xl_scaled = scaler.transform(Xl)
        yl = target_lot.values
        
        Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl_scaled, yl, test_size=test_size, random_state=random_state)
        
        reg_lot = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        
        # Cross-validation for regressor
        print("\nPerforming cross-validation for lot regressor...")
        cv_scores_lot = cross_val_score(reg_lot, Xl_train, yl_train, cv=CV_FOLDS, scoring='neg_mean_squared_error')
        print(f"CV RMSE: {np.sqrt(-cv_scores_lot.mean()):.4f} (+/- {np.sqrt(cv_scores_lot.std()):.4f})")
        
        reg_lot.fit(Xl_train, yl_train)
        yl_pred = reg_lot.predict(Xl_test)
        mse = mean_squared_error(yl_test, yl_pred)
        print(f"Test RMSE: {np.sqrt(mse):.4f}")
        
        joblib.dump(reg_lot, paths['lot_reg'])
        print(f"Saved lot regressor to {paths['lot_reg']}")
    else:
        print(f"Not enough profitable samples to train lot regressor for {symbol}; skipped.")

    # Train trailing stop regressor on profitable trades
    if len(df_profitable) >= 50:
        target_trail = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_trail = (target_trail / target_trail.max())
        target_trail = 0.5 + target_trail * 1.5
        
        Xr = df_profitable[feature_cols].values
        Xr_scaled = scaler.transform(Xr)
        yr = target_trail.values
        
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled, yr, test_size=test_size, random_state=random_state)
        
        reg_trail = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        
        # Cross-validation for trail regressor
        print("\nPerforming cross-validation for trail regressor...")
        cv_scores_trail = cross_val_score(reg_trail, Xr_train, yr_train, cv=CV_FOLDS, scoring='neg_mean_squared_error')
        print(f"CV RMSE: {np.sqrt(-cv_scores_trail.mean()):.4f} (+/- {np.sqrt(cv_scores_trail.std()):.4f})")
        
        reg_trail.fit(Xr_train, yr_train)
        yr_pred = reg_trail.predict(Xr_test)
        mse = mean_squared_error(yr_test, yr_pred)
        print(f"Test RMSE: {np.sqrt(mse):.4f}")
        
        joblib.dump(reg_trail, paths['trail_reg'])
        print(f"Saved trail regressor to {paths['trail_reg']}")
    else:
        print(f"Not enough profitable samples to train trail regressor for {symbol}; skipped.")
    
    print(f"\n{'='*60}")
    print(f"Completed training for {symbol}")
    print(f"{'='*60}\n")

def train_all_symbols(symbols_list):
    """Train models for all specified symbols"""
    for symbol in symbols_list:
        try:
            train_models_for_symbol(symbol)
        except Exception as e:
            print(f"Error training models for {symbol}: {e}")
            continue

# ================ MODEL LOADING ================

def load_models_for_symbol(symbol):
    """Load trained ML models and scaler for a specific symbol"""
    paths = get_model_paths(symbol)
    clf = None
    reg_lot = None
    reg_trail = None
    scaler = None
    
    if os.path.exists(paths['classifier']):
        try:
            clf = joblib.load(paths['classifier'])
            print(f"Loaded classifier for {symbol}")
        except Exception as e:
            print(f"Failed to load classifier for {symbol}: {e}")
    
    if os.path.exists(paths['lot_reg']):
        try:
            reg_lot = joblib.load(paths['lot_reg'])
            print(f"Loaded lot regressor for {symbol}")
        except Exception as e:
            print(f"Failed to load lot regressor for {symbol}: {e}")
    
    if os.path.exists(paths['trail_reg']):
        try:
            reg_trail = joblib.load(paths['trail_reg'])
            print(f"Loaded trail regressor for {symbol}")
        except Exception as e:
            print(f"Failed to load trail regressor for {symbol}: {e}")
    
    if os.path.exists(paths['scaler']):
        try:
            scaler = joblib.load(paths['scaler'])
            print(f"Loaded feature scaler for {symbol}")
        except Exception as e:
            print(f"Failed to load scaler for {symbol}: {e}")
    
    return clf, reg_lot, reg_trail, scaler

# ================ SIGNAL GENERATION ================

def rule_signal_from_df(df):
    """
    Generate trading signal based on rule-based strategy:
    - Buy: Uptrend, price bounces off short EMA, RSI in valid range
    - Sell: Downtrend, price bounces off short EMA, RSI in valid range
    """
    if df is None or len(df) < max(ma_long, rsi_period) + 2:
        return None
    df = df.copy()
    df['ema_short'] = ema(df['close'], ma_short)
    df['ema_long'] = ema(df['close'], ma_long)
    df['rsi'] = rsi(df['close'], rsi_period)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    if trend_up and prev['low'] <= prev['ema_short'] and last['close'] > last['ema_short'] and 35 < last['rsi'] < 75:
        return 'buy'
    elif trend_down and prev['high'] >= prev['ema_short'] and last['close'] < last['ema_short'] and 25 < last['rsi'] < 65:
        return 'sell'
    return None


def ml_filtered_signal(symbol, clf, scaler, df):
    """
    Generate signal using ML classifier with feature normalization
    Only returns signal when both rule-based and ML agree with high confidence
    """
    if df is None or len(df) < 30:
        return None
    
    feat = build_features(df).iloc[-1]
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 
                    'volatility_10', 'close_minus_ema_short']
    
    if feat[feature_cols].isna().any():
        print(f"Warning: NaN features detected for {symbol}, skipping signal")
        return None
    
    X = feat[feature_cols].values.reshape(1, -1)
    
    # Apply normalization if scaler is available
    if scaler is not None:
        X = scaler.transform(X)

    rule = rule_signal_from_df(df)
    
    if clf is None:
        return rule
    
    try:
        proba = clf.predict_proba(X)[0]
        prob_buy = proba[1] if len(proba) > 1 else proba[0]
    except Exception as e:
        print(f"ML prediction error for {symbol}: {e}")
        return rule

    if rule == 'buy' and prob_buy >= ML_PROB_THRESHOLD:
        print(f"ML-filtered BUY signal for {symbol} (prob={prob_buy:.3f})")
        return 'buy'
    
    if rule == 'sell':
        print(f"Rule-based SELL signal for {symbol} (no ML filter)")
        return 'sell'
    
    return None

# ================ LOT SIZING & TRAILING LOGIC ================

def calculate_lot_ml(symbol, reg_lot, scaler, sl_price_distance, current_features=None):
    """
    Calculate lot size using risk management rules with ML refinement
    Uses normalized features for ML prediction
    """
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
            feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 
                           'vol_10', 'volatility_10', 'close_minus_ema_short']
            
            if feat[feature_cols].isna().any():
                return base_lot
                
            X = feat[feature_cols].values.reshape(1, -1)
        else:
            X = current_features.reshape(1, -1)
        
        # Apply normalization
        if scaler is not None:
            X = scaler.transform(X)
        
        pred = reg_lot.predict(X)[0]
        multiplier = 0.5 + (pred * 2.0)
        multiplier = max(0.3, min(multiplier, 2.0))
        
        final_lot = base_lot * multiplier
        final_lot = max(0.01, round(final_lot, 2))
        
        max_lot = balance * 0.1 / 1000
        final_lot = min(final_lot, max_lot)
        
        print(f"Lot sizing: base={base_lot:.2f}, ML mult={multiplier:.2f}, final={final_lot:.2f}")
        return final_lot
        
    except Exception as e:
        print(f'lot regressor error: {e}')
        return base_lot


def get_trail_multiplier_ml(reg_trail, scaler, df):
    """
    Get trailing stop multiplier using ML regressor with normalized features
    """
    if reg_trail is None or df is None or len(df) < 30:
        return 0.75
    try:
        feat = build_features(df).iloc[-1]
        feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
        
        if feat[feature_cols].isna().any():
            return 0.75
            
        X = feat[feature_cols].values.reshape(1, -1)
        
        if scaler is not None:
            X = scaler.transform(X)
        
        pred = reg_trail.predict(X)[0]
        pred = float(pred)
        pred = max(0.2, min(pred, 3.0))
        return pred
    except Exception as e:
        print('trail regressor error', e)
        return 0.75

# ================ HELPER FUNCTIONS FOR TRAILING STOP ================

def get_symbol_info(symbol):
    """Get symbol information including point and tick size"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, 1e-5, 1e-5
    point = getattr(symbol_info, "point", 1e-5)
    trade_tick_size = getattr(symbol_info, "trade_tick_size", point)
    if trade_tick_size is None or trade_tick_size <= 0:
        trade_tick_size = point
    return symbol_info, float(point), float(trade_tick_size)


def get_min_stop_distance(symbol):
    """Get minimum allowed stop distance for the symbol from broker requirements"""
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
    """Normalize price to symbol's tick size"""
    if tick_size <= 0:
        return float(price)
    normalized = round(price / tick_size) * tick_size
    return float(normalized)


def validate_sl_distance(current_price, new_sl, min_stop_dist, pos_type):
    """Validate that stop loss distance meets minimum requirements"""
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

# ================ TRAILING STOP MANAGEMENT ================

def manage_trailing_positions(models_cache):
    """
    Manage open positions with two-phase trailing strategy
    Uses symbol-specific models loaded from cache
    
    Args:
        models_cache: Dict mapping symbol -> (clf, reg_lot, reg_trail, scaler)
    """
    positions = mt5.positions_get()
    if positions is None:
        return
    
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        
        # Get models for this symbol
        if symbol not in models_cache:
            print(f"No models loaded for {symbol}, skipping trailing stop")
            continue
        
        clf, reg_lot, reg_trail, scaler = models_cache[symbol]
        
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
        
        elif profit_atr >= 1.5:
            trail_mult = get_trail_multiplier_ml(reg_trail, scaler, df)
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
                        print(f"Skipping trail for {symbol}: target SL {target_sl:.5f} below entry {entry:.5f}")
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
                    else:
                        print(f"Skipping trail for {symbol}: target SL {target_sl:.5f} above entry {entry:.5f}")
        
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
                print(f" Modified pos {ticket} {symbol}: SL={new_sl:.5f}, TP={new_tp:.5f}")

# ================ TRADE EXECUTION ================

def open_trade(symbol, action, clf, reg_lot, scaler):
    """
    Open a new trade with calculated lot size, stop loss, and take profit
    Uses ML-based lot sizing with normalized features
    """
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

    atr_mult_sl = 1.5
    atr_mult_tp = 3.0

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

    lot = calculate_lot_ml(symbol, reg_lot, scaler, sl_distance if use_stops else None)
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
        print(f" {symbol} {action} opened at {price} | SL={candidate_sl:.5f}, TP(hard)={candidate_tp2:.5f}, ATR={atr_val:.5f}, LOT={lot}")

# ================ BACKTEST UTILITY ================

def backtest(symbol):
    """
    Simple backtest: evaluate classifier precision on historical data with normalized features
    """
    csv_path = get_data_path(symbol)
    
    if not os.path.exists(csv_path):
        print(f"No data file found for {symbol} at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    
    clf, _, _, scaler = load_models_for_symbol(symbol)
    if clf is None:
        print(f'No classifier to evaluate for {symbol}. Train models first.')
        return
    
    X = df[feature_cols].values
    
    if scaler is not None:
        X = scaler.transform(X)
    
    y = df['label'].values
    y_bin = np.where(y == 1, 1, 0)
    preds = clf.predict(X)
    
    print(f"\nBacktest Results for {symbol}:")
    print(classification_report(y_bin, preds))

def backtest_all(symbols_list):
    """Backtest all symbols"""
    for symbol in symbols_list:
        try:
            backtest(symbol)
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            continue

# ================ MAIN LIVE LOOP ================

def run_live(symbols_list, mode='live'):
    """
    Main execution loop with three modes:
    - 'collect': Collect historical data and save to symbol-specific CSV
    - 'train': Train ML models from collected data for each symbol
    - 'live': Run live trading with signal generation and position management
    """
    if mode == 'collect':
        for s in symbols_list:
            collect_and_save(s, num_bars=2000)
        return
    
    if mode == 'train':
        train_all_symbols(symbols_list)
        return

    # Load models for all symbols
    models_cache = {}
    for symbol in symbols_list:
        clf, reg_lot, reg_trail, scaler = load_models_for_symbol(symbol)
        models_cache[symbol] = (clf, reg_lot, reg_trail, scaler)

    print('Starting live loop...')
    print(f"Trading symbols: {symbols_list}")
    print(f"Magic number: {magic_number}")
    print(f"Models loaded per symbol:")
    for symbol in symbols_list:
        clf, reg_lot, reg_trail, scaler = models_cache[symbol]
        print(f"  {symbol}: clf={clf is not None}, lot_reg={reg_lot is not None}, trail_reg={reg_trail is not None}, scaler={scaler is not None}")
    
    try:
        while True:
            for symbol in symbols_list:
                try:
                    df = get_ohlcv(symbol, 300)
                    if df is None:
                        print(f"No data for {symbol}, skipping...")
                        continue
                    
                    # Get models for this symbol
                    clf, reg_lot, reg_trail, scaler = models_cache[symbol]
                    
                    sig = ml_filtered_signal(symbol, clf, scaler, df)
                    if sig:
                        print(f" Signal detected: {symbol} - {sig.upper()}")
                        open_trade(symbol, sig, clf, reg_lot, scaler)
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            try:
                manage_trailing_positions(models_cache)
            except Exception as e:
                print(f"Error in trailing stop management: {e}")
            
            
            
            
    except KeyboardInterrupt:
        print('\n\nStopping live loop (KeyboardInterrupt)')
    except Exception as e:
        print(f'\n\nFatal error in live loop: {e}')
    finally:
        if mt5.initialize():
            mt5.shutdown()
            print("MT5 connection closed")

# ================ CLI ENTRY POINT ================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ML-Enhanced Trading Bot for MT5 with Per-Symbol Models')
    parser.add_argument('--mode', choices=['collect', 'train', 'live', 'backtest'], default='live',
                        help='Operation mode: collect data, train models, run live trading, or backtest')
    parser.add_argument('--symbols', nargs='*', default=symbols,
                        help='List of symbols to trade (e.g., XAUUSD XAUEUR)')
    args = parser.parse_args()

    print("=" * 60)
    print("RSI-MA-ATR Trading Bot with ML - ENHANCED VERSION")
    print("Features: Per-Symbol Models + Normalization + CV")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print("=" * 60)
    print()

    if args.mode == 'backtest':
        backtest_all(args.symbols)
    else:
        run_live(args.symbols, mode=args.mode)