"""
RSI-MA-ATR Trading Bot - RULE-BASED VERSION
- Pure rule-based trading without ML confirmation
- Strict price conditions without bounce tolerance
- Reduced lot sizes for safer trading
- Full diagnostic logging

USAGE:
1. Run with --mode collect to gather data (optional)
2. Run with --mode train to train models (optional)
3. Run with --mode live for live trading

Dependencies: pandas, numpy, scikit-learn, joblib, MetaTrader5
"""

import os
import time
from datetime import datetime
import math
import logging

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib

# ================ LOGGING SETUP ================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'bot_diagnostic_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================ CONFIGURATION ================
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
symbols = ["XAUUSD", "XAUEUR"]
magic_number = 234000
comment_text = "RSI-MA-ATR-Bot"

rsi_period = 14
ma_short = 20
ma_long = 50
atr_period = 14

default_lot = 0.01
risk_per_trade = 0.01  # Reduced from 0.02 to 0.01

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CV_FOLDS = 5

# ================ MT5 INITIALIZATION ================
if not mt5.initialize(path=MT5_PATH):
    logger.error(f"MT5 init failed: {mt5.last_error()}")

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
    """Label historical data based on future price movement"""
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
    """Collect historical data, build features, create labels, and save to CSV"""
    out_csv = get_data_path(symbol)
    logger.info(f"Collecting {symbol}...")
    df = get_ohlcv(symbol, num_bars)
    if df is None:
        logger.warning(f"No data for {symbol}")
        return
    feat = build_features(df)
    labeled = label_future(feat, lookahead=lookahead)
    keep_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'ema_short', 'ema_long', 'rsi', 'atr',
                 'ema_diff', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short', 'close_minus_ema_long', 'label']
    out = labeled[keep_cols].dropna()
    out.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(out)} rows to {out_csv}")

# ================ MODEL TRAINING ================

def train_models_for_symbol(symbol, test_size=0.2, random_state=42):
    """Train ML models for a specific symbol with normalization and CV"""
    csv_path = get_data_path(symbol)
    
    if not os.path.exists(csv_path):
        logger.error(f"No data file found for {symbol} at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training models for {symbol}")
    logger.info(f"{'='*60}")
    logger.info(f"Training on {len(df)} samples")
    
    X = df[feature_cols].values
    y = df['label'].values

    paths = get_model_paths(symbol)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, paths['scaler'])
    logger.info(f"Saved scaler to {paths['scaler']}")

    y_bin = np.where(y == 1, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
    
    logger.info("Performing cross-validation for classifier...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    logger.info("\nClassifier report (buy vs not-buy):")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("\nFeature Importance (Classifier):")
    logger.info("\n" + feature_importance.to_string(index=False))
    
    joblib.dump(clf, paths['classifier'])
    logger.info(f"\nSaved classifier to {paths['classifier']}")

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
        
        logger.info("\nPerforming cross-validation for lot regressor...")
        cv_scores_lot = cross_val_score(reg_lot, Xl_train, yl_train, cv=CV_FOLDS, scoring='neg_mean_squared_error')
        logger.info(f"CV RMSE: {np.sqrt(-cv_scores_lot.mean()):.4f} (+/- {np.sqrt(cv_scores_lot.std()):.4f})")
        
        reg_lot.fit(Xl_train, yl_train)
        yl_pred = reg_lot.predict(Xl_test)
        mse = mean_squared_error(yl_test, yl_pred)
        logger.info(f"Test RMSE: {np.sqrt(mse):.4f}")
        
        joblib.dump(reg_lot, paths['lot_reg'])
        logger.info(f"Saved lot regressor to {paths['lot_reg']}")
    else:
        logger.warning(f"Not enough profitable samples to train lot regressor for {symbol}")

    if len(df_profitable) >= 50:
        target_trail = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_trail = (target_trail / target_trail.max())
        target_trail = 0.5 + target_trail * 1.5
        
        Xr = df_profitable[feature_cols].values
        Xr_scaled = scaler.transform(Xr)
        yr = target_trail.values
        
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled, yr, test_size=test_size, random_state=random_state)
        
        reg_trail = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        
        logger.info("\nPerforming cross-validation for trail regressor...")
        cv_scores_trail = cross_val_score(reg_trail, Xr_train, yr_train, cv=CV_FOLDS, scoring='neg_mean_squared_error')
        logger.info(f"CV RMSE: {np.sqrt(-cv_scores_trail.mean()):.4f} (+/- {np.sqrt(cv_scores_trail.std()):.4f})")
        
        reg_trail.fit(Xr_train, yr_train)
        yr_pred = reg_trail.predict(Xr_test)
        mse = mean_squared_error(yr_test, yr_pred)
        logger.info(f"Test RMSE: {np.sqrt(mse):.4f}")
        
        joblib.dump(reg_trail, paths['trail_reg'])
        logger.info(f"Saved trail regressor to {paths['trail_reg']}")
    else:
        logger.warning(f"Not enough profitable samples to train trail regressor for {symbol}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed training for {symbol}")
    logger.info(f"{'='*60}\n")

def train_all_symbols(symbols_list):
    """Train models for all specified symbols"""
    for symbol in symbols_list:
        try:
            train_models_for_symbol(symbol)
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}", exc_info=True)
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
            logger.info(f"Loaded classifier for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load classifier for {symbol}: {e}")
    
    if os.path.exists(paths['lot_reg']):
        try:
            reg_lot = joblib.load(paths['lot_reg'])
            logger.info(f"Loaded lot regressor for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load lot regressor for {symbol}: {e}")
    
    if os.path.exists(paths['trail_reg']):
        try:
            reg_trail = joblib.load(paths['trail_reg'])
            logger.info(f"Loaded trail regressor for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load trail regressor for {symbol}: {e}")
    
    if os.path.exists(paths['scaler']):
        try:
            scaler = joblib.load(paths['scaler'])
            logger.info(f"Loaded feature scaler for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load scaler for {symbol}: {e}")
    
    return clf, reg_lot, reg_trail, scaler

# ================ DIAGNOSTIC FUNCTIONS ================

def diagnose_no_signal(symbol, df):
    """Detailed diagnostics for why no signal was generated"""
    if df is None or len(df) < max(ma_long, rsi_period) + 2:
        logger.warning(f"{symbol}: Insufficient data - need {max(ma_long, rsi_period) + 2} bars, have {len(df) if df is not None else 0}")
        return
    
    df_check = df.copy()
    df_check['ema_short'] = ema(df_check['close'], ma_short)
    df_check['ema_long'] = ema(df_check['close'], ma_long)
    df_check['rsi'] = rsi(df_check['close'], rsi_period)
    
    last = df_check.iloc[-1]
    prev = df_check.iloc[-2]
    
    atr_series = calc_atr(df_check, atr_period)
    atr_val = atr_series.iloc[-1] if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DIAGNOSTIC REPORT: {symbol} @ {last['time']}")
    logger.info(f"{'='*60}")
    logger.info(f"Price: {last['close']:.2f}")
    logger.info(f"RSI: {last['rsi']:.2f}")
    logger.info(f"EMA Short ({ma_short}): {last['ema_short']:.2f}")
    logger.info(f"EMA Long ({ma_long}): {last['ema_long']:.2f}")
    logger.info(f"EMA Diff: {last['ema_short'] - last['ema_long']:.2f}")
    logger.info(f"ATR: {atr_val:.2f}")
    
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    logger.info(f"Trend: {'UP' if trend_up else 'DOWN' if trend_down else 'NEUTRAL'}")
    
    buy_bounce = prev['low'] <= prev['ema_short']
    distance_to_ema_buy = prev['low'] - prev['ema_short']
    
    logger.info(f"\nBUY Signal Checklist:")
    logger.info(f"  [{'PASS' if trend_up else 'FAIL'}] Trend Up: {trend_up}")
    logger.info(f"  [{'PASS' if buy_bounce else 'FAIL'}] Prev low <= EMA_short: {prev['low']:.2f} vs {prev['ema_short']:.2f} (diff={distance_to_ema_buy:.2f})")
    logger.info(f"  [{'PASS' if last['close'] > last['ema_short'] else 'FAIL'}] Close > EMA_short: {last['close']:.2f} > {last['ema_short']:.2f} = {last['close'] > last['ema_short']}")
    logger.info(f"  [{'PASS' if 35 < last['rsi'] < 75 else 'FAIL'}] RSI in range (35-75): {35 < last['rsi'] < 75} (RSI={last['rsi']:.2f})")
    
    sell_bounce = prev['high'] >= prev['ema_short']
    distance_to_ema_sell = prev['ema_short'] - prev['high']
    
    logger.info(f"\nSELL Signal Checklist:")
    logger.info(f"  [{'PASS' if trend_down else 'FAIL'}] Trend Down: {trend_down}")
    logger.info(f"  [{'PASS' if sell_bounce else 'FAIL'}] Prev high >= EMA_short: {prev['high']:.2f} vs {prev['ema_short']:.2f} (diff={distance_to_ema_sell:.2f})")
    logger.info(f"  [{'PASS' if last['close'] < last['ema_short'] else 'FAIL'}] Close < EMA_short: {last['close']:.2f} < {last['ema_short']:.2f} = {last['close'] < last['ema_short']}")
    logger.info(f"  [{'PASS' if 25 < last['rsi'] < 65 else 'FAIL'}] RSI in range (25-65): {25 < last['rsi'] < 65} (RSI={last['rsi']:.2f})")
    logger.info(f"{'='*60}\n")


def check_existing_positions(symbol):
    """Check if we already have open positions for this symbol"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    
    our_positions = [p for p in positions if p.magic == magic_number]
    
    if our_positions:
        logger.info(f"{symbol}: Already have {len(our_positions)} open position(s)")
        for pos in our_positions:
            pos_type_str = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            logger.info(f"  - Ticket {pos.ticket}: {pos_type_str} @ {pos.price_open}, profit: {pos.profit:.2f}")
    
    return our_positions


def diagnose_market_activity(symbol, df):
    """Check if market is moving enough to generate signals"""
    if df is None or len(df) < 50:
        return
    
    recent_df = df.tail(50)
    
    price_change = recent_df['close'].pct_change()
    avg_volatility = price_change.std() * 100
    
    recent_high = recent_df['high'].max()
    recent_low = recent_df['low'].min()
    price_range = recent_high - recent_low
    price_range_pct = (price_range / recent_df['close'].iloc[-1]) * 100
    
    avg_volume = recent_df['tick_volume'].mean()
    recent_volume = recent_df['tick_volume'].tail(10).mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MARKET ACTIVITY REPORT: {symbol}")
    logger.info(f"{'='*60}")
    logger.info(f"Average Volatility (last 50 bars): {avg_volatility:.4f}%")
    logger.info(f"Price Range (last 50 bars): {price_range:.2f} ({price_range_pct:.2f}%)")
    logger.info(f"Volume Ratio (recent/average): {volume_ratio:.2f}x")
    
    if avg_volatility < 0.01:
        logger.warning(f"[WARNING] LOW VOLATILITY - Market is moving very slowly")
    if volume_ratio < 0.5:
        logger.warning(f"[WARNING] LOW VOLUME - Trading activity is low")
    
    logger.info(f"{'='*60}\n")

# ================ SIGNAL GENERATION ================

def rule_signal_from_df(df, symbol):
    """Generate rule-based trading signal with diagnostics"""
    if df is None or len(df) < max(ma_long, rsi_period) + 2:
        logger.warning(f"{symbol}: Insufficient data for signal")
        return None
    
    df = df.copy()
    df['ema_short'] = ema(df['close'], ma_short)
    df['ema_long'] = ema(df['close'], ma_long)
    df['rsi'] = rsi(df['close'], rsi_period)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    
    buy_bounce = prev['low'] <= prev['ema_short']
    
    if trend_up and buy_bounce and last['close'] > last['ema_short'] and 35 < last['rsi'] < 75:
        logger.info(f"[SIGNAL] {symbol}: RULE-BASED BUY SIGNAL GENERATED")
        return 'buy'
    
    sell_bounce = prev['high'] >= prev['ema_short']
    
    if trend_down and sell_bounce and last['close'] < last['ema_short'] and 25 < last['rsi'] < 65:
        logger.info(f"[SIGNAL] {symbol}: RULE-BASED SELL SIGNAL GENERATED")
        return 'sell'
    
    return None

# ================ LOT SIZING & TRAILING LOGIC ================

def calculate_lot_ml(symbol, reg_lot, scaler, sl_price_distance, current_features=None):
    """Calculate lot size using risk management rules - REDUCED SIZE"""
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
        
        # Apply conservative multiplier (reduce size)
        base_lot = base_lot * 0.5  # Reduce to 50% of calculated size
        base_lot = max(0.01, round(base_lot, 2))
        
    except Exception as e:
        logger.error(f'calculate_lot base error: {e}')
        return default_lot

    return base_lot


def get_trail_multiplier_ml(reg_trail, scaler, df):
    """Get trailing stop multiplier using ML regressor"""
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
        logger.error(f'trail regressor error: {e}')
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
    """Get minimum allowed stop distance for the symbol"""
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
    """Manage open positions with two-phase trailing strategy"""
    positions = mt5.positions_get()
    if positions is None:
        return
    
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        
        if symbol not in models_cache:
            logger.warning(f"No models loaded for {symbol}, skipping trailing stop")
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
                            logger.info(f"Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
            else:
                lock_sl = entry - (0.2 * atr_val)
                lock_sl = normalize_price(lock_sl, tick_size)
                
                if lock_sl > current_price + min_stop_dist:
                    if current_sl is None or lock_sl < current_sl:
                        if validate_sl_distance(current_price, lock_sl, min_stop_dist, pos_type):
                            new_sl = lock_sl
                            need_modify = True
                            logger.info(f"Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
        
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
                                logger.info(f"Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} (mult={trail_mult:.2f}, profit={profit_atr:.2f} ATR)")
                    else:
                        logger.debug(f"Skipping trail for {symbol}: target SL {target_sl:.5f} below entry {entry:.5f}")
            else:
                target_sl = current_price + trail_dist
                target_sl = normalize_price(target_sl, tick_size)
                
                if target_sl > current_price + min_stop_dist:
                    if target_sl < entry:
                        if current_sl is None or target_sl < current_sl:
                            if validate_sl_distance(current_price, target_sl, min_stop_dist, pos_type):
                                new_sl = target_sl
                                need_modify = True
                                logger.info(f"Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} (mult={trail_mult:.2f}, profit={profit_atr:.2f} ATR)")
                    else:
                        logger.debug(f"Skipping trail for {symbol}: target SL {target_sl:.5f} above entry {entry:.5f}")
        
        if current_tp is None or current_tp == 0:
            tp2 = entry + (3.0 * atr_val) if pos_type == mt5.ORDER_TYPE_BUY else entry - (3.0 * atr_val)
            new_tp = normalize_price(tp2, tick_size)
            need_modify = True
            logger.info(f"Setting hard TP for {symbol} ticket {ticket} at {new_tp:.5f}")
        
        if need_modify and (new_sl != current_sl or new_tp != current_tp):
            if new_sl is not None:
                if not validate_sl_distance(current_price, new_sl, min_stop_dist, pos_type):
                    logger.warning(f"Skipping invalid SL for {symbol} ticket {ticket}: SL={new_sl:.5f} too close to price={current_price:.5f}")
                    continue
                
                if profit_atr >= 1.5:
                    if pos_type == mt5.ORDER_TYPE_BUY and new_sl <= entry:
                        logger.warning(f"Skipping SL below entry in trailing phase for {symbol}: SL={new_sl:.5f}, entry={entry:.5f}")
                        continue
                    if pos_type == mt5.ORDER_TYPE_SELL and new_sl >= entry:
                        logger.warning(f"Skipping SL above entry in trailing phase for {symbol}: SL={new_sl:.5f}, entry={entry:.5f}")
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
                logger.error(f"Modify failed for {symbol} ticket {ticket}: None response")
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Modify failed for {symbol} ticket {ticket}: retcode={result.retcode}, comment={result.comment}")
            else:
                logger.info(f"[SUCCESS] Modified pos {ticket} {symbol}: SL={new_sl:.5f}, TP={new_tp:.5f}")

# ================ TRADE EXECUTION ================

def open_trade(symbol, action, clf, reg_lot, scaler):
    """Open a new trade with calculated lot size, stop loss, and take profit"""
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} not available / not selected")
        return
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"No tick for {symbol}")
        return
    
    symbol_info, point, tick_size = get_symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return
    
    price = tick.ask if action == "buy" else tick.bid
    
    df = get_ohlcv(symbol, 500)
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
            logger.warning(f"Warning: Cannot set stops for {symbol} - minimum distance not achievable")

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
        logger.error(f"{symbol} {action} failed: retcode={getattr(result, 'retcode', 'N/A')}, comment={getattr(result, 'comment', 'N/A')}")
    else:
        logger.info(f"[TRADE OPENED] {symbol} {action} at {price} | SL={candidate_sl:.5f}, TP={candidate_tp2:.5f}, ATR={atr_val:.5f}, LOT={lot}")

# ================ BACKTEST UTILITY ================

def backtest(symbol):
    """Simple backtest: evaluate classifier precision on historical data"""
    csv_path = get_data_path(symbol)
    
    if not os.path.exists(csv_path):
        logger.error(f"No data file found for {symbol} at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    
    clf, _, _, scaler = load_models_for_symbol(symbol)
    if clf is None:
        logger.error(f'No classifier to evaluate for {symbol}. Train models first.')
        return
    
    X = df[feature_cols].values
    
    if scaler is not None:
        X = scaler.transform(X)
    
    y = df['label'].values
    y_bin = np.where(y == 1, 1, 0)
    preds = clf.predict(X)
    
    logger.info(f"\nBacktest Results for {symbol}:")
    logger.info("\n" + classification_report(y_bin, preds))

def backtest_all(symbols_list):
    """Backtest all symbols"""
    for symbol in symbols_list:
        try:
            backtest(symbol)
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
            continue

# ================ MAIN LIVE LOOP ================

def run_live(symbols_list, mode='live'):
    """Main execution loop with rule-based trading only"""
    if mode == 'collect':
        for s in symbols_list:
            collect_and_save(s, num_bars=300)
        return
    
    if mode == 'train':
        train_all_symbols(symbols_list)
        return

    models_cache = {}
    for symbol in symbols_list:
        clf, reg_lot, reg_trail, scaler = load_models_for_symbol(symbol)
        models_cache[symbol] = (clf, reg_lot, reg_trail, scaler)

    logger.info('='*60)
    logger.info('STARTING RULE-BASED LIVE TRADING')
    logger.info('='*60)
    logger.info(f"Trading symbols: {symbols_list}")
    logger.info(f"Magic number: {magic_number}")
    logger.info(f"Risk per trade: {risk_per_trade * 100}%")
    logger.info(f"Default lot size: {default_lot}")
    logger.info('='*60)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            logger.info(f"\n{'#'*60}")
            logger.info(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'#'*60}")
            
            for symbol in symbols_list:
                try:
                    logger.info(f"\n--- Processing {symbol} ---")
                    
                    existing_positions = check_existing_positions(symbol)
                    
                    df = get_ohlcv(symbol, 1000)
                    if df is None:
                        logger.error(f"{symbol}: Failed to fetch OHLCV data")
                        continue
                    
                    logger.info(f"{symbol}: Fetched {len(df)} bars, latest time: {df['time'].iloc[-1]}")
                    
                    if iteration % 10 == 1:
                        diagnose_market_activity(symbol, df)
                    
                    clf, reg_lot, reg_trail, scaler = models_cache[symbol]
                    
                    sig = rule_signal_from_df(df, symbol)
                    
                    if sig is None:
                        logger.info(f"{symbol}: No signal generated")
                        if iteration % 5 == 1:
                            diagnose_no_signal(symbol, df)
                    else:
                        logger.info(f"[SIGNAL DETECTED] {symbol}: {sig.upper()}")
                        logger.info(f"{symbol}: Attempting to open {sig.upper()} trade...")
                        open_trade(symbol, sig, clf, reg_lot, scaler)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                    continue
            
            try:
                logger.info(f"\n--- Managing Trailing Stops ---")
                manage_trailing_positions(models_cache)
            except Exception as e:
                logger.error(f"Error in trailing stop management: {e}", exc_info=True)
            
            sleep_seconds = 120
            logger.info(f"\nSleeping for {sleep_seconds} seconds...\n")
            time.sleep(sleep_seconds)
            
    except KeyboardInterrupt:
        logger.info('\n\nStopping live loop (KeyboardInterrupt)')
    except Exception as e:
        logger.error(f'\n\nFatal error in live loop: {e}', exc_info=True)
    finally:
        if mt5.initialize():
            mt5.shutdown()
            logger.info("MT5 connection closed")

# ================ CLI ENTRY POINT ================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rule-Based Trading Bot')
    parser.add_argument('--mode', choices=['collect', 'train', 'live', 'backtest'], default='live',
                        help='Operation mode: collect data, train models, run live trading, or backtest')
    parser.add_argument('--symbols', nargs='*', default=symbols,
                        help='List of symbols to trade (e.g., XAUUSD XAUEUR)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RSI-MA-ATR Trading Bot - RULE-BASED VERSION")
    logger.info("Pure rule-based trading without ML confirmation")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info("=" * 60)

    if args.mode == 'backtest':
        backtest_all(args.symbols)
    else:
        run_live(args.symbols, mode=args.mode)