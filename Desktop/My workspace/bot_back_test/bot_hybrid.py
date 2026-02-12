"""
RSI-MA-ATR Trading Bot with ML integration for MT5 - HYBRID STOP-LOSS VERSION
- Full diagnostic logging to identify why trades aren't executing
- Per-symbol model support with normalization and cross-validation
- Enhanced HYBRID STOP-LOSS SYSTEM:
  * Initial ATR-based fixed stop loss
  * Break-even stop after 1.0 ATR profit
  * Two-phase trailing (profit lock + dynamic trailing)
  * Time-based exit monitoring
  * Volatility-adjusted stop distance

USAGE:
1. Run with --mode collect to gather data
2. Run with --mode train to train models
3. Run with --mode live for live trading with diagnostics
4. Check bot_diagnostic_YYYYMMDD.log for detailed analysis

Dependencies: pandas, numpy, scikit-learn, joblib, MetaTrader5
"""

import os
import time
from datetime import datetime, timedelta
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
comment_text = "RSI-MA-ATR-ML-Bot"

rsi_period = 14
ma_short = 20
ma_long = 50
atr_period = 14

default_lot = 0.01
risk_per_trade = 0.02

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ML_PROB_THRESHOLD = 0.65
CV_FOLDS = 5

# ================ POSITION LIMITS ================
MAX_TOTAL_POSITIONS = 6  # Maximum total open positions across all symbols
MAX_POSITIONS_PER_SYMBOL = 3  # Maximum positions per individual symbol

# ================ HYBRID STOP-LOSS CONFIGURATION ================
BREAKEVEN_PROFIT_ATR = 1.0  # Move to breakeven after 1.0 ATR profit
BREAKEVEN_OFFSET_ATR = 0.2  # Lock in 0.2 ATR profit at breakeven
TIME_BASED_EXIT_HOURS = 24  # Exit if no significant progress after 24 hours
TIME_EXIT_MIN_PROFIT_ATR = 0.5  # Minimum profit to avoid time-based exit
VOLATILITY_SPIKE_THRESHOLD = 2.0  # Tighten stops if volatility spikes 2x

# Dictionary to track position metadata for hybrid stop-loss
position_metadata = {}

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


def get_ohlcv(symbol, num_bars=500, timeframe=mt5.TIMEFRAME_H1):
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
    bounce_tolerance = 0.5 * atr_val if atr_val > 0 else 20
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DIAGNOSTIC REPORT: {symbol} @ {last['time']}")
    logger.info(f"{'='*60}")
    logger.info(f"Price: {last['close']:.2f}")
    logger.info(f"RSI: {last['rsi']:.2f}")
    logger.info(f"EMA Short ({ma_short}): {last['ema_short']:.2f}")
    logger.info(f"EMA Long ({ma_long}): {last['ema_long']:.2f}")
    logger.info(f"EMA Diff: {last['ema_short'] - last['ema_long']:.2f}")
    logger.info(f"ATR: {atr_val:.2f}")
    logger.info(f"Bounce Tolerance: {bounce_tolerance:.2f}")
    
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    logger.info(f"Trend: {'UP' if trend_up else 'DOWN' if trend_down else 'NEUTRAL'}")
    
    buy_bounce = prev['low'] <= (prev['ema_short'] + bounce_tolerance)
    distance_to_ema_buy = prev['low'] - prev['ema_short']
    
    logger.info(f"\nBUY Signal Checklist:")
    logger.info(f"  [{'PASS' if trend_up else 'FAIL'}] Trend Up: {trend_up}")
    logger.info(f"  [{'PASS' if buy_bounce else 'FAIL'}] Prev low near/below EMA_short (tolerance={bounce_tolerance:.2f}): {prev['low']:.2f} vs {prev['ema_short']:.2f} (diff={distance_to_ema_buy:.2f})")
    logger.info(f"  [{'PASS' if last['close'] > last['ema_short'] else 'FAIL'}] Close > EMA_short: {last['close']:.2f} > {last['ema_short']:.2f} = {last['close'] > last['ema_short']}")
    logger.info(f"  [{'PASS' if 35 < last['rsi'] < 75 else 'FAIL'}] RSI in range (35-75): {35 < last['rsi'] < 75} (RSI={last['rsi']:.2f})")
    
    sell_bounce = prev['high'] >= (prev['ema_short'] - bounce_tolerance)
    distance_to_ema_sell = prev['ema_short'] - prev['high']
    
    logger.info(f"\nSELL Signal Checklist:")
    logger.info(f"  [{'PASS' if trend_down else 'FAIL'}] Trend Down: {trend_down}")
    logger.info(f"  [{'PASS' if sell_bounce else 'FAIL'}] Prev high near/above EMA_short (tolerance={bounce_tolerance:.2f}): {prev['high']:.2f} vs {prev['ema_short']:.2f} (diff={distance_to_ema_sell:.2f})")
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


def get_total_positions():
    """Get total number of open positions for this bot across all symbols"""
    positions = mt5.positions_get()
    if positions is None:
        return 0
    
    our_positions = [p for p in positions if p.magic == magic_number]
    return len(our_positions)


def check_position_limits(symbol):
    """Check if we can open a new position based on limits
    
    Returns:
        tuple: (can_open: bool, reason: str)
    """
    # Check total positions limit
    total_positions = get_total_positions()
    if total_positions >= MAX_TOTAL_POSITIONS:
        reason = f"Total position limit reached ({total_positions}/{MAX_TOTAL_POSITIONS})"
        logger.warning(f"[POSITION LIMIT] {reason}")
        return False, reason
    
    # Check per-symbol limit
    symbol_positions = check_existing_positions(symbol)
    if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
        reason = f"Symbol position limit reached for {symbol} ({len(symbol_positions)}/{MAX_POSITIONS_PER_SYMBOL})"
        logger.warning(f"[POSITION LIMIT] {reason}")
        return False, reason
    
    return True, f"OK - Total: {total_positions}/{MAX_TOTAL_POSITIONS}, {symbol}: {len(symbol_positions)}/{MAX_POSITIONS_PER_SYMBOL}"


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
    
    atr_series = calc_atr(df, atr_period)
    atr_val = atr_series.iloc[-1] if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else 0
    bounce_tolerance = 0.5 * atr_val if atr_val > 0 else 20
    
    buy_bounce = prev['low'] <= (prev['ema_short'] + bounce_tolerance)
    
    if trend_up and buy_bounce and last['close'] > last['ema_short'] and 35 < last['rsi'] < 75:
        logger.info(f"[SIGNAL] {symbol}: RULE-BASED BUY SIGNAL GENERATED!")
        return 'buy'
    
    sell_bounce = prev['high'] >= (prev['ema_short'] - bounce_tolerance)
    
    if trend_down and sell_bounce and last['close'] < last['ema_short'] and 25 < last['rsi'] < 65:
        logger.info(f"[SIGNAL] {symbol}: RULE-BASED SELL SIGNAL GENERATED!")
        return 'sell'
    
    return None


def ml_filtered_signal(symbol, clf, scaler, df):
    """Generate signal using ML classifier with feature normalization"""
    if df is None or len(df) < 30:
        logger.warning(f"{symbol}: Insufficient data for ML signal")
        return None
    
    feat = build_features(df).iloc[-1]
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 
                    'volatility_10', 'close_minus_ema_short']
    
    nan_features = feat[feature_cols].isna()
    if nan_features.any():
        logger.warning(f"{symbol}: NaN features detected: {nan_features[nan_features].index.tolist()}")
        return None
    
    X = feat[feature_cols].values.reshape(1, -1)
    
    if scaler is not None:
        X = scaler.transform(X)
    
    rule = rule_signal_from_df(df, symbol)
    
    if clf is None:
        if rule:
            logger.info(f"{symbol}: No ML model - using rule-based signal: {rule}")
        return rule
    
    try:
        proba = clf.predict_proba(X)[0]
        prob_buy = proba[1] if len(proba) > 1 else proba[0]
        prob_not_buy = proba[0] if len(proba) > 1 else 1 - proba[0]
        
        logger.info(f"{symbol}: ML Probabilities - BUY: {prob_buy:.3f}, NOT-BUY: {prob_not_buy:.3f}")
        
    except Exception as e:
        logger.error(f"{symbol}: ML prediction error: {e}")
        return rule
    
    if rule == 'buy':
        return 'buy'
    
    if rule == 'sell':
        logger.info(f"[APPROVED] {symbol}: SELL SIGNAL APPROVED (no ML filter for sell)")
        return 'sell'
    
    return None

# ================ LOT SIZING & TRAILING LOGIC ================

def calculate_lot_ml(symbol, reg_lot, scaler, sl_price_distance, current_features=None):
    """Calculate lot size using risk management rules with ML refinement"""
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
        logger.error(f'calculate_lot base error: {e}')
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
        
        if scaler is not None:
            X = scaler.transform(X)
        
        pred = reg_lot.predict(X)[0]
        multiplier = 0.5 + (pred * 2.0)
        multiplier = max(0.3, min(multiplier, 2.0))
        
        final_lot = base_lot * multiplier
        final_lot = max(0.01, round(final_lot, 2))
        
        max_lot = balance * 0.1 / 1000
        final_lot = min(final_lot, max_lot)
        
        logger.info(f"Lot sizing: base={base_lot:.2f}, ML mult={multiplier:.2f}, final={final_lot:.2f}")
        return final_lot
        
    except Exception as e:
        logger.error(f'lot regressor error: {e}')
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

# ================ HYBRID STOP-LOSS SYSTEM ================

def initialize_position_metadata(ticket, symbol, entry_price, entry_time, initial_atr):
    """Initialize metadata for tracking hybrid stop-loss components"""
    position_metadata[ticket] = {
        'symbol': symbol,
        'entry_price': entry_price,
        'entry_time': entry_time,
        'initial_atr': initial_atr,
        'breakeven_applied': False,
        'trailing_active': False,
        'highest_profit_atr': 0.0,
        'entry_volatility': None,
        'max_favorable_price': entry_price
    }
    logger.info(f"[HYBRID SL] Initialized metadata for ticket {ticket}")


def check_time_based_exit(pos, metadata):
    """Check if position should be exited based on time and lack of progress"""
    if metadata is None:
        return False
    
    entry_time = metadata.get('entry_time')
    if entry_time is None:
        return False
    
    time_in_trade = datetime.now() - entry_time
    hours_in_trade = time_in_trade.total_seconds() / 3600
    
    if hours_in_trade < TIME_BASED_EXIT_HOURS:
        return False
    
    current_profit_atr = metadata.get('highest_profit_atr', 0.0)
    
    if current_profit_atr < TIME_EXIT_MIN_PROFIT_ATR:
        logger.warning(f"[HYBRID SL] Time-based exit triggered for ticket {pos.ticket}: "
                      f"{hours_in_trade:.1f}h in trade with only {current_profit_atr:.2f} ATR profit")
        return True
    
    return False


def detect_volatility_spike(df, initial_volatility):
    """Detect if current volatility has spiked significantly"""
    if df is None or len(df) < 20 or initial_volatility is None:
        return False, 1.0
    
    recent_volatility = df['close'].pct_change().tail(10).std()
    
    if initial_volatility <= 0:
        return False, 1.0
    
    volatility_ratio = recent_volatility / initial_volatility
    
    if volatility_ratio >= VOLATILITY_SPIKE_THRESHOLD:
        return True, volatility_ratio
    
    return False, volatility_ratio


def apply_breakeven_stop(pos, current_price, atr_val, metadata, min_stop_dist, tick_size):
    """Apply break-even stop after reaching minimum profit threshold"""
    if metadata.get('breakeven_applied', False):
        return None
    
    entry = pos.price_open
    pos_type = pos.type
    
    if pos_type == mt5.ORDER_TYPE_BUY:
        profit_atr = (current_price - entry) / atr_val
    else:
        profit_atr = (entry - current_price) / atr_val
    
    if profit_atr >= BREAKEVEN_PROFIT_ATR:
        if pos_type == mt5.ORDER_TYPE_BUY:
            breakeven_sl = entry + (BREAKEVEN_OFFSET_ATR * atr_val)
        else:
            breakeven_sl = entry - (BREAKEVEN_OFFSET_ATR * atr_val)
        
        breakeven_sl = normalize_price(breakeven_sl, tick_size)
        
        if validate_sl_distance(current_price, breakeven_sl, min_stop_dist, pos_type):
            current_sl = pos.sl if pos.sl != 0 else None
            
            if pos_type == mt5.ORDER_TYPE_BUY:
                if current_sl is None or breakeven_sl > current_sl:
                    metadata['breakeven_applied'] = True
                    logger.info(f"[HYBRID SL] BREAKEVEN activated for ticket {pos.ticket}: "
                               f"SL moved to {breakeven_sl:.5f} (+{BREAKEVEN_OFFSET_ATR} ATR from entry)")
                    return breakeven_sl
            else:
                if current_sl is None or breakeven_sl < current_sl:
                    metadata['breakeven_applied'] = True
                    logger.info(f"[HYBRID SL] BREAKEVEN activated for ticket {pos.ticket}: "
                               f"SL moved to {breakeven_sl:.5f} (-{BREAKEVEN_OFFSET_ATR} ATR from entry)")
                    return breakeven_sl
    
    return None

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

# ================ HYBRID TRAILING STOP MANAGEMENT ================

def manage_trailing_positions(models_cache):
    """Manage open positions with comprehensive hybrid stop-loss system"""
    positions = mt5.positions_get()
    if positions is None:
        return
    
    for pos in positions:
        comment = pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment
        if pos.magic != magic_number or comment != comment_text:
            continue
        
        symbol = pos.symbol
        ticket = pos.ticket
        
        if symbol not in models_cache:
            logger.warning(f"No models loaded for {symbol}, skipping trailing stop")
            continue
        
        clf, reg_lot, reg_trail, scaler = models_cache[symbol]
        
        pos_type = pos.type
        entry = pos.price_open
        
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
        
        if ticket not in position_metadata:
            current_volatility = df['close'].pct_change().tail(20).std()
            initialize_position_metadata(ticket, symbol, entry, datetime.now(), atr_val)
            position_metadata[ticket]['entry_volatility'] = current_volatility
        
        metadata = position_metadata[ticket]
        
        current_sl = pos.sl if pos.sl != 0 else None
        current_tp = pos.tp if pos.tp != 0 else None
        new_sl = current_sl
        new_tp = current_tp
        need_modify = False
        
        if pos_type == mt5.ORDER_TYPE_BUY:
            profit_atr = (current_price - entry) / atr_val
            if current_price > metadata['max_favorable_price']:
                metadata['max_favorable_price'] = current_price
        else:
            profit_atr = (entry - current_price) / atr_val
            if current_price < metadata['max_favorable_price']:
                metadata['max_favorable_price'] = current_price
        
        metadata['highest_profit_atr'] = max(metadata.get('highest_profit_atr', 0), profit_atr)
        
        # COMPONENT 1: Time-Based Exit
        if check_time_based_exit(pos, metadata):
            logger.warning(f"[HYBRID SL] Closing position {ticket} due to time-based exit")
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": int(ticket),
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "deviation": 20,
                "magic": magic_number,
                "comment": "Time-based exit"
            }
            mt5.order_send(close_request)
            continue
        
        # COMPONENT 2: Volatility-Adjusted Stops
        is_volatile, vol_ratio = detect_volatility_spike(df, metadata.get('entry_volatility'))
        volatility_adjustment = 1.0
        
        if is_volatile:
            volatility_adjustment = 0.7
            logger.info(f"[HYBRID SL] Volatility spike detected for {symbol} (ratio={vol_ratio:.2f}), "
                       f"tightening stops by {(1-volatility_adjustment)*100:.0f}%")
        
        # COMPONENT 3: Break-Even Stop
        breakeven_sl = apply_breakeven_stop(pos, current_price, atr_val, metadata, min_stop_dist, tick_size)
        if breakeven_sl is not None:
            new_sl = breakeven_sl
            need_modify = True
        
        # COMPONENT 4 & 5: Two-Phase Trailing (Profit Lock + Dynamic Trailing)
        if profit_atr >= 1.0 and profit_atr < 1.5:
            if not metadata.get('breakeven_applied', False):
                if pos_type == mt5.ORDER_TYPE_BUY:
                    lock_sl = entry + (0.2 * atr_val)
                    lock_sl = normalize_price(lock_sl, tick_size)
                    
                    if lock_sl < current_price - min_stop_dist:
                        if current_sl is None or lock_sl > current_sl:
                            if validate_sl_distance(current_price, lock_sl, min_stop_dist, pos_type):
                                new_sl = lock_sl
                                need_modify = True
                                logger.info(f"[HYBRID SL] Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
                else:
                    lock_sl = entry - (0.2 * atr_val)
                    lock_sl = normalize_price(lock_sl, tick_size)
                    
                    if lock_sl > current_price + min_stop_dist:
                        if current_sl is None or lock_sl < current_sl:
                            if validate_sl_distance(current_price, lock_sl, min_stop_dist, pos_type):
                                new_sl = lock_sl
                                need_modify = True
                                logger.info(f"[HYBRID SL] Phase 1: Locking profit for {symbol} ticket {ticket} at {lock_sl:.5f} (profit={profit_atr:.2f} ATR)")
        
        elif profit_atr >= 1.5:
            if not metadata.get('trailing_active', False):
                metadata['trailing_active'] = True
                logger.info(f"[HYBRID SL] Phase 2: Dynamic trailing activated for {symbol} ticket {ticket}")
            
            trail_mult = get_trail_multiplier_ml(reg_trail, scaler, df)
            trail_mult = trail_mult * volatility_adjustment
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
                                logger.info(f"[HYBRID SL] Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} "
                                          f"(mult={trail_mult:.2f}, vol_adj={volatility_adjustment:.2f}, profit={profit_atr:.2f} ATR)")
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
                                logger.info(f"[HYBRID SL] Phase 2: Trailing {symbol} ticket {ticket} to {target_sl:.5f} "
                                          f"(mult={trail_mult:.2f}, vol_adj={volatility_adjustment:.2f}, profit={profit_atr:.2f} ATR)")
                    else:
                        logger.debug(f"Skipping trail for {symbol}: target SL {target_sl:.5f} above entry {entry:.5f}")
        
        # COMPONENT 6: Fixed Take-Profit (ATR-based)
        if current_tp is None or current_tp == 0:
            tp2 = entry + (3.0 * atr_val) if pos_type == mt5.ORDER_TYPE_BUY else entry - (3.0 * atr_val)
            new_tp = normalize_price(tp2, tick_size)
            need_modify = True
            logger.info(f"[HYBRID SL] Setting hard TP for {symbol} ticket {ticket} at {new_tp:.5f}")
        
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
    
    closed_tickets = [ticket for ticket in position_metadata.keys() 
                     if not any(p.ticket == ticket for p in (positions or []))]
    for ticket in closed_tickets:
        logger.info(f"[HYBRID SL] Cleaning up metadata for closed position {ticket}")
        del position_metadata[ticket]

# ================ TRADE EXECUTION ================

def open_trade(symbol, action, clf, reg_lot, scaler):
    """Open a new trade with calculated lot size, stop loss, and take profit"""
    
    # Check position limits before opening trade
    can_open, limit_message = check_position_limits(symbol)
    if not can_open:
        logger.warning(f"[TRADE REJECTED] Cannot open {action} trade for {symbol}: {limit_message}")
        return
    
    logger.info(f"[POSITION CHECK] {limit_message}")
    
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
        ticket = result.order
        total_positions = get_total_positions()
        symbol_positions = len(check_existing_positions(symbol))
        logger.info(f"[TRADE OPENED] {symbol} {action} at {price} | SL={candidate_sl:.5f}, TP={candidate_tp2:.5f}, ATR={atr_val:.5f}, LOT={lot}")
        logger.info(f"[POSITION COUNT] Total: {total_positions}/{MAX_TOTAL_POSITIONS}, {symbol}: {symbol_positions}/{MAX_POSITIONS_PER_SYMBOL}")
        
        current_volatility = df['close'].pct_change().tail(20).std()
        initialize_position_metadata(ticket, symbol, price, datetime.now(), atr_val)
        position_metadata[ticket]['entry_volatility'] = current_volatility

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
    """Main execution loop with diagnostic capabilities and hybrid stop-loss"""
    if mode == 'collect':
        for s in symbols_list:
            collect_and_save(s, num_bars=2000)
        return
    
    if mode == 'train':
        train_all_symbols(symbols_list)
        return

    models_cache = {}
    for symbol in symbols_list:
        clf, reg_lot, reg_trail, scaler = load_models_for_symbol(symbol)
        models_cache[symbol] = (clf, reg_lot, reg_trail, scaler)

    logger.info('='*60)
    logger.info('STARTING HYBRID STOP-LOSS LIVE LOOP')
    logger.info('='*60)
    logger.info(f"Trading symbols: {symbols_list}")
    logger.info(f"Magic number: {magic_number}")
    logger.info(f"ML Probability Threshold: {ML_PROB_THRESHOLD}")
    logger.info(f"\nHybrid Stop-Loss Configuration:")
    logger.info(f"  - Break-even trigger: {BREAKEVEN_PROFIT_ATR} ATR profit")
    logger.info(f"  - Break-even offset: {BREAKEVEN_OFFSET_ATR} ATR above entry")
    logger.info(f"  - Time-based exit: {TIME_BASED_EXIT_HOURS} hours")
    logger.info(f"  - Min profit to avoid time exit: {TIME_EXIT_MIN_PROFIT_ATR} ATR")
    logger.info(f"  - Volatility spike threshold: {VOLATILITY_SPIKE_THRESHOLD}x")
    logger.info(f"\nModels loaded per symbol:")
    for symbol in symbols_list:
        clf, reg_lot, reg_trail, scaler = models_cache[symbol]
        logger.info(f"  {symbol}: clf={clf is not None}, lot_reg={reg_lot is not None}, trail_reg={reg_trail is not None}, scaler={scaler is not None}")
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
                    
                    df = get_ohlcv(symbol, 300)
                    if df is None:
                        logger.error(f"{symbol}: Failed to fetch OHLCV data")
                        continue
                    
                    logger.info(f"{symbol}: Fetched {len(df)} bars, latest time: {df['time'].iloc[-1]}")
                    
                    if iteration % 10 == 1:
                        diagnose_market_activity(symbol, df)
                    
                    clf, reg_lot, reg_trail, scaler = models_cache[symbol]
                    
                    sig = ml_filtered_signal(symbol, clf, scaler, df)
                    
                    if sig is None:
                        logger.info(f"{symbol}: No signal generated - running diagnostics...")
                        diagnose_no_signal(symbol, df)
                    else:
                        logger.info(f"[SIGNAL DETECTED] {symbol}: {sig.upper()}")
                        logger.info(f"{symbol}: Attempting to open {sig.upper()} trade...")
                        open_trade(symbol, sig, clf, reg_lot, scaler)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                    continue
            
            try:
                logger.info(f"\n--- Managing Hybrid Stop-Loss System ---")
                logger.info(f"Active positions tracked: {len(position_metadata)}")
                manage_trailing_positions(models_cache)
            except Exception as e:
                logger.error(f"Error in hybrid stop-loss management: {e}", exc_info=True)
            
            sleep_seconds = 60
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
    parser = argparse.ArgumentParser(description='ML-Enhanced Trading Bot - HYBRID STOP-LOSS VERSION')
    parser.add_argument('--mode', choices=['collect', 'train', 'live', 'backtest'], default='live',
                        help='Operation mode: collect data, train models, run live trading, or backtest')
    parser.add_argument('--symbols', nargs='*', default=symbols,
                        help='List of symbols to trade (e.g., XAUUSD XAUEUR)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RSI-MA-ATR Trading Bot - HYBRID STOP-LOSS VERSION")
    logger.info("Enhanced with 6-component hybrid stop-loss system:")
    logger.info("  1. Initial ATR-based Fixed Stop Loss")
    logger.info("  2. Break-even Stop after minimum profit")
    logger.info("  3. Two-phase Trailing (Profit Lock + Dynamic)")
    logger.info("  4. Time-based Exit monitoring")
    logger.info("  5. Volatility-adjusted Stop distance")
    logger.info("  6. Fixed Take-Profit target")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info("=" * 60)

    if args.mode == 'backtest':
        backtest_all(args.symbols)
    else:
        run_live(args.symbols, mode=args.mode)