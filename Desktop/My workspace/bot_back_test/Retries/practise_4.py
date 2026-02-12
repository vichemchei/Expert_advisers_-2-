"""
RSI-MA-ATR Trading Bot with ML integration for MT5
- Collects features & labels to CSV
- Provides training functions (RandomForest classifier + regressors)
- Integrates ML classifier as a filter for get_signal()
- Integrates ML regressor for adaptive lot sizing and trailing multiplier

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import joblib

# ================ CONFIGURATION ================
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
symbols = ["XAUUSD", "GBPJPY", "EURUSD", "USDJPY"]
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
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clf_signal.pkl")
LOT_REG_PATH = os.path.join(MODEL_DIR, "reg_lot.pkl")
TRAIL_REG_PATH = os.path.join(MODEL_DIR, "reg_trail.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ML prediction thresholds
ML_PROB_THRESHOLD = 0.65  # Minimum probability threshold to accept classifier prediction

# ================ MT5 INITIALIZATION ================
if not mt5.initialize(path=MT5_PATH):
    print("MT5 init failed:", mt5.last_error())
    # Don't quit so user can run training offline without MT5 if needed

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
    # Moving averages
    d['ema_short'] = ema(d['close'], ma_short)
    d['ema_long'] = ema(d['close'], ma_long)
    # RSI and ATR
    d['rsi'] = rsi(d['close'], rsi_period)
    d['atr'] = calc_atr(d, atr_period)
    # Derived features
    d['ema_diff'] = d['ema_short'] - d['ema_long']
    d['ret_1'] = d['close'].pct_change(1)
    d['ret_5'] = d['close'].pct_change(5)
    d['vol_10'] = d['tick_volume'].rolling(10).mean()
    d['volatility_10'] = d['close'].pct_change().rolling(10).std()
    # Price position relative to EMAs
    d['close_minus_ema_short'] = d['close'] - d['ema_short']
    d['close_minus_ema_long'] = d['close'] - d['ema_long']
    return d


def label_future(df, lookahead=20, tp_atr_mult=3.0, sl_atr_mult=1.5):
    """
    Label historical data based on future price movement
    Returns labels: 1=buy (TP hit before SL), -1=sell (TP hit before SL), 0=no clear signal
    """
    d = df.copy()
    # Calculate future high and low over lookahead period
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
        # Calculate buy scenario levels
        tp_buy = entry + tp_atr_mult * atr
        sl_buy = entry - sl_atr_mult * atr
        fut_high = d['future_high'].iloc[i]
        fut_low = d['future_low'].iloc[i]
        # Check if buy TP hit first
        if fut_high >= tp_buy and fut_low > sl_buy:
            labels.append(1)
            continue
        # Calculate sell scenario levels
        tp_sell = entry - tp_atr_mult * atr
        sl_sell = entry + sl_atr_mult * atr
        # Check if sell TP hit first
        if fut_low <= tp_sell and fut_high < sl_sell:
            labels.append(-1)
            continue
        labels.append(0)
    d['label'] = labels
    return d

# ================ DATA COLLECTION ================

def collect_and_save(symbol, out_csv, num_bars=2000, lookahead=20):
    """Collect historical data, build features, create labels, and save to CSV"""
    print(f"Collecting {symbol}...")
    df = get_ohlcv(symbol, num_bars)
    if df is None:
        print("No data for", symbol)
        return
    # Build features and labels
    feat = build_features(df)
    labeled = label_future(feat, lookahead=lookahead)
    # Keep only rows with valid data
    keep_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'ema_short', 'ema_long', 'rsi', 'atr',
                 'ema_diff', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short', 'close_minus_ema_long', 'label']
    out = labeled[keep_cols].dropna()
    # Append to existing CSV or create new one
    if os.path.exists(out_csv):
        out.to_csv(out_csv, mode='a', header=False, index=False)
    else:
        out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")

# ================ MODEL TRAINING ================

def train_models(csv_path, test_size=0.2, random_state=42):
    """Train ML models: classifier for signal prediction, regressors for lot sizing and trailing"""
    df = pd.read_csv(csv_path)
    # Define feature columns
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    X = df[feature_cols].values
    y = df['label'].values

    # Train classifier (binary: buy vs not-buy)
    y_bin = np.where(y == 1, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=test_size, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classifier report (buy vs not-buy):")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, CLASSIFIER_PATH)
    print("Saved classifier ->", CLASSIFIER_PATH)

    # Train lot size regressor on profitable buy trades
    df_profitable = df[df['label'] == 1].copy()
    if len(df_profitable) >= 50:
        # Create synthetic target: higher volatility -> smaller lot size
        target_lot = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_lot = (target_lot / target_lot.max())  # Normalize to 0..1
        target_lot = target_lot * 0.5  # Scale max to 0.5 lots
        Xl = df_profitable[feature_cols].values
        yl = target_lot.values
        reg_lot = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        reg_lot.fit(Xl, yl)
        joblib.dump(reg_lot, LOT_REG_PATH)
        print("Saved lot regressor ->", LOT_REG_PATH)
    else:
        print("Not enough profitable samples to train lot regressor; skipped.")

    # Train trailing stop regressor on profitable trades
    if len(df_profitable) >= 50:
        # Create synthetic target: lower volatility -> tighter trailing stop
        target_trail = (1.0 / (df_profitable['volatility_10'] + 1e-6))
        target_trail = (target_trail / target_trail.max())
        target_trail = 0.5 + target_trail * 1.5  # Range: 0.5 to 2.0
        Xr = df_profitable[feature_cols].values
        yr = target_trail.values
        reg_trail = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
        reg_trail.fit(Xr, yr)
        joblib.dump(reg_trail, TRAIL_REG_PATH)
        print("Saved trail regressor ->", TRAIL_REG_PATH)
    else:
        print("Not enough profitable samples to train trail regressor; skipped.")

# ================ MODEL LOADING ================

def load_models():
    """Load trained ML models from disk"""
    clf = None
    reg_lot = None
    reg_trail = None
    if os.path.exists(CLASSIFIER_PATH):
        try:
            clf = joblib.load(CLASSIFIER_PATH)
            print('Loaded classifier')
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
    # Determine trend direction
    trend_up = last['ema_short'] > last['ema_long']
    trend_down = last['ema_short'] < last['ema_long']
    # Buy signal: uptrend, price bounced from short EMA, RSI in range
    if trend_up and prev['low'] <= prev['ema_short'] and last['close'] > last['ema_short'] and 35 < last['rsi'] < 75:
        return 'buy'
    # Sell signal: downtrend, price bounced from short EMA, RSI in range
    elif trend_down and prev['high'] >= prev['ema_short'] and last['close'] < last['ema_short'] and 25 < last['rsi'] < 65:
        return 'sell'
    return None


def ml_filtered_signal(symbol, clf, df):
    """
    Generate signal using ML classifier to filter rule-based signals
    Only returns signal when both rule-based and ML agree with high confidence
    """
    if df is None or len(df) < 30:
        return None
    # Build features for latest bar
    feat = build_features(df).iloc[-1]
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    X = feat[feature_cols].values.reshape(1, -1)

    # Get ML prediction
    if clf is None:
        return rule_signal_from_df(df)
    try:
        prob = clf.predict_proba(X)[0, 1]
    except Exception:
        prob = clf.predict(X)[0]
        # If classifier only gives classes, set prob=1 for class 1
        prob = 1.0 if prob == 1 else 0.0

    # Get rule-based signal
    rule = rule_signal_from_df(df)
    # Only act when rule-based signal aligns with ML and probability is high
    if rule == 'buy' and prob >= ML_PROB_THRESHOLD:
        return 'buy'
    if rule == 'sell' and prob >= ML_PROB_THRESHOLD:
        return 'sell'
    # Optional: Allow ML-only signals (uncomment if desired)
    # if prob >= 0.9: return 'buy'
    return None

# ================ LOT SIZING & TRAILING LOGIC ================

def calculate_lot_ml(symbol, reg_lot, sl_price_distance):
    """
    Calculate lot size using risk management rules
    Refined with ML regressor prediction if available
    """
    # Fallback to rule-based calculation
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
        lot = risk_amount / (sl_pips * pip_value_per_lot)
        lot = max(round(lot, 2), 0.01)
    except Exception as e:
        print('calculate_lot default error:', e)
        lot = default_lot

    # Refine with ML regressor if available
    if reg_lot is None:
        return lot
    try:
        # Build feature vector from latest data
        df = get_ohlcv(symbol, 200)
        if df is None or len(df) < 20:
            return lot
        feat = build_features(df).iloc[-1]
        feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
        X = feat[feature_cols].values.reshape(1, -1)
        pred = reg_lot.predict(X)[0]
        # Regressor outputs absolute lot size (0..0.5 range)
        pred = max(0.01, float(pred))
        # Take conservative minimum between calculated and predicted lot
        final_lot = min(lot, pred)
        final_lot = max(final_lot, 0.01)
        return round(final_lot, 2)
    except Exception as e:
        print('lot regressor error', e)
        return lot


def get_trail_multiplier_ml(reg_trail, df):
    """
    Get trailing stop multiplier using ML regressor
    Returns ATR multiplier for trailing stop distance
    """
    if reg_trail is None or df is None or len(df) < 30:
        return 0.75  # Default multiplier
    try:
        feat = build_features(df).iloc[-1]
        feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
        X = feat[feature_cols].values.reshape(1, -1)
        pred = reg_trail.predict(X)[0]
        pred = float(pred)
        # Clamp to reasonable bounds
        pred = max(0.2, min(pred, 3.0))
        return pred
    except Exception as e:
        print('trail regressor error', e)
        return 0.75

# ================ TRAILING STOP MANAGEMENT ================

def get_min_stop_distance(symbol):
    """Get minimum allowed stop distance for the symbol from broker requirements"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 10 * 1e-5
    point = getattr(symbol_info, "point", None) or 1e-5
    # Check various possible attributes for stop level
    possible_attrs = ["trade_stops_level", "stop_level", "stoplevel", "trade_stops", "trade_tick_size"]
    for attr in possible_attrs:
        val = getattr(symbol_info, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            if val < 1000:
                return float(val) * float(point)
            return float(val)
    return max(10 * float(point), 1e-5)


def manage_trailing_positions(clf, reg_lot, reg_trail):
    """
    Manage open positions with two-phase trailing strategy:
    Phase 1: Lock in profit when price moves favorably by 1 ATR
    Phase 2: Dynamic trailing stop using ML-predicted ATR multiplier
    """
    positions = mt5.positions_get()
    if positions is None:
        return
    for pos in positions:
        # Only manage positions opened by this bot
        if pos.magic != magic_number or (pos.comment.decode() if isinstance(pos.comment, bytes) else pos.comment) != comment_text:
            continue
        symbol = pos.symbol
        pos_type = pos.type
        entry = pos.price_open
        ticket = pos.ticket
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        # Get current price based on position direction
        current_price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
        # Fetch data and calculate ATR
        df = get_ohlcv(symbol, 200)
        if df is None or len(df) < 30:
            continue
        atr_series = calc_atr(df, atr_period)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            continue
        atr_val = float(atr_series.iloc[-1])
        min_stop_dist = get_min_stop_distance(symbol)
        # Calculate key price levels
        if pos_type == mt5.ORDER_TYPE_BUY:
            activation_price = entry + (1.0 * atr_val)
            tp1 = entry + (1.0 * atr_val)
            tp2 = entry + (3.0 * atr_val)
            lock_sl = entry + (0.2 * atr_val)
        else:
            activation_price = entry - (1.0 * atr_val)
            tp1 = entry - (1.0 * atr_val)
            tp2 = entry - (3.0 * atr_val)
            lock_sl = entry - (0.2 * atr_val)

        current_sl = pos.sl
        current_tp = pos.tp
        need_modify = False
        new_sl = current_sl
        new_tp = current_tp

        # Phase 1: Lock in profit when activation price reached
        if (pos_type == mt5.ORDER_TYPE_BUY and current_price >= activation_price) or (pos_type == mt5.ORDER_TYPE_SELL and current_price <= activation_price):
            if pos_type == mt5.ORDER_TYPE_BUY and (current_sl is None or current_sl < lock_sl):
                candidate = max(lock_sl, entry + min_stop_dist)
                new_sl = candidate
                need_modify = True
            elif pos_type == mt5.ORDER_TYPE_SELL and (current_sl is None or current_sl > lock_sl):
                candidate = min(lock_sl, entry - min_stop_dist)
                new_sl = candidate
                need_modify = True

        # Phase 2: Dynamic trailing stop using ML-predicted multiplier
        trail_mult = get_trail_multiplier_ml(reg_trail, df)
        if (pos_type == mt5.ORDER_TYPE_BUY and current_price >= tp1) or (pos_type == mt5.ORDER_TYPE_SELL and current_price <= tp1):
            trail_dist = max(trail_mult * atr_val, min_stop_dist)
            if pos_type == mt5.ORDER_TYPE_BUY:
                target_sl = current_price - trail_dist
                if current_sl is None or target_sl > current_sl:
                    if (current_price - target_sl) >= min_stop_dist:
                        new_sl = target_sl
                        need_modify = True
            else:
                target_sl = current_price + trail_dist
                if current_sl is None or target_sl < current_sl:
                    if (target_sl - current_price) >= min_stop_dist:
                        new_sl = target_sl
                        need_modify = True

        # Ensure hard take profit at TP2 level
        if pos_type == mt5.ORDER_TYPE_BUY:
            if current_tp is None or abs(current_tp - tp2) > (0.5 * atr_val):
                new_tp = tp2
                need_modify = True
        else:
            if current_tp is None or abs(current_tp - tp2) > (0.5 * atr_val):
                new_tp = tp2
                need_modify = True

        # Send modification request if needed
        if need_modify:
            modify_req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(new_sl) if new_sl is not None else 0.0,
                "tp": float(new_tp) if new_tp is not None else 0.0,
                "magic": magic_number,
                "comment": comment_text
            }
            res = mt5.order_send(modify_req)
            if res is None:
                print(f"Modify failed for {symbol} ticket {ticket}: None")
            else:
                print(f"Modified pos {ticket} {symbol} -> SL={new_sl}, TP={new_tp}, retcode={getattr(res, 'retcode', 'N/A')}")

# ================ TRADE EXECUTION ================

def open_trade(symbol, action, clf, reg_lot):
    """
    Open a new trade with calculated lot size, stop loss, and take profit
    Uses ML-based lot sizing if regressor is available
    """
    if not mt5.symbol_select(symbol, True):
        print(f"Symbol {symbol} not available / not selected")
        return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"No tick for {symbol}")
        return
    # Get entry price based on direction
    price = tick.ask if action == "buy" else tick.bid
    # Fetch data and calculate ATR
    df = get_ohlcv(symbol, 200)
    if df is None or len(df) < 20:
        return
    atr_series = calc_atr(df, atr_period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return
    atr_val = float(atr_series.iloc[-1])

    # Define ATR multipliers for SL and TP
    atr_mult_sl = 1.5
    atr_mult_tp = 3.0

    # Calculate stop loss and take profit levels
    if action == "buy":
        candidate_sl = price - (atr_mult_sl * atr_val)
        candidate_tp2 = price + (atr_mult_tp * atr_val)
    else:
        candidate_sl = price + (atr_mult_sl * atr_val)
        candidate_tp2 = price - (atr_mult_tp * atr_val)

    # Check minimum stop distance
    min_stop_dist = get_min_stop_distance(symbol)
    sl_distance = abs(price - candidate_sl)
    use_stops = True
    if sl_distance < min_stop_dist:
        use_stops = False

    # Calculate lot size using ML regressor if available
    lot = calculate_lot_ml(symbol, reg_lot, sl_distance if use_stops else None)
    if lot < 0.01:
        lot = default_lot

    filling_mode = mt5.ORDER_FILLING_IOC

    # Prepare order request
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

    # Add stop loss and take profit if using stops
    if use_stops:
        request["sl"] = candidate_sl
        request["tp"] = candidate_tp2

    # Send order
    result = mt5.order_send(request)
    if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
        print(f"{symbol} {action} failed: {getattr(result, 'retcode', 'N/A')}")
    else:
        print(f"{symbol} {action} opened at {price} | SL={candidate_sl:.5f}, TP(hard)={candidate_tp2:.5f}, ATR={atr_val:.5f}, LOT={lot}")

# ================ BACKTEST UTILITY ================

def backtest(csv_path):
    """
    Simple backtest: evaluate classifier precision on historical data
    Loads trained classifier and evaluates performance on full dataset
    """
    df = pd.read_csv(csv_path)
    feature_cols = ['rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 'volatility_10', 'close_minus_ema_short']
    df = df.dropna(subset=feature_cols + ['label'])
    clf, _, _ = load_models()
    if clf is None:
        print('No classifier to evaluate. Train models first.')
        return
    X = df[feature_cols].values
    y = df['label'].values
    y_bin = np.where(y == 1, 1, 0)
    preds = clf.predict(X)
    print(classification_report(y_bin, preds))

# ================ MAIN LIVE LOOP ================

def run_live(symbols, mode='live', data_csv='training_data.csv'):
    """
    Main execution loop with three modes:
    - 'collect': Collect historical data and save to CSV
    - 'train': Train ML models from collected data
    - 'live': Run live trading with signal generation and position management
    """
    clf, reg_lot, reg_trail = load_models()

    if mode == 'collect':
        # Collect data once per symbol and append to CSV
        for s in symbols:
            collect_and_save(s, data_csv, num_bars=2000)
        return
    if mode == 'train':
        # Train all ML models from collected data
        train_models(data_csv)
        return

    # Live trading mode
    print('Starting live loop...')
    try:
        while True:
            # Process each symbol
            for symbol in symbols:
                # Fetch data and check for signals
                df = get_ohlcv(symbol, 500)
                sig = ml_filtered_signal(symbol, clf, df)
                if sig:
                    print(f"Signal detected: {symbol} - {sig}")
                    open_trade(symbol, sig, clf, reg_lot)
                # Manage trailing stops on open positions
                manage_trailing_positions(clf, reg_lot, reg_trail)
            # Wait 5 minutes before next check
            time.sleep(300)
    except KeyboardInterrupt:
        print('Stopping live loop')
    finally:
        if mt5.initialize():
            mt5.shutdown()

# ================ CLI ENTRY POINT ================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['collect', 'train', 'live', 'backtest'], default='live',
                        help='Operation mode: collect data, train models, run live trading, or backtest')
    parser.add_argument('--data', default='training_data.csv',
                        help='Path to CSV file for training data')
    parser.add_argument('--symbols', nargs='*', default=symbols,
                        help='List of symbols to trade')
    args = parser.parse_args()

    if args.mode == 'backtest':
        backtest(args.data)
    else:
        run_live(args.symbols, mode=args.mode, data_csv=args.data)