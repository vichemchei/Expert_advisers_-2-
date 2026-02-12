"""
MetaTrader 5 RSI-MA-ATR Machine Learning Trading Bot
Institutional-Grade Refactored Implementation
Version: 2.0.2 - Complete Fix

Fixed Issues:
- Enhanced debug logging to identify why trades aren't executing
- Fixed ML result unpacking issue
- Added detailed signal generation logging
- Improved error handling and visibility
- FIXED lot size calculation with proper MT5 formula
- FIXED SL/TP ratios for proper risk:reward (1:2)
- Complete RiskManager class with all methods
"""

import os
import time
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
import sys
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Position:
    """Position data structure"""
    ticket: int
    symbol: str
    type: int  # mt5.ORDER_TYPE_BUY or SELL
    volume: float
    entry_price: float
    entry_time: datetime
    sl: float
    tp: float
    current_price: float = 0.0
    profit: float = 0.0
    
    @property
    def is_buy(self) -> bool:
        return self.type == mt5.ORDER_TYPE_BUY
    
    @property
    def profit_atr(self) -> float:
        """Calculate profit in ATR units"""
        if not hasattr(self, '_atr') or self._atr <= 0:
            return 0.0
        if self.is_buy:
            return (self.current_price - self.entry_price) / self._atr
        else:
            return (self.entry_price - self.current_price) / self._atr
    
    def set_atr(self, atr: float):
        """Set ATR value for calculations"""
        self._atr = atr


@dataclass
class TradeSignal:
    """Trading signal structure"""
    symbol: str
    direction: str  # 'buy' or 'sell'
    timestamp: datetime
    confidence: float  # ML probability
    features: Dict[str, float]
    price: float
    atr: float


@dataclass
class TrailingStop:
    """Trailing stop state with hybrid management"""
    ticket: int
    symbol: str
    entry_price: float
    entry_time: datetime
    pos_type: int
    atr: float
    # Stop loss triggers
    be_triggered: bool = False  # Break-even triggered
    trailing_triggered: bool = False  # Standard trailing triggered
    aggressive_triggered: bool = False  # Aggressive trailing triggered
    partial_triggered: bool = False  # Partial close triggered
    # Profit tracking
    highest_profit_atr: float = 0.0  # Max profit in ATR units
    highest_profit_pips: float = 0.0  # Max profit in pips
    current_profit_atr: float = 0.0  # Current profit in ATR
    current_profit_pips: float = 0.0  # Current profit in pips
    # Stop loss state
    last_sl: Optional[float] = None  # Last set stop loss
    trail_mode: str = "initial"  # Current trailing mode
    momentum_score: float = 0.0  # Current momentum strength


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Force UTF-8 encoding for console on Windows
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# CSV TRADE LOGGER
# ============================================================================

class CSVTradeLogger:
    """CSV-based trade logging for analysis"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.trades_log = os.path.join(log_dir, "trades.csv")
        self.signals_log = os.path.join(log_dir, "signals.csv")
        self.performance_log = os.path.join(log_dir, "daily_performance.csv")
        
        self._init_files()
    
    def _init_files(self):
        """Initialize CSV files with headers"""
        if not os.path.exists(self.trades_log):
            with open(self.trades_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'ticket', 'entry_price',
                    'lot_size', 'sl', 'tp', 'atr', 'exit_time', 'exit_price',
                    'profit', 'hold_hours', 'exit_reason'
                ])
        
        if not os.path.exists(self.signals_log):
            with open(self.signals_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'signal_type', 'ml_confidence',
                    'rsi', 'ema_diff', 'atr', 'executed', 'rejection_reason'
                ])
        
        if not os.path.exists(self.performance_log):
            with open(self.performance_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'start_balance', 'end_balance', 'daily_pnl',
                    'daily_pnl_pct', 'trades', 'wins', 'losses', 'win_rate'
                ])
    
    def log_trade_open(self, symbol: str, action: str, ticket: int, 
                       entry_price: float, lot_size: float, sl: float, 
                       tp: float, atr: float):
        """Log trade opening"""
        try:
            with open(self.trades_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), symbol, action.upper(), ticket,
                    f"{entry_price:.5f}", lot_size, f"{sl:.5f}", f"{tp:.5f}",
                    f"{atr:.5f}", "", "", "", "", ""
                ])
        except Exception as e:
            logging.error(f"Error logging trade open: {e}")
    
    def log_trade_close(self, ticket: int, exit_price: float, 
                        profit: float, exit_reason: str):
        """Log trade closure"""
        try:
            with open(self.trades_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), "CLOSE", "", ticket, "",
                    "", "", "", "", datetime.now().isoformat(),
                    f"{exit_price:.5f}", f"{profit:.2f}", "", exit_reason
                ])
        except Exception as e:
            logging.error(f"Error logging trade close: {e}")
    
    def log_signal(self, symbol: str, signal_type: str, ml_confidence: float,
                   rsi: float, ema_diff: float, atr: float, 
                   executed: bool, rejection_reason: str = ""):
        """Log trading signal"""
        try:
            with open(self.signals_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), symbol, signal_type,
                    f"{ml_confidence:.3f}",
                    f"{rsi:.2f}", f"{ema_diff:.5f}", f"{atr:.5f}",
                    "YES" if executed else "NO", rejection_reason
                ])
        except Exception as e:
            logging.error(f"Error logging signal: {e}")
    
    def log_daily_performance(self, start_balance: float, end_balance: float,
                             trades: int, wins: int, losses: int):
        """Log daily performance"""
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
            logging.error(f"Error logging daily performance: {e}")


# ============================================================================
# DATA HANDLER
# ============================================================================

class DataHandler:
    """Handles data fetching and caching"""
    
    def __init__(self, timeframe: int = mt5.TIMEFRAME_M5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.timeframe = timeframe
        self.cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
    
    def fetch_ohlcv(self, symbol: str, num_bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, num_bars)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            
            # Cache data
            self.cache[symbol] = df
            self.last_update[symbol] = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_cached(self, symbol: str, max_age_seconds: int = 60) -> Optional[pd.DataFrame]:
        """Get cached data if recent enough"""
        if symbol not in self.cache:
            return None
        
        age = (datetime.now() - self.last_update.get(symbol, datetime.min)).total_seconds()
        if age > max_age_seconds:
            return None
        
        return self.cache[symbol]


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/period, adjust=False).mean()
        ma_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-12)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        if df is None or len(df) < 2:
            return pd.Series()
        
        d = df.copy()
        d['h-l'] = d['high'] - d['low']
        d['h-pc'] = (d['high'] - d['close'].shift(1)).abs()
        d['l-pc'] = (d['low'] - d['close'].shift(1)).abs()
        d['tr'] = d[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        if len(d) < period:
            atr_series = d['tr'].expanding(min_periods=2).mean()
        else:
            atr_series = d['tr'].rolling(window=period, min_periods=period).mean()
        
        # Forward fill NaN values
        atr_series = atr_series.bfill().ffill()
        
        return atr_series
    
    @classmethod
    def build_features(cls, df: pd.DataFrame, 
                      rsi_period: int = 14,
                      ma_short: int = 20,
                      ma_long: int = 50,
                      atr_period: int = 14) -> pd.DataFrame:
        """Build comprehensive feature set"""
        d = df.copy()
        
        # Base indicators
        d['ema_short'] = cls.ema(d['close'], ma_short)
        d['ema_long'] = cls.ema(d['close'], ma_long)
        d['rsi'] = cls.rsi(d['close'], rsi_period)
        d['atr'] = cls.atr(d, atr_period)
        
        # Derived features
        d['ema_diff'] = d['ema_short'] - d['ema_long']
        d['ret_1'] = d['close'].pct_change(1)
        d['ret_5'] = d['close'].pct_change(5)
        d['vol_10'] = d['tick_volume'].rolling(10, min_periods=1).mean()
        d['volatility_10'] = d['close'].pct_change().rolling(10, min_periods=1).std()
        d['close_minus_ema_short'] = d['close'] - d['ema_short']
        d['close_minus_ema_long'] = d['close'] - d['ema_long']
        
        # Enhanced features
        d['rsi_momentum'] = d['rsi'].diff(3)
        d['ema_diff_pct'] = d['ema_diff'] / d['close'].replace(0, np.nan)
        d['atr_pct'] = d['atr'] / d['close'].replace(0, np.nan)
        d['volume_ratio'] = d['tick_volume'] / d['vol_10'].replace(0, np.nan)
        d['price_distance_short'] = (d['close'] - d['ema_short']) / d['atr'].replace(0, np.nan)
        d['price_distance_long'] = (d['close'] - d['ema_long']) / d['atr'].replace(0, np.nan)
        d['trend_strength'] = d['ema_diff'].abs() / d['atr'].replace(0, np.nan)
        
        # Volatility regime
        vol_mean_50 = d['volatility_10'].rolling(50, min_periods=10).mean()
        d['volatility_regime'] = d['volatility_10'] / vol_mean_50.replace(0, np.nan)
        
        # Fill NaN values
        d = d.fillna(0)
        
        return d


# ============================================================================
# ML MODEL MANAGER
# ============================================================================

class MLModelManager:
    """Manages ML models for signal generation"""
    
    def __init__(self, model_dir: str = "models"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.classifier_path = os.path.join(model_dir, "clf_signal.pkl")
        self.regressor_path = os.path.join(model_dir, "reg_lot.pkl")
        
        self.buy_classifier: Optional[RandomForestClassifier] = None
        self.sell_classifier: Optional[RandomForestClassifier] = None
        self.lot_regressor: Optional[RandomForestRegressor] = None
        
        self.feature_cols = [
            'rsi', 'ema_diff', 'atr', 'ret_1', 'ret_5', 'vol_10', 
            'volatility_10', 'close_minus_ema_short', 'rsi_momentum', 
            'ema_diff_pct', 'atr_pct', 'volume_ratio', 'price_distance_short', 
            'trend_strength', 'volatility_regime'
        ]
    
    def load_models(self) -> bool:
        """Load trained models"""
        success = True
        
        # Load classifiers
        if os.path.exists(self.classifier_path):
            try:
                clf_dict = joblib.load(self.classifier_path)
                self.buy_classifier = clf_dict.get('buy')
                self.sell_classifier = clf_dict.get('sell')
                self.logger.info(f"{Fore.GREEN}✓ Loaded classifiers{Style.RESET_ALL}")
            except Exception as e:
                self.logger.error(f"Failed to load classifiers: {e}")
                success = False
        else:
            self.logger.warning("No classifier model found")
            success = False
        
        # Load regressor
        if os.path.exists(self.regressor_path):
            try:
                self.lot_regressor = joblib.load(self.regressor_path)
                self.logger.info(f"{Fore.GREEN}✓ Loaded lot regressor{Style.RESET_ALL}")
            except Exception as e:
                self.logger.error(f"Failed to load regressor: {e}")
        else:
            self.logger.warning("No lot regressor found")
        
        return success
    
    def predict_signal(self, features: pd.Series, 
                       rule_signal: Optional[str]) -> Optional[Tuple[str, float]]:
        """Predict trading signal with confidence"""
        if rule_signal is None:
            return None
        
        # Check for NaN features
        if features[self.feature_cols].isna().any():
            self.logger.warning("NaN features detected, skipping prediction")
            return None
        
        X = features[self.feature_cols].values.reshape(1, -1)
        
        try:
            if rule_signal == 'buy' and self.buy_classifier is not None:
                proba = self.buy_classifier.predict_proba(X)[0]
                confidence = proba[1] if len(proba) > 1 else proba[0]
                return ('buy', confidence)
            
            elif rule_signal == 'sell' and self.sell_classifier is not None:
                proba = self.sell_classifier.predict_proba(X)[0]
                confidence = proba[1] if len(proba) > 1 else proba[0]
                return ('sell', confidence)
            
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
        
        return None
    
    def predict_lot_multiplier(self, features: pd.Series) -> float:
        """Predict lot size multiplier"""
        if self.lot_regressor is None:
            return 1.0
        
        try:
            if features[self.feature_cols].isna().any():
                return 1.0
            
            X = features[self.feature_cols].values.reshape(1, -1)
            pred = self.lot_regressor.predict(X)[0]
            
            # Convert to multiplier (0.5 to 1.5)
            multiplier = 0.7 + (pred * 0.8)
            multiplier = max(0.5, min(multiplier, 1.5))
            
            return multiplier
            
        except Exception as e:
            self.logger.error(f"Lot prediction error: {e}")
            return 1.0


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Generate trading signals"""
    
    def __init__(self, rsi_period: int = 14, ma_short: int = 20, 
                 ma_long: int = 50, atr_period: int = 14):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.atr_period = atr_period
    
    def generate_rule_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Generate rule-based signal with detailed logging"""
        if df is None or len(df) < max(self.ma_long, self.rsi_period) + 2:
            self.logger.debug("Not enough data for signal generation")
            return None
        
        # Build features
        df_feat = TechnicalIndicators.build_features(
            df, self.rsi_period, self.ma_short, self.ma_long, self.atr_period
        )
        
        last = df_feat.iloc[-1]
        prev = df_feat.iloc[-2]
        
        # Log current technical state
        self.logger.debug(f"Technical State - RSI: {last['rsi']:.2f}, "
                         f"EMA_short: {last['ema_short']:.5f}, "
                         f"EMA_long: {last['ema_long']:.5f}, "
                         f"Price: {last['close']:.5f}, "
                         f"ATR: {last['atr']:.5f}")
        
        # Check trend strength
        trend_strength = abs(last['ema_short'] - last['ema_long']) / last['atr']
        self.logger.debug(f"Trend strength: {trend_strength:.2f} ATR")
        
        if trend_strength < 0.5:
            self.logger.debug(f"Trend too weak ({trend_strength:.2f} < 0.5 ATR)")
            return None
        
        trend_up = last['ema_short'] > last['ema_long']
        trend_down = last['ema_short'] < last['ema_long']
        
        # Buy signal
        if trend_up:
            self.logger.debug(f"Uptrend detected. Checking buy conditions...")
            self.logger.debug(f"  prev_low <= ema_short: {prev['low']:.5f} <= {prev['ema_short']:.5f} = {prev['low'] <= prev['ema_short']}")
            self.logger.debug(f"  close > ema_short: {last['close']:.5f} > {last['ema_short']:.5f} = {last['close'] > last['ema_short']}")
            self.logger.debug(f"  RSI in range: 35 < {last['rsi']:.2f} < 70 = {35 < last['rsi'] < 70}")
            
            if (prev['low'] <= prev['ema_short'] and 
                last['close'] > last['ema_short'] and 
                35 < last['rsi'] < 70):
                self.logger.info(f"{Fore.YELLOW}>>> BUY SIGNAL CONDITIONS MET <<<{Style.RESET_ALL}")
                return 'buy'
        
        # Sell signal
        elif trend_down:
            self.logger.debug(f"Downtrend detected. Checking sell conditions...")
            self.logger.debug(f"  prev_high >= ema_short: {prev['high']:.5f} >= {prev['ema_short']:.5f} = {prev['high'] >= prev['ema_short']}")
            self.logger.debug(f"  close < ema_short: {last['close']:.5f} < {last['ema_short']:.5f} = {last['close'] < last['ema_short']}")
            self.logger.debug(f"  RSI in range: 30 < {last['rsi']:.2f} < 65 = {30 < last['rsi'] < 65}")
            
            if (prev['high'] >= prev['ema_short'] and 
                last['close'] < last['ema_short'] and 
                30 < last['rsi'] < 65):
                self.logger.info(f"{Fore.YELLOW}>>> SELL SIGNAL CONDITIONS MET <<<{Style.RESET_ALL}")
                return 'sell'
        
        return None


# ============================================================================
# RISK MANAGER - COMPLETE FIXED VERSION
# ============================================================================

class RiskManager:
    """Manages risk and position sizing - COMPLETE FIXED VERSION"""
    
    def __init__(self, 
                 risk_per_trade_pct: float = 2.0,
                 max_positions_per_symbol: int = 2,
                 max_total_positions: int = 3,
                 max_daily_loss_pct: float = 5.0,
                 max_total_risk_pct: float = 5.0,
                 force_min_lot: bool = True):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions_per_symbol = max_positions_per_symbol
        self.max_total_positions = max_total_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.force_min_lot = force_min_lot
        
        self.starting_balance = 0.0
        self.current_balance = 0.0
        self.trading_suspended = False
        self.last_reset_date = datetime.now().date()
        
        # Daily stats
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.closed_tickets = set()
        
        # Log risk settings
        self.logger.info(f"{Fore.CYAN}Risk Manager Initialized:")
        self.logger.info(f"  Risk per trade: {self.risk_per_trade_pct}%")
        self.logger.info(f"  Max positions per symbol: {self.max_positions_per_symbol}")
        self.logger.info(f"  Max total positions: {self.max_total_positions}")
        self.logger.info(f"  Force min lot: {self.force_min_lot}{Style.RESET_ALL}")
    
    def update_balance(self):
        """Update balance from MT5"""
        acc_info = mt5.account_info()
        if acc_info:
            self.current_balance = acc_info.balance
            
            # Reset on new day
            current_date = datetime.now().date()
            if current_date > self.last_reset_date:
                self.reset_daily()
                self.last_reset_date = current_date
    
    def reset_daily(self):
        """Reset daily tracking"""
        self.starting_balance = self.current_balance
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.closed_tickets = set()
        self.trading_suspended = False
        
        self.logger.info(
            f"{Fore.GREEN}[DAILY RESET] Starting balance: ${self.starting_balance:.2f}{Style.RESET_ALL}"
        )
    
    def check_daily_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        if self.trading_suspended:
            return False
        
        self.update_balance()
        
        daily_pnl = self.current_balance - self.starting_balance
        if daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / self.starting_balance if self.starting_balance > 0 else 0
            
            if daily_loss_pct >= (self.max_daily_loss_pct / 100):
                self.trading_suspended = True
                self.logger.critical(
                    f"{Fore.RED}[TRADING SUSPENDED] Daily loss: {daily_loss_pct*100:.2f}%{Style.RESET_ALL}"
                )
                return False
        
        return True
    
    def calculate_lot_size(self, symbol: str, sl_distance: float, 
                           ml_multiplier: float = 1.0) -> float:
        """
        Calculate position size using proper MT5 formula
        
        Formula: Lot = Risk Amount / (SL in ticks × Tick Value)
        """
        try:
            acc_info = mt5.account_info()
            if acc_info is None:
                self.logger.error("Cannot get account info")
                return 0.01 if self.force_min_lot else 0.0
            
            if sl_distance <= 0:
                self.logger.error(f"Invalid SL distance: {sl_distance}")
                return 0.01 if self.force_min_lot else 0.0
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return 0.01 if self.force_min_lot else 0.0
            
            # Account parameters
            balance = acc_info.balance
            equity = acc_info.equity
            free_margin = acc_info.margin_free
            
            # Use equity if it's lower (account in drawdown)
            effective_balance = min(balance, equity)
            risk_amount = effective_balance * (self.risk_per_trade_pct / 100)
            
            # Symbol parameters
            point = symbol_info.point if symbol_info.point else 1e-5
            tick_size = symbol_info.trade_tick_size if symbol_info.trade_tick_size else point
            tick_value = symbol_info.trade_tick_value if symbol_info.trade_tick_value else 1.0
            min_volume = symbol_info.volume_min if symbol_info.volume_min else 0.01
            max_volume = symbol_info.volume_max if symbol_info.volume_max else 100.0
            volume_step = symbol_info.volume_step if symbol_info.volume_step else 0.01
            
            # Calculate lot size
            # Step 1: Convert SL distance to number of ticks
            sl_in_ticks = sl_distance / tick_size
            
            # Step 2: Calculate money at risk per 1 lot
            money_per_lot = sl_in_ticks * tick_value
            
            if money_per_lot <= 0:
                self.logger.error(f"Invalid calculation: money_per_lot={money_per_lot}")
                return 0.01 if self.force_min_lot else 0.0
            
            # Step 3: Calculate base lot size
            base_lot = risk_amount / money_per_lot
            
            # Step 4: Apply ML multiplier
            base_lot *= ml_multiplier
            
            # Step 5: Normalize to volume step
            base_lot = round(base_lot / volume_step) * volume_step
            
            # Step 6: Apply limits
            calculated_lot = max(min_volume, min(base_lot, max_volume))
            
            # Detailed logging
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"{Fore.CYAN}📊 LOT SIZE CALCULATION FOR {symbol}{Style.RESET_ALL}")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"💰 Account:")
            self.logger.info(f"   Balance: ${balance:.2f}")
            self.logger.info(f"   Equity: ${equity:.2f}")
            self.logger.info(f"   Free Margin: ${free_margin:.2f}")
            self.logger.info(f"   Effective Balance: ${effective_balance:.2f}")
            self.logger.info(f"   Risk %: {self.risk_per_trade_pct}%")
            self.logger.info(f"   Risk Amount: ${risk_amount:.2f}")
            self.logger.info(f"\n📏 Symbol Parameters:")
            self.logger.info(f"   Point: {point}")
            self.logger.info(f"   Tick Size: {tick_size}")
            self.logger.info(f"   Tick Value: ${tick_value:.2f}")
            self.logger.info(f"   Volume Min/Max: {min_volume}/{max_volume}")
            self.logger.info(f"   Volume Step: {volume_step}")
            self.logger.info(f"\n🎯 Calculation:")
            self.logger.info(f"   SL Distance: {sl_distance:.5f}")
            self.logger.info(f"   SL in Ticks: {sl_in_ticks:.2f}")
            self.logger.info(f"   Money at Risk per Lot: ${money_per_lot:.2f}")
            self.logger.info(f"   ML Multiplier: {ml_multiplier:.2f}")
            self.logger.info(f"   Base Lot (before limits): {base_lot:.4f}")
            self.logger.info(f"   Final Lot (after limits): {calculated_lot:.2f}")
            
            # Check if we hit minimum
            if calculated_lot <= min_volume:
                actual_risk = calculated_lot * money_per_lot
                actual_risk_pct = (actual_risk / effective_balance * 100) if effective_balance > 0 else 0
                
                self.logger.warning(f"\n{Fore.YELLOW}⚠️  LOT SIZE AT MINIMUM!{Style.RESET_ALL}")
                self.logger.warning(f"   Requested risk: ${risk_amount:.2f} ({self.risk_per_trade_pct}%)")
                self.logger.warning(f"   Actual risk: ${actual_risk:.2f} ({actual_risk_pct:.2f}%)")
                self.logger.warning(f"   Shortfall: ${risk_amount - actual_risk:.2f}")
                self.logger.warning(f"\n💡 Solutions:")
                self.logger.warning(f"   1. Increase account balance")
                self.logger.warning(f"   2. Tighten stop loss (current: {sl_distance:.5f})")
                self.logger.warning(f"   3. Increase risk % (current: {self.risk_per_trade_pct}%)")
                
                if not self.force_min_lot:
                    self.logger.error(f"   ❌ Trade REJECTED (force_min_lot=False)")
                    self.logger.info(f"{'='*70}\n")
                    return 0.0
            
            self.logger.info(f"\n{Fore.GREEN}✅ Lot size: {calculated_lot:.2f}{Style.RESET_ALL}")
            self.logger.info(f"{'='*70}\n")
            
            return calculated_lot
            
        except Exception as e:
            self.logger.error(f"Lot calculation error: {e}", exc_info=True)
            return 0.01 if self.force_min_lot else 0.0
    
    def can_open_trade(self, symbol: str, position_count: Dict[str, int]) -> Tuple[bool, str]:
        """Check if new trade can be opened"""
        # Check daily limit
        if not self.check_daily_limit():
            return False, "Daily loss limit exceeded"
        
        # Check symbol limit
        symbol_positions = position_count.get(symbol, 0)
        if symbol_positions >= self.max_positions_per_symbol:
            return False, f"Max positions for {symbol}: {self.max_positions_per_symbol}"
        
        # Check total limit
        total_positions = sum(position_count.values())
        if total_positions >= self.max_total_positions:
            return False, f"Max total positions: {self.max_total_positions}"
        
        return True, ""


# ============================================================================
# POSITION TRACKER
# ============================================================================

class PositionTracker:
    """Track and manage open positions"""
    
    def __init__(self, magic_number: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.magic_number = magic_number
        self.positions: Dict[int, Position] = {}
        self.last_update = datetime.now()
    
    def update(self) -> List[int]:
        """Update positions and return closed ticket list"""
        mt5_positions = mt5.positions_get(magic=self.magic_number)
        current_tickets = set()
        
        if mt5_positions:
            for pos in mt5_positions:
                ticket = pos.ticket
                current_tickets.add(ticket)
                
                # Add or update position
                if ticket not in self.positions:
                    self.positions[ticket] = Position(
                        ticket=ticket,
                        symbol=pos.symbol,
                        type=pos.type,
                        volume=pos.volume,
                        entry_price=pos.price_open,
                        entry_time=datetime.fromtimestamp(pos.time),
                        sl=pos.sl,
                        tp=pos.tp,
                        current_price=pos.price_current,
                        profit=pos.profit
                    )
                    self.logger.info(
                        f"{Fore.CYAN}[NEW POSITION] {pos.symbol} ticket {ticket}{Style.RESET_ALL}"
                    )
                else:
                    # Update current price and profit
                    self.positions[ticket].current_price = pos.price_current
                    self.positions[ticket].profit = pos.profit
                    self.positions[ticket].sl = pos.sl
                    self.positions[ticket].tp = pos.tp
        
        # Detect closed positions
        closed_tickets = list(set(self.positions.keys()) - current_tickets)
        
        for ticket in closed_tickets:
            pos = self.positions[ticket]
            self.logger.info(
                f"{Fore.YELLOW}[POSITION CLOSED] {pos.symbol} ticket {ticket} | "
                f"Profit: ${pos.profit:.2f}{Style.RESET_ALL}"
            )
            del self.positions[ticket]
        
        self.last_update = datetime.now()
        return closed_tickets
    
    def get_position(self, ticket: int) -> Optional[Position]:
        """Get position by ticket"""
        return self.positions.get(ticket)
    
    def get_count(self, symbol: Optional[str] = None) -> int:
        """Get position count"""
        if symbol:
            return sum(1 for p in self.positions.values() if p.symbol == symbol)
        return len(self.positions)
    
    def get_count_by_symbol(self) -> Dict[str, int]:
        """Get position count per symbol"""
        counts = {}
        for pos in self.positions.values():
            counts[pos.symbol] = counts.get(pos.symbol, 0) + 1
        return counts


# ============================================================================
# TRAILING STOP MANAGER
# ============================================================================

class TrailingStopManager:
    """Manages trailing stops for positions"""
    
    def __init__(self, magic_number: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.magic_number = magic_number
        self.trailing_stops: Dict[int, TrailingStop] = {}
    
    def add_position(self, ticket: int, symbol: str, entry_price: float,
                     entry_time: datetime, pos_type: int, atr: float):
        """Add position to trailing stop tracking"""
        self.trailing_stops[ticket] = TrailingStop(
            ticket=ticket,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            pos_type=pos_type,
            atr=atr
        )
        self.logger.debug(f"Added trailing stop for ticket {ticket}")
    
    def remove_position(self, ticket: int):
        """Remove position from tracking"""
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]
    
    def calculate_stop(self, ticket: int, current_price: float) -> Tuple[Optional[float], str]:
        """Calculate hybrid trailing stop level"""
        if ticket not in self.trailing_stops:
            return None, "No trailing stop"
        
        trail_stop = self.trailing_stops[ticket]
        is_buy = (trail_stop.pos_type == mt5.ORDER_TYPE_BUY)
        
        # Calculate profits
        pip_value = trail_stop.atr * 0.1
        if is_buy:
            trail_stop.current_profit_atr = (current_price - trail_stop.entry_price) / trail_stop.atr
            trail_stop.current_profit_pips = (current_price - trail_stop.entry_price) / pip_value
        else:
            trail_stop.current_profit_atr = (trail_stop.entry_price - current_price) / trail_stop.atr
            trail_stop.current_profit_pips = (trail_stop.entry_price - current_price) / pip_value
        
        # Update highest profits
        trail_stop.highest_profit_atr = max(trail_stop.highest_profit_atr, trail_stop.current_profit_atr)
        trail_stop.highest_profit_pips = max(trail_stop.highest_profit_pips, trail_stop.current_profit_pips)
        
        # Time-based checks
        hours_held = (datetime.now() - trail_stop.entry_time).total_seconds() / 3600
        trail_stop.momentum_score = trail_stop.current_profit_pips / (hours_held + 1)
        
        # Quick loss exit
        if hours_held >= 4 and trail_stop.current_profit_atr < 0.5:
            return "CLOSE", f"Time exit ({hours_held:.1f}h, profit={trail_stop.current_profit_atr:.2f}ATR)"
        
        # Aggressive trailing
        if (trail_stop.current_profit_atr >= 3.0 or 
            (trail_stop.current_profit_pips >= 50 and trail_stop.momentum_score >= 5)):
            trail_stop.trail_mode = "aggressive"
            trail_stop.aggressive_triggered = True
            
            if trail_stop.momentum_score >= 5:
                trail_pips = min(15, trail_stop.current_profit_pips * 0.15)
            else:
                trail_pips = min(20, trail_stop.current_profit_pips * 0.2)
            
            trail_distance = trail_pips * pip_value
            new_sl = current_price - trail_distance if is_buy else current_price + trail_distance
            trail_stop.last_sl = new_sl
            return new_sl, f"Aggressive trail ({trail_pips:.1f} pips)"
        
        # Standard trailing
        if trail_stop.current_profit_atr >= 2.0:
            trail_stop.trail_mode = "standard"
            if not trail_stop.trailing_triggered:
                trail_stop.trailing_triggered = True
            
            atr_trail = 1.0 * trail_stop.atr
            pip_trail = 25 * pip_value
            trail_distance = min(atr_trail, pip_trail)
            
            new_sl = current_price - trail_distance if is_buy else current_price + trail_distance
            trail_stop.last_sl = new_sl
            return new_sl, f"Hybrid trail ({trail_stop.current_profit_atr:.1f} ATR)"
        
        # Break-even plus buffer
        if trail_stop.current_profit_atr >= 1.5 and not trail_stop.be_triggered:
            trail_stop.trail_mode = "breakeven"
            trail_stop.be_triggered = True
            
            buffer = max(10 * pip_value, 0.2 * trail_stop.atr)
            be_sl = trail_stop.entry_price + (buffer if is_buy else -buffer)
            trail_stop.last_sl = be_sl
            return be_sl, f"Break-even+ ({trail_stop.current_profit_atr:.1f} ATR)"
        
        # Initial stop
        if trail_stop.last_sl is None:
            trail_stop.trail_mode = "initial"
            atr_stop = 1.5 * trail_stop.atr
            pip_stop = 30 * pip_value
            stop_distance = max(atr_stop, pip_stop)
            initial_sl = trail_stop.entry_price - (stop_distance if is_buy else -stop_distance)
            trail_stop.last_sl = initial_sl
            return initial_sl, f"Initial ({stop_distance / pip_value:.0f} pips)"
        
        return trail_stop.last_sl, "Maintain SL"


# ============================================================================
# TRADE EXECUTOR
# ============================================================================

class TradeExecutor:
    """Handles trade execution"""
    
    def __init__(self, magic_number: int, comment: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.magic_number = magic_number
        self.comment = comment
    
    def get_filling_mode(self, symbol: str) -> int:
        """Determine correct filling mode"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_FOK
        
        filling_mode = symbol_info.filling_mode
        
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_FOK
    
    def normalize_price(self, price: float, tick_size: float) -> float:
        """Normalize price to tick size"""
        if tick_size <= 0:
            return float(price)
        normalized = round(price / tick_size) * tick_size
        return float(normalized)
    
    def get_min_stop_distance(self, symbol: str) -> float:
        """Get minimum stop distance for symbol"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0001
        
        point = symbol_info.point if symbol_info.point else 1e-5
        
        for attr in ["trade_stops_level", "stops_level", "stop_level"]:
            val = getattr(symbol_info, attr, None)
            if val is not None and val > 0:
                if val < 1000:
                    return float(val) * point
                return float(val)
        
        return max(10 * point, symbol_info.trade_tick_size * 2)
    
    def open_trade(self, symbol: str, direction: str, lot_size: float,
                   sl: float, tp: float) -> Optional[int]:
        """Open trade"""
        if not mt5.symbol_select(symbol, True):
            self.logger.error(f"Symbol {symbol} not available")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"No tick for {symbol}")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        tick_size = symbol_info.trade_tick_size if symbol_info.trade_tick_size else symbol_info.point
        
        # Normalize prices
        sl = self.normalize_price(sl, tick_size)
        tp = self.normalize_price(tp, tick_size)
        
        # Get price and order type
        if direction == 'buy':
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {getattr(result, 'comment', 'Unknown error')}")
            return None
        
        ticket = result.order
        self.logger.info(
            f"{Fore.GREEN}[TRADE OPENED] {direction.upper()} {symbol} | "
            f"Ticket: {ticket} | Price: {price:.5f} | "
            f"SL: {sl:.5f} | TP: {tp:.5f} | Lot: {lot_size}{Style.RESET_ALL}"
        )
        
        return ticket
    
    def close_position(self, ticket: int, reason: str = "") -> bool:
        """Close position by ticket"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return False
        
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"Close: {reason}",
            "type_filling": self.get_filling_mode(pos.symbol),
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(
                f"{Fore.YELLOW}[POSITION CLOSED] {pos.symbol} ticket {ticket} | "
                f"Reason: {reason}{Style.RESET_ALL}"
            )
            return True
        else:
            self.logger.error(f"Failed to close position {ticket}")
            return False
    
    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify stop loss and take profit"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)
        if not symbol_info:
            return False
        
        tick_size = symbol_info.trade_tick_size if symbol_info.trade_tick_size else symbol_info.point
        
        new_sl = self.normalize_price(new_sl, tick_size)
        new_tp = self.normalize_price(new_tp, tick_size)
        
        if abs(new_sl - pos.sl) < tick_size and abs(new_tp - pos.tp) < tick_size:
            return True
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": float(new_sl),
            "tp": float(new_tp),
            "magic": self.magic_number,
            "comment": self.comment
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.debug(f"Modified SL/TP for ticket {ticket}")
            return True
        
        return False


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class RSIMLTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Initialize components
        self.data_handler = DataHandler(timeframe=config['timeframe'])
        self.signal_generator = SignalGenerator(
            rsi_period=config['rsi_period'],
            ma_short=config['ma_short'],
            ma_long=config['ma_long'],
            atr_period=config['atr_period']
        )
        self.ml_manager = MLModelManager(model_dir=config['model_dir'])
        self.risk_manager = RiskManager(
            risk_per_trade_pct=config['risk_per_trade_pct'],
            max_positions_per_symbol=config['max_positions_per_symbol'],
            max_total_positions=config['max_total_positions'],
            max_daily_loss_pct=config['max_daily_loss_pct'],
            max_total_risk_pct=config['max_total_risk_pct'],
            force_min_lot=True
        )
        self.position_tracker = PositionTracker(magic_number=config['magic_number'])
        self.trailing_manager = TrailingStopManager(magic_number=config['magic_number'])
        self.trade_executor = TradeExecutor(
            magic_number=config['magic_number'],
            comment=config['comment']
        )
        self.csv_logger = CSVTradeLogger(log_dir=config.get('log_dir', 'logs'))
        
        self.symbols = config['symbols']
        self.ml_threshold = config.get('ml_threshold', 0.60)
        self.update_interval = config.get('update_interval', 60)
        self.trading_hours = config.get('trading_hours', [(9, 15), (15, 18)])
        
        # SL/TP configuration
        self.sl_atr_mult = config.get('sl_atr_multiplier', 1.2)
        self.tp_atr_mult = config.get('tp_atr_multiplier', 2.4)
        self.min_rr_ratio = config.get('min_risk_reward_ratio', 2.0)
        
        self.is_running = False
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        mt5_path = self.config.get('mt5_path')
        
        if mt5_path:
            if not mt5.initialize(path=mt5_path):
                self.logger.error(f"MT5 initialization failed with path: {mt5_path}")
                return False
        else:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed (auto-detect)")
                return False
        
        self.logger.info(f"{Fore.GREEN}MT5 initialized successfully{Style.RESET_ALL}")
        
        # Get account info
        acc_info = mt5.account_info()
        if acc_info:
            self.logger.info(f"Account: {acc_info.login} | Balance: ${acc_info.balance:.2f}")
            self.risk_manager.update_balance()
            self.risk_manager.reset_daily()
        else:
            self.logger.error("Failed to get account info")
            return False
        
        return True
    
    def shutdown_mt5(self):
        """Shutdown MT5"""
        mt5.shutdown()
        self.logger.info("MT5 connection closed")
    
    def is_trading_hours(self) -> bool:
        """Check if within trading hours"""
        current_hour = datetime.now().hour
        for start, end in self.trading_hours:
            if start <= current_hour < end:
                return True
        return False
    
    def rebuild_trailing_stops(self):
        """Rebuild trailing stops on restart"""
        self.logger.info("Rebuilding trailing stops from existing positions...")
        
        positions = mt5.positions_get(magic=self.config['magic_number'])
        if not positions:
            self.logger.info("No existing positions to rebuild")
            return
        
        rebuilt = 0
        for pos in positions:
            try:
                df = self.data_handler.fetch_ohlcv(pos.symbol, 200)
                if df is None:
                    continue
                
                df_feat = TechnicalIndicators.build_features(
                    df, self.config['rsi_period'], self.config['ma_short'],
                    self.config['ma_long'], self.config['atr_period']
                )
                
                atr_val = df_feat['atr'].iloc[-1]
                if pd.isna(atr_val):
                    continue
                
                self.trailing_manager.add_position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    entry_price=pos.price_open,
                    entry_time=datetime.fromtimestamp(pos.time),
                    pos_type=pos.type,
                    atr=float(atr_val)
                )
                
                rebuilt += 1
                hours_ago = (datetime.now() - datetime.fromtimestamp(pos.time)).total_seconds() / 3600
                self.logger.info(f"✓ Rebuilt: {pos.symbol} ticket {pos.ticket} ({hours_ago:.1f}h ago)")
                
            except Exception as e:
                self.logger.error(f"Error rebuilding position {pos.ticket}: {e}")
        
        self.logger.info(f"Successfully rebuilt {rebuilt} trailing stops")
    
    def process_symbol(self, symbol: str):
        """Process single symbol for trading signals"""
        try:
            self.logger.debug(f"--- Processing {symbol} ---")
            
            # Fetch data
            df = self.data_handler.fetch_ohlcv(symbol, 500)
            if df is None:
                self.logger.debug(f"Failed to fetch data for {symbol}")
                return
            
            # Build features
            df_feat = TechnicalIndicators.build_features(
                df, self.config['rsi_period'], self.config['ma_short'],
                self.config['ma_long'], self.config['atr_period']
            )
            
            # Generate rule-based signal
            rule_signal = self.signal_generator.generate_rule_signal(df)
            
            if rule_signal is None:
                self.logger.debug(f"{symbol}: No rule-based signal")
                return
            
            # Get last features
            last_features = df_feat.iloc[-1]
            
            # Apply ML filter if enabled
            if self.config['use_ml']:
                ml_result = self.ml_manager.predict_signal(last_features, rule_signal)
                
                if ml_result is None:
                    self.csv_logger.log_signal(
                        symbol, rule_signal.upper(), 0.0,
                        last_features['rsi'], last_features['ema_diff'],
                        last_features['atr'], False, "No ML model"
                    )
                    self.logger.debug(f"{symbol}: ML prediction returned None")
                    return
                
                direction, confidence = ml_result
            else:
                # When ML is disabled, use rule-based signal directly
                direction = rule_signal
                confidence = 1.0
                self.logger.info(f"{Fore.CYAN}ML disabled - using rule signal with confidence=1.0{Style.RESET_ALL}")
            
            # Check confidence threshold
            if confidence < self.ml_threshold:
                self.csv_logger.log_signal(
                    symbol, direction.upper(), confidence,
                    last_features['rsi'], last_features['ema_diff'],
                    last_features['atr'], False,
                    f"ML confidence too low: {confidence:.3f}"
                )
                self.logger.info(
                    f"{symbol} {direction.upper()} rejected by ML (confidence={confidence:.3f} < {self.ml_threshold:.2f})"
                )
                return
            
            # Signal approved
            self.logger.info(
                f"{Fore.GREEN}✅ [SIGNAL APPROVED] {symbol} {direction.upper()} | "
                f"ML confidence: {confidence:.3f}{Style.RESET_ALL}"
            )
            
            # Check if can open trade
            position_counts = self.position_tracker.get_count_by_symbol()
            can_trade, reason = self.risk_manager.can_open_trade(symbol, position_counts)
            
            if not can_trade:
                self.csv_logger.log_signal(
                    symbol, direction.upper(), confidence,
                    last_features['rsi'], last_features['ema_diff'],
                    last_features['atr'], False, reason
                )
                self.logger.warning(f"Cannot open trade for {symbol}: {reason}")
                return
            
            # Check trading hours
            if not self.is_trading_hours():
                current_hour = datetime.now().hour
                self.csv_logger.log_signal(
                    symbol, direction.upper(), confidence,
                    last_features['rsi'], last_features['ema_diff'],
                    last_features['atr'], False, f"Outside trading hours (current: {current_hour}:00)"
                )
                self.logger.info(f"Outside trading hours - current: {current_hour}:00, allowed: {self.trading_hours}")
                return
            
            # Execute trade
            self.logger.info(f"{Fore.YELLOW}>>> EXECUTING TRADE FOR {symbol} <<<{Style.RESET_ALL}")
            self.execute_trade(symbol, direction, last_features, confidence)
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    def execute_trade(self, symbol: str, direction: str, 
                      features: pd.Series, ml_confidence: float):
        """Execute trade with FIXED risk:reward ratios"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Cannot get tick for {symbol}")
                return
            
            price = tick.ask if direction == 'buy' else tick.bid
            atr = features['atr']
            
            if pd.isna(atr) or atr <= 0:
                self.logger.error(f"Invalid ATR: {atr}")
                return
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return
            
            # ==================================================================
            # FIXED SL/TP CALCULATION - PROPER 1:2 RISK:REWARD
            # ==================================================================
            
            # Calculate SL and TP distances
            sl_distance_atr = self.sl_atr_mult
            tp_distance_atr = self.tp_atr_mult
            
            # Verify risk:reward ratio
            actual_rr = tp_distance_atr / sl_distance_atr
            if actual_rr < self.min_rr_ratio:
                self.logger.warning(
                    f"Risk:Reward ratio {actual_rr:.2f} below minimum {self.min_rr_ratio:.2f}. "
                    f"Adjusting TP to meet minimum."
                )
                tp_distance_atr = sl_distance_atr * self.min_rr_ratio
            
            # Calculate actual SL and TP prices
            if direction == 'buy':
                sl = price - (sl_distance_atr * atr)
                tp = price + (tp_distance_atr * atr)
            else:  # sell
                sl = price + (sl_distance_atr * atr)
                tp = price - (tp_distance_atr * atr)
            
            # Normalize prices to symbol's tick size
            tick_size = symbol_info.trade_tick_size if symbol_info.trade_tick_size else symbol_info.point
            sl = self.trade_executor.normalize_price(sl, tick_size)
            tp = self.trade_executor.normalize_price(tp, tick_size)
            
            # Calculate distances for verification
            sl_distance = abs(price - sl)
            tp_distance = abs(tp - price)
            verified_rr = tp_distance / sl_distance if sl_distance > 0 else 0
            
            # ==================================================================
            # ENHANCED TRADE LOGGING
            # ==================================================================
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"{Fore.YELLOW}📈 TRADE SETUP FOR {symbol} - {direction.upper()}{Style.RESET_ALL}")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"💰 Entry Price: {price:.5f}")
            self.logger.info(f"📊 ATR: {atr:.5f}")
            self.logger.info(f"\n🛡️  STOP LOSS:")
            self.logger.info(f"   SL Price: {sl:.5f}")
            self.logger.info(f"   SL Distance: {sl_distance:.5f} ({sl_distance_atr:.1f} ATR)")
            self.logger.info(f"   SL Pips: {sl_distance / symbol_info.point:.1f}")
            self.logger.info(f"\n🎯 TAKE PROFIT:")
            self.logger.info(f"   TP Price: {tp:.5f}")
            self.logger.info(f"   TP Distance: {tp_distance:.5f} ({tp_distance_atr:.1f} ATR)")
            self.logger.info(f"   TP Pips: {tp_distance / symbol_info.point:.1f}")
            self.logger.info(f"\n⚖️  RISK:REWARD RATIO: 1:{verified_rr:.2f}")
            
            if verified_rr >= 2.0:
                self.logger.info(f"{Fore.GREEN}   ✅ Excellent risk:reward!{Style.RESET_ALL}")
            elif verified_rr >= 1.5:
                self.logger.info(f"{Fore.YELLOW}   ⚠️  Acceptable risk:reward{Style.RESET_ALL}")
            else:
                self.logger.warning(f"{Fore.RED}   ❌ Poor risk:reward - consider skipping{Style.RESET_ALL}")
            
            self.logger.info(f"{'='*70}\n")
            
            # ==================================================================
            # CALCULATE LOT SIZE
            # ==================================================================
            
            ml_multiplier = self.ml_manager.predict_lot_multiplier(features)
            lot_size = self.risk_manager.calculate_lot_size(symbol, sl_distance, ml_multiplier)
            
            if lot_size <= 0:
                self.logger.error(f"Invalid lot size: {lot_size}")
                return
            
            # ==================================================================
            # EXECUTE TRADE
            # ==================================================================
            
            ticket = self.trade_executor.open_trade(symbol, direction, lot_size, sl, tp)
            
            if ticket:
                # Log trade
                self.csv_logger.log_trade_open(
                    symbol, direction, ticket, price, lot_size, sl, tp, atr
                )
                self.csv_logger.log_signal(
                    symbol, direction.upper(), ml_confidence,
                    features['rsi'], features['ema_diff'], atr,
                    True, ""
                )
                
                # Add to trailing stop manager
                self.trailing_manager.add_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=price,
                    entry_time=datetime.now(),
                    pos_type=mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                    atr=atr
                )
                
                # Update stats
                self.risk_manager.trades_today += 1
                
                self.logger.info(
                    f"{Fore.GREEN}✅ TRADE EXECUTED: {symbol} {direction.upper()} | "
                    f"Ticket: {ticket} | Lot: {lot_size} | R:R 1:{verified_rr:.2f}{Style.RESET_ALL}\n"
                )
            else:
                self.logger.error(f"Failed to execute trade for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)
    
    def manage_trailing_stops(self):
        """Manage trailing stops for all positions"""
        try:
            positions = mt5.positions_get(magic=self.config['magic_number'])
            if not positions:
                return
            
            for pos in positions:
                ticket = pos.ticket
                symbol = pos.symbol
                
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                
                current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                
                new_sl, reason = self.trailing_manager.calculate_stop(ticket, current_price)
                
                if new_sl == "CLOSE":
                    if self.trade_executor.close_position(ticket, reason):
                        self.csv_logger.log_trade_close(ticket, current_price, pos.profit, reason)
                        self.trailing_manager.remove_position(ticket)
                        
                        # Update win/loss stats
                        if ticket not in self.risk_manager.closed_tickets:
                            if pos.profit > 0:
                                self.risk_manager.wins_today += 1
                            else:
                                self.risk_manager.losses_today += 1
                            self.risk_manager.closed_tickets.add(ticket)
                    continue
                
                # Modify stop if changed
                if new_sl and new_sl != pos.sl:
                    self.trade_executor.modify_sl_tp(ticket, new_sl, pos.tp)
                    self.logger.info(
                        f"{Fore.CYAN}[SL MODIFIED] {symbol} ticket {ticket} | "
                        f"New SL: {new_sl:.5f} | {reason}{Style.RESET_ALL}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error managing trailing stops: {e}", exc_info=True)
    
    def print_status(self):
        """Print bot status"""
        acc_info = mt5.account_info()
        if not acc_info:
            return
        
        daily_pnl = acc_info.balance - self.risk_manager.starting_balance
        daily_pnl_pct = (daily_pnl / self.risk_manager.starting_balance * 100) if self.risk_manager.starting_balance > 0 else 0
        
        win_rate = (self.risk_manager.wins_today / self.risk_manager.trades_today * 100) if self.risk_manager.trades_today > 0 else 0
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"💰 Balance: ${acc_info.balance:.2f} | Equity: ${acc_info.equity:.2f}")
        print(f"📊 Daily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%)")
        print(f"📈 Open Positions: {self.position_tracker.get_count()}/{self.config['max_total_positions']}")
        print(f"📋 Trades: {self.risk_manager.trades_today} | W:{self.risk_manager.wins_today} L:{self.risk_manager.losses_today}")
        if self.risk_manager.trades_today > 0:
            print(f"📊 Win Rate: {win_rate:.1f}%")
        print(f"🚦 Trading Status: {'SUSPENDED' if self.risk_manager.trading_suspended else 'ACTIVE'}")
        
        # Show current hour and trading hours
        current_hour = datetime.now().hour
        in_hours = self.is_trading_hours()
        print(f"🕐 Current Hour: {current_hour}:00 | Trading Hours: {self.trading_hours} | Active: {in_hours}")
        print(f"⚖️  SL/TP Strategy: {self.sl_atr_mult} ATR / {self.tp_atr_mult} ATR (1:{self.tp_atr_mult/self.sl_atr_mult:.1f} R:R)")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    def run(self, mode: str = 'live'):
        """Main run loop"""
        self.logger.info(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}RSI-MA-ATR ML TRADING BOT STARTED{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}Mode: {mode.upper()}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        
        if mode == 'live':
            if not self.initialize_mt5():
                return
            
            # Load ML models
            if self.config['use_ml']:
                if not self.ml_manager.load_models():
                    self.logger.warning("No ML models loaded - disabling ML")
                    self.config['use_ml'] = False
            else:
                self.logger.info(f"{Fore.YELLOW}ML DISABLED - Using rule-based signals only{Style.RESET_ALL}")
            
            # Rebuild trailing stops
            self.rebuild_trailing_stops()
            
            # Update position tracker
            self.position_tracker.update()
            
            self.logger.info(f"Symbols: {self.symbols}")
            self.logger.info(f"Magic Number: {self.config['magic_number']}")
            self.logger.info(f"Risk per trade: {self.config['risk_per_trade_pct']:.1f}%")
            self.logger.info(f"ML Threshold: {self.ml_threshold:.2f}")
            self.logger.info(f"Trading Hours: {self.trading_hours}")
            self.logger.info(f"SL/TP: {self.sl_atr_mult} ATR / {self.tp_atr_mult} ATR")
            self.logger.info(f"Risk:Reward Target: 1:{self.tp_atr_mult/self.sl_atr_mult:.1f}")
            self.logger.info(f"Current Hour: {datetime.now().hour}:00")
            
            self.is_running = True
            cycle = 0
            status_counter = 0
            
            try:
                while self.is_running:
                    cycle += 1
                    
                    self.logger.info(f"\n{'='*70}")
                    self.logger.info(f"🔄 Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    self.logger.info(f"{'='*70}")
                    
                    # Update position tracker
                    closed_tickets = self.position_tracker.update()
                    
                    # Handle closed positions
                    for ticket in closed_tickets:
                        if ticket not in self.risk_manager.closed_tickets:
                            # Get profit from history
                            deals = mt5.history_deals_get(ticket=ticket)
                            if deals:
                                profit = sum(deal.profit for deal in deals)
                                if profit > 0:
                                    self.risk_manager.wins_today += 1
                                else:
                                    self.risk_manager.losses_today += 1
                                self.risk_manager.closed_tickets.add(ticket)
                            
                            self.trailing_manager.remove_position(ticket)
                    
                    # Check daily limits
                    if not self.risk_manager.check_daily_limit():
                        self.logger.warning("Daily loss limit exceeded. Sleeping...")
                        time.sleep(3600)
                        continue
                    
                    # Process each symbol
                    for symbol in self.symbols:
                        self.logger.info(f"🔍 Analyzing {symbol}...")
                        self.process_symbol(symbol)
                    
                    # Manage trailing stops
                    self.logger.info("🎯 Managing trailing stops...")
                    self.manage_trailing_stops()
                    
                    # Print status periodically
                    status_counter += 1
                    if status_counter >= 5:  # Every 5 cycles
                        self.print_status()
                        status_counter = 0
                    
                    self.logger.info(f"{'='*70}")
                    self.logger.info(f"✅ Cycle #{cycle} complete. Sleeping {self.update_interval}s...")
                    self.logger.info(f"{'='*70}\n")
                    
                    time.sleep(self.update_interval)
                    
            except KeyboardInterrupt:
                self.logger.info(f"{Fore.YELLOW}Bot stopped by user{Style.RESET_ALL}")
            except Exception as e:
                self.logger.error(f"Critical error: {e}", exc_info=True)
            finally:
                # Log final performance
                acc_info = mt5.account_info()
                if acc_info:
                    self.csv_logger.log_daily_performance(
                        self.risk_manager.starting_balance,
                        acc_info.balance,
                        self.risk_manager.trades_today,
                        self.risk_manager.wins_today,
                        self.risk_manager.losses_today
                    )
                
                self.shutdown_mt5()
                self.logger.info(f"{Fore.RED}Bot shutdown complete{Style.RESET_ALL}")


# ============================================================================
# CONFIGURATION & MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSI-MA-ATR ML Trading Bot')
    parser.add_argument('--mode', choices=['live'],
                       default='live', help='Operation mode')
    parser.add_argument('--symbols', nargs='*',
                       default=['XAUUSD', 'EURUSD', 'GBPUSD', 'XAGUSD','XAUEUR','AUDUSD'],
                       help='Trading symbols')
    parser.add_argument('--use-ml', action='store_true',
                       help='Enable ML predictions (default: disabled)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    config = {
        # MT5 Settings
        'mt5_path': r"C:\Program Files\MetaTrader 5\terminal64.exe",
        'magic_number': 234000,
        'comment': "RSI-MA-ATR-ML-Bot-v2.0.2",
        
        # Trading Mode
        'use_ml': args.use_ml,
        
        # Symbols and Timeframe
        'symbols': args.symbols,
        'timeframe': mt5.TIMEFRAME_M1,
        
        # Technical Indicator Settings
        'rsi_period': 14,
        'ma_short': 20,
        'ma_long': 50,
        'atr_period': 14,
        
        # Risk Management - UPDATED FOR SMALL ACCOUNT
        'risk_per_trade_pct': 1.0,  # 2% risk per trade
        'max_positions_per_symbol': 2,
        'max_total_positions': 10,
        'max_daily_loss_pct': 5.0,  # 5% daily loss limit
        'max_total_risk_pct': 5.0,
        
        # SL/TP Configuration - FIXED FOR PROPER 1:2 R:R
        'sl_atr_multiplier': 1,  # 1.2 ATR stop loss
        'tp_atr_multiplier': 10,  # 2.4 ATR take profit (1:2 R:R)
        'min_risk_reward_ratio': 2.0,  # Minimum 1:2 risk:reward
        
        # ML Settings
        'model_dir': 'models',
        'ml_threshold': 0.60,
        
        # Trading Hours (24-hour format)
        'trading_hours': [(7, 24)],  # 8 AM to 12 AM
        
        # Bot Settings
        'update_interval': 10,
        'log_dir': 'logs',
    }
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🤖 RSI-MA-ATR ML TRADING BOT v2.0.2 (COMPLETE FIX)")
    logger.info(f"{'='*70}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Symbols: {config['symbols']}")
    logger.info(f"ML Enabled: {config['use_ml']}")
    logger.info(f"\n✅ FIXES:")
    logger.info(f"  • Fixed lot size calculation (proper MT5 formula)")
    logger.info(f"  • Fixed SL/TP ratios (1:{config['tp_atr_multiplier']/config['sl_atr_multiplier']:.1f} R:R)")
    logger.info(f"  • Complete RiskManager with all methods")
    logger.info(f"  • Enhanced trade logging")
    logger.info(f"  • Detailed risk:reward verification")
    logger.info(f"{'='*70}\n")
    
    # Execute
    bot = RSIMLTradingBot(config)
    bot.run(mode='live')


if __name__ == '__main__':
    main()