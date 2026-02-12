"""
MetaTrader 5 FVG-BOS Smart Money Concept Trading Bot
SESSION-BASED COMPLETE IMPLEMENTATION
Version: 1.2.0 - Fixed FVG Detection with Historical Scanning

Key Features:
- Automatic session detection (Asian/London/NY)
- Session-specific symbol filtering
- Enhanced FVG detection with historical scanning
- Correlation management per session
- Dynamic risk adjustment by volatility
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
import time as time_module
from colorama import Fore, Style, init
import sys
from collections import defaultdict

# Initialize colorama
init(autoreset=True)

# ============================================================================
# SESSION CONFIGURATION
# ============================================================================

SESSION_CONFIG = {
    'ASIAN': {
        'symbols': ['USDJPY', 'AUDJPY', 'NZDJPY', 'AUDUSD', 'NZDUSD'],
        'hours': [(3, 7)],
        'risk_multiplier': 0.85,
        'max_spread_pips': 2.5,
    },
    'LONDON': {
        'symbols': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY', 'GBPJPY'],
        'hours': [(8, 19)],
        'risk_multiplier': 1.0,
        'max_spread_pips': 2.0,
    },
    'NEW_YORK': {
        'symbols': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY'],
        'hours': [(15, 24), (0, 3)],
        'risk_multiplier': 1.0,
        'max_spread_pips': 2.0,
    },
}

CORRELATION_GROUPS = {
    'USD_BASKET': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
    'JPY_BASKET': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY'],
    'EUR_BASKET': ['EURUSD', 'EURJPY', 'EURGBP'],
    'GBP_BASKET': ['GBPUSD', 'GBPJPY', 'EURGBP'],
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FVG:
    """Fair Value Gap structure"""
    symbol: str
    timeframe: str
    direction: str
    top: float
    bottom: float
    midpoint: float
    timestamp: datetime
    candle_index: int
    is_active: bool = True
    touched: bool = False
    
    def __repr__(self):
        return f"FVG({self.direction}, {self.bottom:.5f}-{self.top:.5f}, active={self.is_active})"


@dataclass
class BOS:
    """Break of Structure detection"""
    symbol: str
    timeframe: str
    direction: str
    break_price: float
    swing_price: float
    timestamp: datetime
    candle_index: int
    
    def __repr__(self):
        return f"BOS({self.direction}, price={self.break_price:.5f})"


@dataclass
class SwingPoint:
    """Swing high/low structure"""
    price: float
    index: int
    timestamp: datetime
    type: str


@dataclass
class TradeRecord:
    """Trade execution record"""
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    entry_time: datetime
    session: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = 'open'
    initial_sl: float = field(init=False)
    trailing_activated: bool = False
    last_trailing_update: Optional[datetime] = None
    
    def __post_init__(self):
        self.initial_sl = self.sl


# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manages trading sessions and symbol selection"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_session = None
        self.session_start_time = None
        self.session_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0})
    
    def get_current_session(self) -> Optional[str]:
        """Determine current trading session based on local time"""
        now = datetime.now()
        current_hour = now.hour
        
        for session_name, config in SESSION_CONFIG.items():
            for start_hour, end_hour in config['hours']:
                if start_hour <= current_hour < end_hour:
                    if self.current_session != session_name:
                        self.current_session = session_name
                        self.session_start_time = now
                        self.logger.info("="*60)
                        self.logger.info(f"  SESSION CHANGED: {session_name}")
                        self.logger.info(f"  Local Time: {now.strftime('%H:%M')}")
                        self.logger.info("="*60)
                    return session_name
        
        if self.current_session is not None:
            self.logger.info(f"{Fore.YELLOW}[NO ACTIVE SESSION] Current hour: {current_hour}:00{Style.RESET_ALL}")
            self.current_session = None
        
        return None
    
    def get_active_symbols(self) -> List[str]:
        """Get symbols for current session"""
        session = self.get_current_session()
        if session is None:
            return []
        return SESSION_CONFIG[session]['symbols']
    
    def get_risk_multiplier(self) -> float:
        """Get risk multiplier for current session"""
        session = self.get_current_session()
        if session is None:
            return 0.5
        return SESSION_CONFIG[session]['risk_multiplier']
    
    def get_max_spread(self) -> float:
        """Get maximum allowed spread for current session"""
        session = self.get_current_session()
        if session is None:
            return 1.5
        return SESSION_CONFIG[session]['max_spread_pips']
    
    def is_trading_active(self) -> bool:
        """Check if trading is active"""
        return self.get_current_session() is not None
    
    def record_trade(self, won: bool):
        """Record trade statistics per session"""
        if self.current_session:
            self.session_stats[self.current_session]['trades'] += 1
            if won:
                self.session_stats[self.current_session]['wins'] += 1
            else:
                self.session_stats[self.current_session]['losses'] += 1
    
    def get_session_stats(self) -> Dict:
        """Get statistics for all sessions"""
        stats = {}
        for session, data in self.session_stats.items():
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            stats[session] = {
                'trades': data['trades'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': win_rate
            }
        return stats


# ============================================================================
# SPREAD MONITOR
# ============================================================================

class SpreadMonitor:
    """Monitor and validate spread conditions"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.spread_history: Dict[str, List[float]] = defaultdict(list)
    
    def get_current_spread_pips(self, symbol: str) -> Optional[float]:
        """Get current spread in pips"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            
            if not tick or not symbol_info:
                return None
            
            spread = tick.ask - tick.bid
            point = symbol_info.point if symbol_info.point else 1e-5
            spread_pips = spread / point / 10
            
            self.spread_history[symbol].append(spread_pips)
            if len(self.spread_history[symbol]) > 100:
                self.spread_history[symbol].pop(0)
            
            return spread_pips
            
        except Exception as e:
            self.logger.error(f"Error getting spread for {symbol}: {e}")
            return None
    
    def is_spread_acceptable(self, symbol: str, max_spread_pips: float) -> Tuple[bool, str]:
        """Check if spread is acceptable for trading"""
        spread_pips = self.get_current_spread_pips(symbol)
        
        if spread_pips is None:
            return False, "Failed to get spread"
        
        if spread_pips > max_spread_pips:
            return False, f"Spread too high: {spread_pips:.1f} pips (max: {max_spread_pips:.1f})"
        
        return True, f"Spread OK: {spread_pips:.1f} pips"


# ============================================================================
# CORRELATION MANAGER
# ============================================================================

class CorrelationManager:
    """Manage correlated pairs to avoid over-exposure"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_correlation_group(self, symbol: str) -> Optional[str]:
        """Find which correlation group a symbol belongs to"""
        for group_name, symbols in CORRELATION_GROUPS.items():
            if symbol in symbols:
                return group_name
        return None
    
    def check_correlation_conflict(self, symbol: str, direction: str, 
                                   active_trades: Dict[int, TradeRecord]) -> Tuple[bool, str]:
        """Check if opening this position would create correlation conflict"""
        group = self.get_correlation_group(symbol)
        if not group:
            return True, "No correlation group"
        
        group_symbols = CORRELATION_GROUPS[group]
        
        for ticket, trade in active_trades.items():
            if trade.symbol in group_symbols and trade.symbol != symbol:
                if trade.direction == direction:
                    return False, f"Correlation risk: {trade.symbol} already {direction}"
        
        return True, "No correlation conflict"
    
    def get_exposure_by_currency(self, active_trades: Dict[int, TradeRecord]) -> Dict[str, int]:
        """Calculate net exposure by currency"""
        exposure = defaultdict(int)
        
        for ticket, trade in active_trades.items():
            base = trade.symbol[:3]
            quote = trade.symbol[3:6]
            
            if trade.direction == 'buy':
                exposure[base] += 1
                exposure[quote] -= 1
            else:
                exposure[base] -= 1
                exposure[quote] += 1
        
        return dict(exposure)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding"""
    
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


def setup_logging():
    """Setup logging configuration"""
    log_filename = f"fvg_bos_session_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# DATA HANDLER
# ============================================================================

class DataHandler:
    """Handles data fetching and management from MT5"""
    
    def __init__(self, timeframes: Dict[str, int]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.timeframes = timeframes
        self.data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        
    def fetch_ohlc(self, symbol: str, timeframe: int, bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLC data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def update_all_data(self, active_symbols: List[str]) -> bool:
        """Update data cache for active symbols only"""
        try:
            for symbol in active_symbols:
                for tf_name, tf_value in self.timeframes.items():
                    df = self.fetch_ohlc(symbol, tf_value)
                    if df is not None:
                        self.data_cache[(symbol, tf_name)] = df
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating data cache: {e}")
            return False
    
    def get_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached data for symbol and timeframe"""
        return self.data_cache.get((symbol, timeframe))


# ============================================================================
# STRUCTURE ANALYZER
# ============================================================================

class StructureAnalyzer:
    """Analyzes market structure and detects Break of Structure (BOS)"""
    
    def __init__(self, swing_lookback: int = 10, min_bos_distance: int = 5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.swing_lookback = swing_lookback
        self.min_bos_distance = min_bos_distance
        self.swing_highs: Dict[Tuple[str, str], List[SwingPoint]] = {}
        self.swing_lows: Dict[Tuple[str, str], List[SwingPoint]] = {}
        self.last_bos: Dict[Tuple[str, str], BOS] = {}
        
    def detect_swing_points(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect swing highs and lows"""
        highs = []
        lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            window_high = df['high'].iloc[i-self.swing_lookback:i+self.swing_lookback+1]
            if df['high'].iloc[i] == window_high.max():
                highs.append(SwingPoint(
                    price=df['high'].iloc[i],
                    index=i,
                    timestamp=df.index[i],
                    type='high'
                ))
            
            window_low = df['low'].iloc[i-self.swing_lookback:i+self.swing_lookback+1]
            if df['low'].iloc[i] == window_low.min():
                lows.append(SwingPoint(
                    price=df['low'].iloc[i],
                    index=i,
                    timestamp=df.index[i],
                    type='low'
                ))
        
        key = (symbol, timeframe)
        self.swing_highs[key] = highs
        self.swing_lows[key] = lows
        
        return highs, lows
    
    def detect_bos(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[BOS]:
        """Detect Break of Structure"""
        key = (symbol, timeframe)
        
        highs = self.swing_highs.get(key, [])
        lows = self.swing_lows.get(key, [])
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        current_price = df['close'].iloc[-1]
        current_index = len(df) - 1
        current_time = df.index[-1]
        
        # Check for Bullish BOS
        for swing_high in reversed(highs[-5:]):
            if current_price > swing_high.price:
                if key in self.last_bos:
                    last_bos = self.last_bos[key]
                    bars_since_last = current_index - last_bos.candle_index
                    if last_bos.direction == 'bearish' and bars_since_last < self.min_bos_distance:
                        continue
                
                bos = BOS(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction='bullish',
                    break_price=current_price,
                    swing_price=swing_high.price,
                    timestamp=current_time,
                    candle_index=current_index
                )
                self.last_bos[key] = bos
                self.logger.info(f"{Fore.GREEN}[BOS DETECTED] {symbol} {timeframe} - BULLISH @ {current_price:.5f}{Style.RESET_ALL}")
                return bos
        
        # Check for Bearish BOS
        for swing_low in reversed(lows[-5:]):
            if current_price < swing_low.price:
                if key in self.last_bos:
                    last_bos = self.last_bos[key]
                    bars_since_last = current_index - last_bos.candle_index
                    if last_bos.direction == 'bullish' and bars_since_last < self.min_bos_distance:
                        continue
                
                bos = BOS(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction='bearish',
                    break_price=current_price,
                    swing_price=swing_low.price,
                    timestamp=current_time,
                    candle_index=current_index
                )
                self.last_bos[key] = bos
                self.logger.info(f"{Fore.RED}[BOS DETECTED] {symbol} {timeframe} - BEARISH @ {current_price:.5f}{Style.RESET_ALL}")
                return bos
        
        return None
    
    def get_trend_direction(self, symbol: str, timeframe: str) -> Optional[str]:
        """Get trend direction based on last BOS"""
        key = (symbol, timeframe)
        if key in self.last_bos:
            return self.last_bos[key].direction
        return None


# ============================================================================
# FVG DETECTOR - FIXED VERSION
# ============================================================================

class FVGDetector:
    """Detects and manages Fair Value Gaps"""
    
    def __init__(self, min_fvg_size_atr_multiplier: float = 0.2):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_fvg_size_multiplier = min_fvg_size_atr_multiplier
        self.active_fvgs: Dict[Tuple[str, str], List[FVG]] = {}
        self.historical_scan_done: Dict[Tuple[str, str], bool] = {}  # CRITICAL FIX
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def detect_fvgs(self, df: pd.DataFrame, symbol: str, timeframe: str, full_scan: bool = False) -> List[FVG]:
        """Detect Fair Value Gaps with proper validation and historical scanning"""
        new_fvgs = []
        atr = self.calculate_atr(df)
        
        # Use 20% of ATR as minimum gap size
        min_gap_size = atr * self.min_fvg_size_multiplier
        
        key = (symbol, timeframe)
        
        # Determine scan range
        if full_scan or not self.historical_scan_done.get(key, False):
            # Full historical scan
            lookback_start = max(2, 10)
            self.historical_scan_done[key] = True
            self.logger.info(f"{Fore.YELLOW}[FULL SCAN] Scanning {len(df)-lookback_start} candles for {symbol} {timeframe}{Style.RESET_ALL}")
        else:
            # Incremental scan (last 20 candles)
            lookback_start = max(2, len(df) - 20)
        
        for i in range(lookback_start, len(df) - 1):
            if i < 2:
                continue
            
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]  # The momentum candle
            candle3 = df.iloc[i]
            
            # Check if FVG already exists at this index
            existing_fvg = any(
                fvg.candle_index == i 
                for fvg in self.active_fvgs.get(key, [])
            )
            if existing_fvg:
                continue
            
            # BULLISH FVG Detection
            if candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                
                if gap_size >= min_gap_size:
                    # Validate momentum candle
                    candle2_body = abs(candle2['close'] - candle2['open'])
                    candle2_is_bullish = candle2['close'] > candle2['open']
                    
                    if candle2_is_bullish and candle2_body > (atr * 0.1):
                        # Check if already tapped
                        was_tapped = self._check_if_fvg_was_tapped(
                            df, i, candle1['high'], candle3['low'], 'bullish'
                        )
                        
                        fvg = FVG(
                            symbol=symbol,
                            timeframe=timeframe,
                            direction='bullish',
                            top=candle3['low'],
                            bottom=candle1['high'],
                            midpoint=(candle3['low'] + candle1['high']) / 2,
                            timestamp=df.index[i],
                            candle_index=i,
                            is_active=not was_tapped,
                            touched=was_tapped
                        )
                        new_fvgs.append(fvg)
                        
                        status = "FILLED" if was_tapped else "ACTIVE"
                        self.logger.info(
                            f"{Fore.CYAN}[FVG DETECTED] {symbol} {timeframe} - BULLISH "
                            f"{fvg.bottom:.5f}-{fvg.top:.5f} (Gap: {gap_size*10000:.1f} pips) [{status}]{Style.RESET_ALL}"
                        )
            
            # BEARISH FVG Detection
            if candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                
                if gap_size >= min_gap_size:
                    # Validate momentum candle
                    candle2_body = abs(candle2['close'] - candle2['open'])
                    candle2_is_bearish = candle2['close'] < candle2['open']
                    
                    if candle2_is_bearish and candle2_body > (atr * 0.1):
                        # Check if already tapped
                        was_tapped = self._check_if_fvg_was_tapped(
                            df, i, candle3['high'], candle1['low'], 'bearish'
                        )
                        
                        fvg = FVG(
                            symbol=symbol,
                            timeframe=timeframe,
                            direction='bearish',
                            top=candle1['low'],
                            bottom=candle3['high'],
                            midpoint=(candle1['low'] + candle3['high']) / 2,
                            timestamp=df.index[i],
                            candle_index=i,
                            is_active=not was_tapped,
                            touched=was_tapped
                        )
                        new_fvgs.append(fvg)
                        
                        status = "FILLED" if was_tapped else "ACTIVE"
                        self.logger.info(
                            f"{Fore.MAGENTA}[FVG DETECTED] {symbol} {timeframe} - BEARISH "
                            f"{fvg.bottom:.5f}-{fvg.top:.5f} (Gap: {gap_size*10000:.1f} pips) [{status}]{Style.RESET_ALL}"
                        )
        
        if key not in self.active_fvgs:
            self.active_fvgs[key] = []
        self.active_fvgs[key].extend(new_fvgs)
        
        return new_fvgs
    
    def _check_if_fvg_was_tapped(self, df: pd.DataFrame, fvg_index: int, 
                                  bottom: float, top: float, direction: str) -> bool:
        """Check if price has already entered the FVG zone after it formed"""
        for i in range(fvg_index + 1, len(df)):
            candle_low = df.iloc[i]['low']
            candle_high = df.iloc[i]['high']
            
            if direction == 'bullish':
                if candle_low <= top:
                    return True
            else:
                if candle_high >= bottom:
                    return True
        
        return False
    
    def update_fvg_status(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Update FVG status based on current price"""
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            return
        
        current_price = df['close'].iloc[-1]
        
        for fvg in self.active_fvgs[key]:
            if not fvg.is_active:
                continue
            
            if fvg.bottom <= current_price <= fvg.top:
                fvg.touched = True
            
            if fvg.direction == 'bullish' and current_price < fvg.bottom:
                fvg.is_active = False
            elif fvg.direction == 'bearish' and current_price > fvg.top:
                fvg.is_active = False
        
        active = [f for f in self.active_fvgs[key] if f.is_active]
        inactive = [f for f in self.active_fvgs[key] if not f.is_active][-10:]
        self.active_fvgs[key] = active + inactive
    
    def get_nearest_fvg(self, symbol: str, timeframe: str, direction: str, current_price: float, 
                        bos_index: Optional[int] = None) -> Optional[FVG]:
        """Get nearest active FVG in specified direction that formed BEFORE the BOS
        
        Smart Money Concept Logic:
        - Bullish BOS (uptrend) → Look for bullish FVG BELOW price (demand zone to buy from)
        - Bearish BOS (downtrend) → Look for bearish FVG ABOVE price (supply zone to sell from)
        """
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            return None
        
        valid_fvgs = [
            fvg for fvg in self.active_fvgs[key]
            if fvg.is_active and fvg.direction == direction
        ]
        
        # Filter FVGs that formed BEFORE the BOS (critical for valid setups)
        if bos_index is not None:
            valid_fvgs = [fvg for fvg in valid_fvgs if fvg.candle_index < bos_index]
        
        if not valid_fvgs:
            return None
        
        if direction == 'bullish':
            # For bullish trend, look for FVG BELOW current price (pullback zone)
            below_fvgs = [fvg for fvg in valid_fvgs if fvg.top < current_price]
            if below_fvgs:
                # Return the nearest one (highest FVG below price)
                return max(below_fvgs, key=lambda x: x.top)
        else:
            # For bearish trend, look for FVG ABOVE current price (pullback zone)
            above_fvgs = [fvg for fvg in valid_fvgs if fvg.bottom > current_price]
            if above_fvgs:
                # Return the nearest one (lowest FVG above price)
                return min(above_fvgs, key=lambda x: x.bottom)
        
        return None
    
    def get_fvg_statistics(self, symbol: str, timeframe: str) -> Dict:
        """Get detailed FVG statistics"""
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            return {
                'total': 0,
                'active': 0,
                'filled': 0,
                'bullish_active': 0,
                'bearish_active': 0,
                'untapped_bullish': 0,
                'untapped_bearish': 0
            }
        
        fvgs = self.active_fvgs[key]
        active = [f for f in fvgs if f.is_active]
        filled = [f for f in fvgs if not f.is_active]
        
        return {
            'total': len(fvgs),
            'active': len(active),
            'filled': len(filled),
            'bullish_active': len([f for f in active if f.direction == 'bullish']),
            'bearish_active': len([f for f in active if f.direction == 'bearish']),
            'untapped_bullish': len([f for f in active if f.direction == 'bullish' and not f.touched]),
            'untapped_bearish': len([f for f in active if f.direction == 'bearish' and not f.touched]),
        }


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Manages risk, position sizing, and exposure limits"""
    
    def __init__(self, 
                 risk_per_trade_pct: float = 1.0,
                 max_open_trades: int = 6,
                 max_exposure_per_symbol_pct: float = 2.0,
                 max_daily_loss_pct: float = 3.0,
                 max_daily_loss_usd: float = 300.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_trades = max_open_trades
        self.max_exposure_per_symbol_pct = max_exposure_per_symbol_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss_usd = max_daily_loss_usd
        
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.daily_pnl = 0.0
        self.is_trading_suspended = False
        self.last_reset_date = datetime.now().date()
        
    def update_equity(self):
        """Update current equity from MT5"""
        account_info = mt5.account_info()
        if account_info:
            self.current_equity = account_info.equity
            
            current_date = datetime.now().date()
            if current_date > self.last_reset_date:
                self.reset_daily_tracking()
                self.last_reset_date = current_date
    
    def reset_daily_tracking(self):
        """Reset daily tracking variables"""
        self.starting_equity = self.current_equity
        self.daily_pnl = 0.0
        self.is_trading_suspended = False
        self.logger.info(f"{Fore.GREEN}[DAILY RESET] Starting equity: ${self.starting_equity:.2f}{Style.RESET_ALL}")
    
    def check_drawdown_limits(self) -> bool:
        """Check if drawdown limits have been exceeded"""
        if self.is_trading_suspended:
            return False
        
        self.update_equity()
        
        daily_loss = self.starting_equity - self.current_equity
        daily_loss_pct = (daily_loss / self.starting_equity) * 100 if self.starting_equity > 0 else 0
        
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.is_trading_suspended = True
            self.logger.critical(
                f"{Fore.RED}[TRADING SUSPENDED] Daily loss limit reached: "
                f"{daily_loss_pct:.2f}% (limit: {self.max_daily_loss_pct}%){Style.RESET_ALL}"
            )
            return False
        
        if daily_loss >= self.max_daily_loss_usd:
            self.is_trading_suspended = True
            self.logger.critical(
                f"{Fore.RED}[TRADING SUSPENDED] Daily loss limit reached: "
                f"${daily_loss:.2f} (limit: ${self.max_daily_loss_usd}){Style.RESET_ALL}"
            )
            return False
        
        return True
    
    def calculate_lot_size(self, symbol: str, entry_price: float, sl_price: float, 
                          session_risk_multiplier: float = 1.0) -> float:
        """Calculate position size based on risk"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0
        
        # Calculate risk amount with session multiplier
        base_risk_amount = (self.current_equity * self.risk_per_trade_pct) / 100
        risk_amount = base_risk_amount * session_risk_multiplier
        
        # Calculate pip distance
        pip_distance = abs(entry_price - sl_price)
        
        # Calculate lot size
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        if tick_size == 0:
            return 0.0
        
        pips = pip_distance / tick_size
        pip_value = tick_value
        
        lot_size = risk_amount / (pips * pip_value) if pips > 0 else 0
        
        # Round to allowed lot step
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Ensure within limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        self.logger.debug(f"Calculated lot size: {lot_size} for {symbol} (Risk: ${risk_amount:.2f})")
        
        return lot_size
    
    def can_open_trade(self, symbol: str, active_trade_count: int) -> bool:
        """Check if new trade can be opened based on limits"""
        if not self.check_drawdown_limits():
            return False
        
        if active_trade_count >= self.max_open_trades:
            self.logger.warning(f"Max open trades limit reached: {active_trade_count}/{self.max_open_trades}")
            return False
        
        return True


# ============================================================================
# TRADE EXECUTOR
# ============================================================================

class TradeExecutor:
    """Handles trade execution and management"""
    
    def __init__(self, magic_number: int = 234567):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.magic_number = magic_number
        
    def send_order(self, 
                   symbol: str,
                   order_type: int,
                   lot_size: float,
                   entry_price: float,
                   sl: float,
                   tp: float,
                   comment: str = "FVG-BOS Bot") -> Optional[int]:
        """Send order to MT5"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select {symbol}")
                return None
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.comment}")
            return None
        
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        self.logger.info(
            f"{Fore.YELLOW}[TRADE OPENED] {direction} {symbol} | "
            f"Lots: {lot_size} | Entry: {entry_price:.5f} | "
            f"SL: {sl:.5f} | TP: {tp:.5f} | Ticket: {result.order}{Style.RESET_ALL}"
        )
        
        return result.order
    
    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify stop loss and take profit for a position"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to modify position {ticket}: {result.comment}")
            return False
        
        return True
    
    def close_position(self, ticket: int) -> bool:
        """Close position by ticket"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "Close by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to close position {ticket}: {result.comment}")
            return False
        
        self.logger.info(f"{Fore.CYAN}[POSITION CLOSED] Ticket: {ticket}{Style.RESET_ALL}")
        return True


# ============================================================================
# TRADE MONITOR
# ============================================================================

class TradeMonitor:
    """Monitors trades, cooldowns, and manages trade lifecycle"""
    
    def __init__(self, cooldown_minutes: int = 30):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cooldown_minutes = cooldown_minutes
        self.trade_history: Dict[str, List[TradeRecord]] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.active_trades: Dict[int, TradeRecord] = {}
        
    def record_trade(self, trade: TradeRecord):
        """Record a new trade"""
        self.active_trades[trade.ticket] = trade
        
        if trade.symbol not in self.trade_history:
            self.trade_history[trade.symbol] = []
        
        self.trade_history[trade.symbol].append(trade)
        self.last_trade_time[trade.symbol] = trade.entry_time
        
    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.last_trade_time:
            return False
        
        time_since_last = datetime.now() - self.last_trade_time[symbol]
        in_cooldown = time_since_last < timedelta(minutes=self.cooldown_minutes)
        
        if in_cooldown:
            remaining = self.cooldown_minutes - (time_since_last.seconds // 60)
            self.logger.debug(f"{symbol} in cooldown: {remaining} minutes remaining")
        
        return in_cooldown
    
    def calculate_trailing_stop(self, trade: TradeRecord, config: Dict) -> Optional[float]:
        """Calculate new trailing stop level"""
        symbol_info = mt5.symbol_info(trade.symbol)
        if not symbol_info:
            return None
            
        tick = mt5.symbol_info_tick(trade.symbol)
        if not tick:
            return None
            
        current_price = tick.bid if trade.direction == 'buy' else tick.ask
        
        # Calculate profit in R multiples
        initial_risk = abs(trade.entry_price - trade.initial_sl)
        current_profit = (current_price - trade.entry_price) if trade.direction == 'buy' else (trade.entry_price - current_price)
        r_multiple = current_profit / initial_risk if initial_risk != 0 else 0
        
        # Check if trailing should be activated
        if not trade.trailing_activated and r_multiple >= config['trailing_stop_activation']:
            trade.trailing_activated = True
            self.logger.info(f"Trailing stop activated for ticket {trade.ticket} at {r_multiple:.2f}R profit")
            
        if trade.trailing_activated:
            # Calculate ATR-based trailing distance
            atr = self.calculate_atr(trade.symbol, mt5.TIMEFRAME_M15)
            trail_distance = atr * config['trailing_stop_distance']
            
            # Calculate new stop level
            if trade.direction == 'buy':
                new_sl = current_price - trail_distance
                if new_sl > trade.sl:
                    return new_sl
            else:
                new_sl = current_price + trail_distance
                if new_sl < trade.sl:
                    return new_sl
                    
        return None
        
    def calculate_atr(self, symbol: str, timeframe: int, period: int = 14) -> float:
        """Calculate ATR for trailing stop"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return 0.0
            
        df = pd.DataFrame(rates)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()
        return float(atr)
        
    def update_trade_status(self):
        """Update status of active trades from MT5"""
        positions = mt5.positions_get()
        active_tickets = {pos.ticket for pos in positions} if positions else set()
        
        for ticket, trade in list(self.active_trades.items()):
            if ticket not in active_tickets:
                trade.status = 'closed'
                trade.exit_time = datetime.now()
                
                deals = mt5.history_deals_get(ticket=ticket)
                if deals and len(deals) > 0:
                    exit_deal = deals[-1]
                    trade.exit_price = exit_deal.price
                    trade.pnl = exit_deal.profit
                    
                    self.logger.info(
                        f"{Fore.YELLOW}[TRADE CLOSED] {trade.symbol} | "
                        f"Direction: {trade.direction} | P&L: ${trade.pnl:.2f}{Style.RESET_ALL}"
                    )
                
                del self.active_trades[ticket]


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class SessionBasedFVGBOSBot:
    """Session-based FVG-BOS Trading Bot"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Initialize managers
        self.session_manager = SessionManager()
        self.spread_monitor = SpreadMonitor()
        self.correlation_manager = CorrelationManager()
        
        # Initialize components
        self.data_handler = DataHandler(
            timeframes=config['timeframes']
        )
        
        self.structure_analyzer = StructureAnalyzer(
            swing_lookback=config.get('swing_lookback', 10),
            min_bos_distance=config.get('min_bos_distance', 5)
        )
        
        self.fvg_detector = FVGDetector(
            min_fvg_size_atr_multiplier=config.get('min_fvg_atr', 0.2)
        )
        
        self.risk_manager = RiskManager(
            risk_per_trade_pct=config.get('risk_per_trade', 1.0),
            max_open_trades=config.get('max_open_trades', 6),
            max_exposure_per_symbol_pct=config.get('max_exposure', 2.0),
            max_daily_loss_pct=config.get('max_daily_loss_pct', 3.0),
            max_daily_loss_usd=config.get('max_daily_loss_usd', 300.0)
        )
        
        self.trade_executor = TradeExecutor(
            magic_number=config.get('magic_number', 234567)
        )
        
        self.trade_monitor = TradeMonitor(
            cooldown_minutes=config.get('cooldown_minutes', 30)
        )
        
        self.is_running = False
        self.update_interval = config.get('update_interval_seconds', 60)
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        
        try:
            if not mt5.initialize(path=mt5_path):
                if not mt5.initialize():
                    self.logger.error("MT5 initialization failed")
                    return False
            
            self.logger.info(f"{Fore.GREEN}MT5 initialized successfully{Style.RESET_ALL}")
            
            account_info = mt5.account_info()
            if account_info:
                self.logger.info(f"Account: {account_info.login} | Balance: ${account_info.balance:.2f}")
                self.risk_manager.update_equity()
                self.risk_manager.reset_daily_tracking()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def check_entry_signal(self, symbol: str, trade_tf: str, trend_tf: str) -> Optional[Dict]:
        """Check for entry signal based on BOS and FVG with proper Smart Money logic
        
        Valid Setup:
        1. BOS confirms trend direction on higher timeframe
        2. Price pulls back into FVG that formed BEFORE the BOS
        3. FVG must be in correct position relative to current price
        """
        trend_direction = self.structure_analyzer.get_trend_direction(symbol, trend_tf)
        if not trend_direction:
            return None
        
        trade_data = self.data_handler.get_data(symbol, trade_tf)
        if trade_data is None:
            return None
        
        bos = self.structure_analyzer.detect_bos(trade_data, symbol, trade_tf)
        
        # Only proceed if BOS aligns with higher timeframe trend
        if bos and bos.direction == trend_direction:
            current_price = trade_data['close'].iloc[-1]
            
            # Get nearest FVG that formed BEFORE the BOS
            nearest_fvg = self.fvg_detector.get_nearest_fvg(
                symbol, trade_tf, bos.direction, current_price, bos_index=bos.candle_index
            )
            
            if nearest_fvg:
                # Verify FVG is in correct position (formed before BOS)
                if nearest_fvg.candle_index >= bos.candle_index:
                    self.logger.debug(f"{symbol}: FVG formed AFTER BOS - invalid setup")
                    return None
                
                # Check if price is in FVG entry zone (20-70% fill)
                if bos.direction == 'bullish':
                    # For bullish: price should be pulling back DOWN into FVG below
                    if nearest_fvg.top >= current_price:
                        self.logger.debug(f"{symbol}: Price not below FVG for bullish setup")
                        return None
                    
                    fvg_range = nearest_fvg.top - nearest_fvg.bottom
                    entry_zone_bottom = nearest_fvg.bottom + (0.2 * fvg_range)  # 20% fill
                    entry_zone_top = nearest_fvg.bottom + (0.7 * fvg_range)     # 70% fill
                    
                    # Price must be pulling back INTO the FVG (between 20-70%)
                    if entry_zone_bottom <= current_price <= entry_zone_top:
                        fill_percentage = ((current_price - nearest_fvg.bottom) / fvg_range) * 100
                        self.logger.info(
                            f"{Fore.GREEN}[VALID SETUP] {symbol} BULLISH | "
                            f"BOS@{bos.break_price:.5f} | FVG: {nearest_fvg.bottom:.5f}-{nearest_fvg.top:.5f} | "
                            f"Price: {current_price:.5f} ({fill_percentage:.1f}% fill){Style.RESET_ALL}"
                        )
                        return {
                            'symbol': symbol,
                            'direction': 'buy',
                            'entry_price': current_price,
                            'fvg': nearest_fvg,
                            'bos': bos
                        }
                else:
                    # For bearish: price should be pulling back UP into FVG above
                    if nearest_fvg.bottom <= current_price:
                        self.logger.debug(f"{symbol}: Price not above FVG for bearish setup")
                        return None
                    
                    fvg_range = nearest_fvg.top - nearest_fvg.bottom
                    entry_zone_top = nearest_fvg.top - (0.2 * fvg_range)        # 20% fill from top
                    entry_zone_bottom = nearest_fvg.top - (0.7 * fvg_range)     # 70% fill from top
                    
                    # Price must be pulling back INTO the FVG (between 20-70%)
                    if entry_zone_bottom <= current_price <= entry_zone_top:
                        fill_percentage = ((nearest_fvg.top - current_price) / fvg_range) * 100
                        self.logger.info(
                            f"{Fore.RED}[VALID SETUP] {symbol} BEARISH | "
                            f"BOS@{bos.break_price:.5f} | FVG: {nearest_fvg.bottom:.5f}-{nearest_fvg.top:.5f} | "
                            f"Price: {current_price:.5f} ({fill_percentage:.1f}% fill){Style.RESET_ALL}"
                        )
                        return {
                            'symbol': symbol,
                            'direction': 'sell',
                            'entry_price': current_price,
                            'fvg': nearest_fvg,
                            'bos': bos
                        }
        
        return None
    
    def calculate_sl_tp(self, signal: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit"""
        fvg = signal['fvg']
        entry_price = signal['entry_price']
        
        symbol = signal['symbol']
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point
        buffer_pips = 5 * point
        
        if signal['direction'] == 'buy':
            sl = fvg.bottom - buffer_pips
            risk = entry_price - sl
            tp = entry_price + (risk * self.config.get('risk_reward_ratio', 2.0))
        else:
            sl = fvg.top + buffer_pips
            risk = sl - entry_price
            tp = entry_price - (risk * self.config.get('risk_reward_ratio', 2.0))
        
        return sl, tp
    
    def execute_signal(self, signal: Dict, session: str):
        """Execute trading signal"""
        symbol = signal['symbol']
        
        # Check spread
        max_spread = self.session_manager.get_max_spread()
        spread_ok, spread_msg = self.spread_monitor.is_spread_acceptable(symbol, max_spread)
        
        if not spread_ok:
            self.logger.warning(f"Skipping {symbol}: {spread_msg}")
            return
        
        # Check correlation
        can_trade, corr_msg = self.correlation_manager.check_correlation_conflict(
            symbol, signal['direction'], self.trade_monitor.active_trades
        )
        
        if not can_trade:
            self.logger.warning(f"Skipping {symbol}: {corr_msg}")
            return
        
        # Check if can open trade
        active_count = len(self.trade_monitor.active_trades)
        if not self.risk_manager.can_open_trade(symbol, active_count):
            self.logger.warning(f"Cannot open trade for {symbol} - risk limits")
            return
        
        # Check cooldown
        if self.trade_monitor.is_in_cooldown(symbol):
            self.logger.debug(f"Skipping {symbol} - in cooldown period")
            return
        
        # Calculate SL and TP
        sl, tp = self.calculate_sl_tp(signal)
        
        # Calculate lot size with session risk multiplier
        session_risk = self.session_manager.get_risk_multiplier()
        lot_size = self.risk_manager.calculate_lot_size(symbol, signal['entry_price'], sl, session_risk)
        
        if lot_size == 0:
            self.logger.error(f"Invalid lot size for {symbol}")
            return
        
        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if signal['direction'] == 'buy' else mt5.ORDER_TYPE_SELL
        
        # Send order
        ticket = self.trade_executor.send_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            entry_price=signal['entry_price'],
            sl=sl,
            tp=tp,
            comment=f"FVG-BOS {session}"
        )
        
        if ticket:
            trade = TradeRecord(
                ticket=ticket,
                symbol=symbol,
                direction=signal['direction'],
                entry_price=signal['entry_price'],
                sl=sl,
                tp=tp,
                lot_size=lot_size,
                entry_time=datetime.now(),
                session=session
            )
            self.trade_monitor.record_trade(trade)
    
    def update_cycle(self):
        """Single update cycle with session awareness"""
        try:
            # Check current session
            if not self.session_manager.is_trading_active():
                self.logger.debug(f"Outside trading hours")
                return
            
            # Get active symbols for current session
            active_symbols = self.session_manager.get_active_symbols()
            
            if not active_symbols:
                return
            
            # Update data for active symbols only
            if not self.data_handler.update_all_data(active_symbols):
                return
            
            # Update trade statuses and trailing stops
            for ticket, trade in list(self.trade_monitor.active_trades.items()):
                new_sl = self.trade_monitor.calculate_trailing_stop(trade, self.config)
                
                if new_sl is not None and abs(new_sl - trade.sl) > mt5.symbol_info(trade.symbol).point:
                    if self.trade_executor.modify_sl_tp(ticket, new_sl, trade.tp):
                        trade.sl = new_sl
                        trade.last_trailing_update = datetime.now()
            
            self.trade_monitor.update_trade_status()
            
            # Get current session
            current_session = self.session_manager.current_session
            
            # Process each active symbol
            for symbol in active_symbols:
                trade_tf = self.config['trading_timeframe']
                trend_tf = self.config['trend_timeframe']
                
                trade_data = self.data_handler.get_data(symbol, trade_tf)
                trend_data = self.data_handler.get_data(symbol, trend_tf)
                
                if trade_data is None or trend_data is None:
                    continue
                
                # Detect swing points and BOS
                self.structure_analyzer.detect_swing_points(trade_data, symbol, trade_tf)
                self.structure_analyzer.detect_swing_points(trend_data, symbol, trend_tf)
                self.structure_analyzer.detect_bos(trend_data, symbol, trend_tf)
                
                # Detect FVGs (full_scan=False for incremental detection)
                self.fvg_detector.detect_fvgs(trade_data, symbol, trade_tf, full_scan=False)
                self.fvg_detector.update_fvg_status(trade_data, symbol, trade_tf)
                
                # Check for entry signals
                signal = self.check_entry_signal(symbol, trade_tf, trend_tf)
                
                if signal:
                    self.logger.info(
                        f"{Fore.GREEN}[SIGNAL] {symbol} {signal['direction'].upper()} @ {signal['entry_price']:.5f}{Style.RESET_ALL}"
                    )
                    self.execute_signal(signal, current_session)
            
        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}", exc_info=True)
    
    def print_status(self):
        """Print bot status with session info"""
        session = self.session_manager.current_session or "INACTIVE"
        local_time = datetime.now().strftime('%H:%M:%S')
        
        session_stats = self.session_manager.get_session_stats()
        exposure = self.correlation_manager.get_exposure_by_currency(self.trade_monitor.active_trades)
        
        self.risk_manager.update_equity()
        
        print(f"\n{'='*80}")
        print(f"SESSION-BASED BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Local Time: {local_time}")
        print(f"Current Session: {session}")
        print(f"Equity: ${self.risk_manager.current_equity:.2f}")
        print(f"Active Trades: {len(self.trade_monitor.active_trades)}/{self.config['max_open_trades']}")
        print(f"Status: {'SUSPENDED' if self.risk_manager.is_trading_suspended else 'ACTIVE'}")
        
        if session != "INACTIVE":
            active_symbols = self.session_manager.get_active_symbols()
            print(f"\nActive Symbols ({len(active_symbols)}): {', '.join(active_symbols)}")
            print(f"Risk Multiplier: {self.session_manager.get_risk_multiplier():.2f}")
            print(f"Max Spread: {self.session_manager.get_max_spread():.1f} pips")
            
            # Show FVG statistics for active symbols
            print(f"\nFVG Statistics:")
            for symbol in active_symbols[:3]:  # Show first 3 symbols
                stats = self.fvg_detector.get_fvg_statistics(symbol, self.config['trading_timeframe'])
                if stats['total'] > 0:
                    print(f"  {symbol}: Total={stats['total']} | "
                          f"Active={stats['active']} (↑{stats['bullish_active']} ↓{stats['bearish_active']}) | "
                          f"Untapped={stats['untapped_bullish']+stats['untapped_bearish']}")
        
        if exposure:
            print(f"\nCurrency Exposure:")
            for currency, exp in sorted(exposure.items(), key=lambda x: abs(x[1]), reverse=True):
                direction = "LONG" if exp > 0 else "SHORT"
                print(f"  {currency}: {direction} {abs(exp)}")
        
        if session_stats:
            print(f"\nSession Statistics:")
            for sess, stats in session_stats.items():
                print(f"  {sess}: {stats['trades']} trades | "
                      f"W:{stats['wins']} L:{stats['losses']} | "
                      f"WR:{stats['win_rate']:.1f}%")
        
        print(f"{'='*80}\n")
    
    def run(self):
        """Main run loop"""
        self.logger.info(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}SESSION-BASED FVG-BOS TRADING BOT STARTED{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        
        if not self.initialize_mt5():
            return
        
        self.is_running = True
        status_counter = 0
        
        try:
            while self.is_running:
                self.update_cycle()
                
                status_counter += 1
                if status_counter >= 10:
                    self.print_status()
                    status_counter = 0
                
                time_module.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            self.logger.info(f"{Fore.YELLOW}Bot stopped by user{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"Critical error: {e}", exc_info=True)
        finally:
            mt5.shutdown()
            self.logger.info(f"{Fore.RED}Bot shutdown complete{Style.RESET_ALL}")


# ============================================================================
# CONFIGURATION & ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    setup_logging()
    
    config = {
        # Timeframes
        'timeframes': {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
        },
        'trading_timeframe': 'M15',
        'trend_timeframe': 'H1',
        
        # Strategy parameters
        'swing_lookback': 10,
        'min_bos_distance': 5,
        'min_fvg_atr': 0.2,  # 20% of ATR minimum
        'risk_reward_ratio': 2.0,
        
        # Risk management
        'risk_per_trade': 2.0,
        'max_open_trades': 10,
        'max_exposure': 2.0,
        'max_daily_loss_pct': 3.0,
        'max_daily_loss_usd': 300.0,
        
        # Trade management
        'cooldown_minutes': 30,
        'magic_number': 234567,
        'trailing_stop_activation': 1.5,
        'trailing_stop_distance': 2.0,
        
        # Bot settings
        'update_interval_seconds': 60,
    }
    
    bot = SessionBasedFVGBOSBot(config)
    bot.run()


if __name__ == "__main__":
    main()