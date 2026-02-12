"""
MetaTrader 5 FVG-BOS Smart Money Concept Trading Bot
Institutional-Grade Implementation
Author: Senior Quantitative Engineer
Version: 1.0.0
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

# Initialize colorama for colored console output
init(autoreset=True)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FVG:
    """Fair Value Gap structure"""
    symbol: str
    timeframe: str
    direction: str  # 'bullish' or 'bearish'
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
    direction: str  # 'bullish' or 'bearish'
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
    type: str  # 'high' or 'low'


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
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'cancelled'
    initial_sl: float = field(init=False)
    trailing_activated: bool = False
    last_trailing_update: Optional[datetime] = None
    
    def __post_init__(self):
        self.initial_sl = self.sl  # Store initial stop loss


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
    log_filename = f"fvg_bos_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
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
# DATA HANDLER
# ============================================================================

class DataHandler:
    """Handles data fetching and management from MT5"""
    
    def __init__(self, symbols: List[str], timeframes: Dict[str, int]):
        """
        Initialize data handler
        
        Args:
            symbols: List of trading symbols
            timeframes: Dict mapping names to MT5 timeframe constants
                       e.g., {'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        
    def fetch_ohlc(self, symbol: str, timeframe: int, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data from MT5
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLC data or None if error
        """
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
    
    def update_all_data(self) -> bool:
        """
        Update data cache for all symbols and timeframes
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for symbol in self.symbols:
                for tf_name, tf_value in self.timeframes.items():
                    df = self.fetch_ohlc(symbol, tf_value)
                    if df is not None:
                        self.data_cache[(symbol, tf_name)] = df
                    else:
                        self.logger.warning(f"Failed to update {symbol} {tf_name}")
                        return False
            
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
        """
        Initialize structure analyzer
        
        Args:
            swing_lookback: Bars to look back for swing detection
            min_bos_distance: Minimum bars between opposite BOS to avoid whipsaws
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.swing_lookback = swing_lookback
        self.min_bos_distance = min_bos_distance
        self.swing_highs: Dict[Tuple[str, str], List[SwingPoint]] = {}
        self.swing_lows: Dict[Tuple[str, str], List[SwingPoint]] = {}
        self.last_bos: Dict[Tuple[str, str], BOS] = {}
        
    def detect_swing_points(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows
        
        Args:
            df: OHLC DataFrame
            symbol: Trading symbol
            timeframe: Timeframe name
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        highs = []
        lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            # Swing High: highest point in lookback window
            window_high = df['high'].iloc[i-self.swing_lookback:i+self.swing_lookback+1]
            if df['high'].iloc[i] == window_high.max():
                highs.append(SwingPoint(
                    price=df['high'].iloc[i],
                    index=i,
                    timestamp=df.index[i],
                    type='high'
                ))
            
            # Swing Low: lowest point in lookback window
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
        """
        Detect Break of Structure
        
        Args:
            df: OHLC DataFrame
            symbol: Trading symbol
            timeframe: Timeframe name
            
        Returns:
            BOS object if detected, None otherwise
        """
        key = (symbol, timeframe)
        
        # Get swing points
        highs = self.swing_highs.get(key, [])
        lows = self.swing_lows.get(key, [])
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        current_price = df['close'].iloc[-1]
        current_index = len(df) - 1
        current_time = df.index[-1]
        
        # Check for Bullish BOS (close above previous swing high)
        for swing_high in reversed(highs[-5:]):  # Check last 5 swing highs
            if current_price > swing_high.price:
                # Check if we had a recent opposite BOS
                if key in self.last_bos:
                    last_bos = self.last_bos[key]
                    bars_since_last = current_index - last_bos.candle_index
                    if last_bos.direction == 'bearish' and bars_since_last < self.min_bos_distance:
                        continue  # Skip to avoid whipsaw
                
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
        
        # Check for Bearish BOS (close below previous swing low)
        for swing_low in reversed(lows[-5:]):  # Check last 5 swing lows
            if current_price < swing_low.price:
                # Check if we had a recent opposite BOS
                if key in self.last_bos:
                    last_bos = self.last_bos[key]
                    bars_since_last = current_index - last_bos.candle_index
                    if last_bos.direction == 'bullish' and bars_since_last < self.min_bos_distance:
                        continue  # Skip to avoid whipsaw
                
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
        """
        Get trend direction based on last BOS
        
        Returns:
            'bullish', 'bearish', or None
        """
        key = (symbol, timeframe)
        if key in self.last_bos:
            return self.last_bos[key].direction
        return None


# ============================================================================
# FVG DETECTOR
# ============================================================================

class FVGDetector:
    """Detects and manages Fair Value Gaps"""
    
    def __init__(self, min_fvg_size_atr_multiplier: float = 0.05):
        """
        Initialize FVG detector
        
        Args:
            min_fvg_size_atr_multiplier: Minimum FVG size as ATR multiplier
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_fvg_size_multiplier = min_fvg_size_atr_multiplier
        self.active_fvgs: Dict[Tuple[str, str], List[FVG]] = {}
        
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
    
    def detect_fvgs(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[FVG]:
        """
        Detect Fair Value Gaps in recent data
        
        Args:
            df: OHLC DataFrame
            symbol: Trading symbol
            timeframe: Timeframe name
            
        Returns:
            List of newly detected FVGs
        """
        new_fvgs = []
        atr = self.calculate_atr(df)
        min_gap_size = atr * self.min_fvg_size_multiplier
        
        # Check last 50 candles for FVGs
        for i in range(len(df) - 50, len(df) - 2):
            if i < 2:
                continue
            
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Candle1.high < Candle3.low
            if candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size >= min_gap_size:
                    fvg = FVG(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction='bullish',
                        top=candle3['low'],
                        bottom=candle1['high'],
                        midpoint=(candle3['low'] + candle1['high']) / 2,
                        timestamp=df.index[i],
                        candle_index=i,
                        is_active=True
                    )
                    new_fvgs.append(fvg)
                    self.logger.info(f"{Fore.CYAN}[FVG DETECTED] {symbol} {timeframe} - BULLISH {fvg.bottom:.5f}-{fvg.top:.5f}{Style.RESET_ALL}")
            
            # Bearish FVG: Candle1.low > Candle3.high
            if candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= min_gap_size:
                    fvg = FVG(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction='bearish',
                        top=candle1['low'],
                        bottom=candle3['high'],
                        midpoint=(candle1['low'] + candle3['high']) / 2,
                        timestamp=df.index[i],
                        candle_index=i,
                        is_active=True
                    )
                    new_fvgs.append(fvg)
                    self.logger.info(f"{Fore.MAGENTA}[FVG DETECTED] {symbol} {timeframe} - BEARISH {fvg.bottom:.5f}-{fvg.top:.5f}{Style.RESET_ALL}")
        
        # Add to active FVGs
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            self.active_fvgs[key] = []
        self.active_fvgs[key].extend(new_fvgs)
        
        return new_fvgs
    
    def update_fvg_status(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Update FVG status based on current price
        Mark FVGs as inactive if price has fully retraced through them
        """
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            return
        
        current_price = df['close'].iloc[-1]
        
        for fvg in self.active_fvgs[key]:
            if not fvg.is_active:
                continue
            
            # Check if price has entered FVG
            if fvg.bottom <= current_price <= fvg.top:
                fvg.touched = True
            
            # Deactivate FVG if price has moved completely through it
            if fvg.direction == 'bullish' and current_price < fvg.bottom:
                fvg.is_active = False
            elif fvg.direction == 'bearish' and current_price > fvg.top:
                fvg.is_active = False
        
        # Clean up old inactive FVGs (keep last 10)
        active = [f for f in self.active_fvgs[key] if f.is_active]
        inactive = [f for f in self.active_fvgs[key] if not f.is_active][-10:]
        self.active_fvgs[key] = active + inactive
    
    def get_nearest_fvg(self, symbol: str, timeframe: str, direction: str, current_price: float) -> Optional[FVG]:
        """
        Get nearest active FVG in specified direction
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe name
            direction: 'bullish' or 'bearish'
            current_price: Current market price
            
        Returns:
            Nearest FVG or None
        """
        key = (symbol, timeframe)
        if key not in self.active_fvgs:
            return None
        
        valid_fvgs = [
            fvg for fvg in self.active_fvgs[key]
            if fvg.is_active and fvg.direction == direction
        ]
        
        if not valid_fvgs:
            return None
        
        # Find nearest FVG
        if direction == 'bullish':
            # For bullish, find FVG below current price
            below_fvgs = [fvg for fvg in valid_fvgs if fvg.top < current_price]
            if below_fvgs:
                return max(below_fvgs, key=lambda x: x.top)
        else:
            # For bearish, find FVG above current price
            above_fvgs = [fvg for fvg in valid_fvgs if fvg.bottom > current_price]
            if above_fvgs:
                return min(above_fvgs, key=lambda x: x.bottom)
        
        return None


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Manages risk, position sizing, and exposure limits"""
    
    def __init__(self, 
                 risk_per_trade_pct: float = 1.0,
                 max_open_trades: int = 3,
                 max_exposure_per_symbol_pct: float = 2.0,
                 max_daily_loss_pct: float = 3.0,
                 max_daily_loss_usd: float = 300.0):
        """
        Initialize risk manager
        
        Args:
            risk_per_trade_pct: Risk per trade as % of equity
            max_open_trades: Maximum number of open trades
            max_exposure_per_symbol_pct: Max exposure per symbol as % of equity
            max_daily_loss_pct: Max daily loss as % of starting equity
            max_daily_loss_usd: Max daily loss in USD
        """
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
            
            # Reset daily tracking if new day
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
        """
        Check if drawdown limits have been exceeded
        
        Returns:
            True if trading should continue, False if suspended
        """
        if self.is_trading_suspended:
            return False
        
        self.update_equity()
        
        # Calculate daily loss
        daily_loss = self.starting_equity - self.current_equity
        daily_loss_pct = (daily_loss / self.starting_equity) * 100 if self.starting_equity > 0 else 0
        
        # Check percentage limit
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.is_trading_suspended = True
            self.logger.critical(
                f"{Fore.RED}[TRADING SUSPENDED] Daily loss limit reached: "
                f"{daily_loss_pct:.2f}% (limit: {self.max_daily_loss_pct}%){Style.RESET_ALL}"
            )
            return False
        
        # Check USD limit
        if daily_loss >= self.max_daily_loss_usd:
            self.is_trading_suspended = True
            self.logger.critical(
                f"{Fore.RED}[TRADING SUSPENDED] Daily loss limit reached: "
                f"${daily_loss:.2f} (limit: ${self.max_daily_loss_usd}){Style.RESET_ALL}"
            )
            return False
        
        return True
    
    def calculate_lot_size(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """
        Calculate position size based on risk
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            sl_price: Stop loss price
            
        Returns:
            Lot size
        """
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0
        
        # Calculate risk amount in USD
        risk_amount = (self.current_equity * self.risk_per_trade_pct) / 100
        
        # Calculate pip distance
        pip_distance = abs(entry_price - sl_price)
        
        # Calculate lot size
        # For forex: pip_value = (pip_distance * lot_size * contract_size)
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
        
        self.logger.debug(f"Calculated lot size: {lot_size} for {symbol} (Risk: ${risk_amount:.2f}, Distance: {pip_distance:.5f})")
        
        return lot_size
    
    def can_open_trade(self, symbol: str) -> bool:
        """
        Check if new trade can be opened based on limits
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if trade can be opened, False otherwise
        """
        # Check drawdown limits
        if not self.check_drawdown_limits():
            return False
        
        # Check max open trades
        positions = mt5.positions_get()
        if positions and len(positions) >= self.max_open_trades:
            self.logger.warning(f"Max open trades limit reached: {len(positions)}/{self.max_open_trades}")
            return False
        
        # Check symbol exposure
        symbol_positions = mt5.positions_get(symbol=symbol)
        if symbol_positions:
            total_volume = sum(pos.volume for pos in symbol_positions)
            # Simplified exposure check (would need proper calculation in production)
            if total_volume >= self.max_exposure_per_symbol_pct:
                self.logger.warning(f"Max exposure limit reached for {symbol}")
                return False
        
        return True


# ============================================================================
# TRADE EXECUTOR
# ============================================================================

# ============================================================================
# TRADE EXECUTOR (CORRECTED)
# ============================================================================

class TradeExecutor:
    """Handles trade execution and management"""
    
    def __init__(self, magic_number: int = 234567):
        """
        Initialize trade executor
        
        Args:
            magic_number: Magic number for identifying bot trades
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.magic_number = magic_number
        self.filling_modes = {}  # Cache filling modes per symbol
    
    def get_filling_mode(self, symbol: str) -> int:
        """
        Get the appropriate filling mode for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5 filling mode constant
        """
        # Check cache first
        if symbol in self.filling_modes:
            return self.filling_modes[symbol]
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"Could not get symbol info for {symbol}, defaulting to FOK")
            return mt5.ORDER_FILLING_FOK
        
        # Check which filling modes are supported
        filling_mode = None
        
        if symbol_info.filling_mode & 1:  # FOK (Fill or Kill)
            filling_mode = mt5.ORDER_FILLING_FOK
            mode_name = "FOK"
        elif symbol_info.filling_mode & 2:  # IOC (Immediate or Cancel)
            filling_mode = mt5.ORDER_FILLING_IOC
            mode_name = "IOC"
        elif symbol_info.filling_mode & 4:  # RETURN
            filling_mode = mt5.ORDER_FILLING_RETURN
            mode_name = "RETURN"
        else:
            # Default to FOK if nothing is specified
            filling_mode = mt5.ORDER_FILLING_FOK
            mode_name = "FOK (default)"
        
        self.logger.info(f"Using filling mode {mode_name} for {symbol}")
        
        # Cache the result
        self.filling_modes[symbol] = filling_mode
        
        return filling_mode
        
    def send_order(self, 
                   symbol: str,
                   order_type: int,
                   lot_size: float,
                   entry_price: float,
                   sl: float,
                   tp: float,
                   comment: str = "FVG-BOS Bot") -> Optional[int]:
        """
        Send order to MT5
        
        Args:
            symbol: Trading symbol
            order_type: mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
            lot_size: Position size
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            
        Returns:
            Ticket number if successful, None otherwise
        """
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select {symbol}")
                return None
        
        # Get current market price for the order
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get tick data for {symbol}")
            return None
        
        # Use current price instead of entry_price for market orders
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Get appropriate filling mode for this symbol
        filling_mode = self.get_filling_mode(symbol)
        
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
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed for {symbol}: {result.comment} (retcode: {result.retcode})")
            
            # If filling mode failed, try alternatives
            if result.retcode == 10030:  # Unsupported filling mode
                self.logger.info(f"Trying alternative filling modes for {symbol}")
                
                # Try all filling modes
                for mode, mode_name in [(mt5.ORDER_FILLING_FOK, "FOK"), 
                                       (mt5.ORDER_FILLING_IOC, "IOC"), 
                                       (mt5.ORDER_FILLING_RETURN, "RETURN")]:
                    if mode != filling_mode:
                        self.logger.info(f"Attempting with {mode_name} filling mode")
                        request["type_filling"] = mode
                        result = mt5.order_send(request)
                        
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            # Cache successful mode
                            self.filling_modes[symbol] = mode
                            self.logger.info(f"Success with {mode_name} mode")
                            break
                        else:
                            self.logger.debug(f"{mode_name} mode failed: {result.comment}")
            
            # Check if ultimately successful
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return None
        
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        self.logger.info(
            f"[TRADE OPENED] {direction} {symbol} | "
            f"Lots: {lot_size} | Entry: {result.price:.5f} | "
            f"SL: {sl:.5f} | TP: {tp:.5f} | Ticket: {result.order}"
        )
        
        return result.order
    
    def close_position(self, ticket: int) -> bool:
        """
        Close position by ticket
        
        Args:
            ticket: Position ticket
            
        Returns:
            True if successful, False otherwise
        """
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        symbol = position.symbol
        
        # Get tick data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get tick data for {symbol}")
            return False
        
        # Prepare close request
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        # Get filling mode
        filling_mode = self.get_filling_mode(symbol)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "Close by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to close position {ticket}: {result.comment}")
            return False
        
        self.logger.info(f"[POSITION CLOSED] Ticket: {ticket}")
        return True
    
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
        
        self.logger.info(f"Modified SL/TP for ticket {ticket}: SL={new_sl:.5f}, TP={new_tp:.5f}")
        return True

# ============================================================================
# TRADE MONITOR
# ============================================================================

class TradeMonitor:
    """Monitors trades, cooldowns, and manages trade lifecycle"""
    
    def __init__(self, cooldown_minutes: int = 30):
        """
        Initialize trade monitor
        
        Args:
            cooldown_minutes: Cooldown period after trade closure
        """
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
        """
        Check if symbol is in cooldown period
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if in cooldown, False otherwise
        """
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
            
        current_price = mt5.symbol_info_tick(trade.symbol).bid if trade.direction == 'buy' else mt5.symbol_info_tick(trade.symbol).ask
        if not current_price:
            return None
            
        # Calculate profit in R multiples (risk multiples)
        initial_risk = abs(trade.entry_price - trade.initial_sl)
        current_profit = (current_price - trade.entry_price) if trade.direction == 'buy' else (trade.entry_price - current_price)
        r_multiple = current_profit / initial_risk if initial_risk != 0 else 0
        
        # Check if trailing should be activated
        if not trade.trailing_activated and r_multiple >= config['trailing_stop_activation']:
            trade.trailing_activated = True
            self.logger.info(f"Trailing stop activated for ticket {trade.ticket} at {r_multiple:.2f}R profit")
            
        if trade.trailing_activated:
            # Calculate ATR-based trailing distance
            atr = self.calculate_atr(trade.symbol, mt5.TIMEFRAME_M15)  # Use 15m timeframe for trailing
            trail_distance = atr * config['trailing_stop_distance']
            
            # Calculate new stop level
            if trade.direction == 'buy':
                new_sl = current_price - trail_distance
                if new_sl > trade.sl:  # Move stop up
                    return new_sl
            else:
                new_sl = current_price + trail_distance
                if new_sl < trade.sl:  # Move stop down
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
        atr = tr.mean()  # Simple average for real-time calculation
        return float(atr)
        
    def update_trade_status(self):
        """Update status of active trades from MT5"""
        positions = mt5.positions_get()
        active_tickets = {pos.ticket for pos in positions} if positions else set()
        
        # Check for closed trades
        for ticket, trade in list(self.active_trades.items()):
            if ticket not in active_tickets:
                # Trade has been closed
                trade.status = 'closed'
                trade.exit_time = datetime.now()
                
                # Try to get history
                deals = mt5.history_deals_get(ticket=ticket)
                if deals and len(deals) > 0:
                    exit_deal = deals[-1]
                    trade.exit_price = exit_deal.price
                    trade.pnl = exit_deal.profit
                    
                    self.logger.info(
                        f"{Fore.YELLOW}[TRADE CLOSED] {trade.symbol} | "
                        f"Direction: {trade.direction} | P&L: ${trade.pnl:.2f}{Style.RESET_ALL}"
                    )
                
                # Remove from active trades
                del self.active_trades[ticket]
    
    def get_trade_statistics(self) -> Dict:
        """Get trading statistics"""
        all_trades = []
        for trades in self.trade_history.values():
            all_trades.extend(trades)
        
        closed_trades = [t for t in all_trades if t.status == 'closed' and t.pnl is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': (len(winning) / len(closed_trades)) * 100 if closed_trades else 0,
            'total_pnl': sum(t.pnl for t in closed_trades),
            'avg_win': sum(t.pnl for t in winning) / len(winning) if winning else 0,
            'avg_loss': sum(t.pnl for t in losing) / len(losing) if losing else 0
        }


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class FVGBOSTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Dict):
        """
        Initialize trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Initialize components
        self.data_handler = DataHandler(
            symbols=config['symbols'],
            timeframes=config['timeframes']
        )
        
        self.structure_analyzer = StructureAnalyzer(
            swing_lookback=config.get('swing_lookback', 10),
            min_bos_distance=config.get('min_bos_distance', 5)
        )
        
        self.fvg_detector = FVGDetector(
            min_fvg_size_atr_multiplier=config.get('min_fvg_atr', 0.05)
        )
        
        self.risk_manager = RiskManager(
            risk_per_trade_pct=config.get('risk_per_trade', 1.0),
            max_open_trades=config.get('max_open_trades', 3),
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
        # Try to initialize with explicit path first
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        
        try:
            if not mt5.initialize(path=mt5_path):
                self.logger.warning(f"MT5 initialization failed with path: {mt5_path}")
                # Try auto-detection as fallback
                if not mt5.initialize():
                    self.logger.error("MT5 initialization failed with auto-detection")
                    self.logger.error(f"Last error: {mt5.last_error()}")
                    return False
            
            self.logger.info(f"{Fore.GREEN}MT5 initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            return False
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            self.logger.info(f"Account: {account_info.login} | Balance: ${account_info.balance:.2f}")
            self.risk_manager.update_equity()
            self.risk_manager.reset_daily_tracking()
        
        return True
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        self.logger.info("MT5 connection closed")
    
    def check_entry_signal(self, symbol: str, trade_tf: str, trend_tf: str) -> Optional[Dict]:
        """
        Check for entry signal based on BOS and FVG
        
        Args:
            symbol: Trading symbol
            trade_tf: Trading timeframe
            trend_tf: Higher timeframe for trend confirmation
            
        Returns:
            Signal dictionary or None
        """
        # Get trend direction from higher timeframe
        trend_direction = self.structure_analyzer.get_trend_direction(symbol, trend_tf)
        if not trend_direction:
            return None
        
        # Check for BOS on trading timeframe
        trade_data = self.data_handler.get_data(symbol, trade_tf)
        if trade_data is None:
            return None
        
        bos = self.structure_analyzer.detect_bos(trade_data, symbol, trade_tf)
        
        # BOS must align with higher timeframe trend
        if bos and bos.direction == trend_direction:
            current_price = trade_data['close'].iloc[-1]
            
            # Look for FVG in same direction
            nearest_fvg = self.fvg_detector.get_nearest_fvg(symbol, trade_tf, bos.direction, current_price)
            
            if nearest_fvg:
                # Check if price is touching FVG midpoint (50% level)
                if bos.direction == 'bullish':
                    if abs(current_price - nearest_fvg.midpoint) / nearest_fvg.midpoint < 0.001:  # Within 0.1%
                        return {
                            'symbol': symbol,
                            'direction': 'buy',
                            'entry_price': current_price,
                            'fvg': nearest_fvg,
                            'bos': bos
                        }
                else:  # bearish
                    if abs(current_price - nearest_fvg.midpoint) / nearest_fvg.midpoint < 0.001:
                        return {
                            'symbol': symbol,
                            'direction': 'sell',
                            'entry_price': current_price,
                            'fvg': nearest_fvg,
                            'bos': bos
                        }
        
        return None
    
    def calculate_sl_tp(self, signal: Dict) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit
        
        Args:
            signal: Entry signal dictionary
            
        Returns:
            Tuple of (sl_price, tp_price)
        """
        fvg = signal['fvg']
        entry_price = signal['entry_price']
        
        # Get symbol info for pip calculation
        symbol = signal['symbol']
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point
        buffer_pips = 5 * point  # 5 pip buffer
        
        if signal['direction'] == 'buy':
            # SL below FVG bottom with buffer
            sl = fvg.bottom - buffer_pips
            # TP = 2 x risk
            risk = entry_price - sl
            tp = entry_price + (risk * self.config.get('risk_reward_ratio', 2.0))
        else:  # sell
            # SL above FVG top with buffer
            sl = fvg.top + buffer_pips
            # TP = 2 x risk
            risk = sl - entry_price
            tp = entry_price - (risk * self.config.get('risk_reward_ratio', 2.0))
        
        return sl, tp
    
    def execute_signal(self, signal: Dict):
        """Execute trading signal"""
        symbol = signal['symbol']
        
        # Check if can open trade
        if not self.risk_manager.can_open_trade(symbol):
            self.logger.warning(f"Cannot open trade for {symbol} - risk limits")
            return
        
        # Check cooldown
        if self.trade_monitor.is_in_cooldown(symbol):
            self.logger.debug(f"Skipping {symbol} - in cooldown period")
            return
        
        # Calculate SL and TP
        sl, tp = self.calculate_sl_tp(signal)
        
        # Calculate lot size
        lot_size = self.risk_manager.calculate_lot_size(symbol, signal['entry_price'], sl)
        
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
            comment=f"FVG-BOS {signal['direction'].upper()}"
        )
        
        if ticket:
            # Record trade
            trade = TradeRecord(
                ticket=ticket,
                symbol=symbol,
                direction=signal['direction'],
                entry_price=signal['entry_price'],
                sl=sl,
                tp=tp,
                lot_size=lot_size,
                entry_time=datetime.now()
            )
            self.trade_monitor.record_trade(trade)
    
    def update_cycle(self):
        """Single update cycle"""
        try:
            # Update data
            if not self.data_handler.update_all_data():
                self.logger.warning("Failed to update data")
                return
            
            # Update trade status and trailing stops
            for ticket, trade in list(self.trade_monitor.active_trades.items()):
                # Calculate new trailing stop if needed
                new_sl = self.trade_monitor.calculate_trailing_stop(trade, self.config)
                
                if new_sl is not None and abs(new_sl - trade.sl) > mt5.symbol_info(trade.symbol).point:
                    # Modify position with new stop loss
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": trade.symbol,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": trade.tp
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        trade.sl = new_sl
                        trade.last_trailing_update = datetime.now()
                        self.logger.info(f"Updated trailing stop for ticket {ticket} to {new_sl:.5f}")
                    else:
                        self.logger.warning(f"Failed to update trailing stop for ticket {ticket}: {result.comment}")
            
            # Update trade statuses
            self.trade_monitor.update_trade_status()
            
            # Check each symbol
            for symbol in self.config['symbols']:
                trade_tf = self.config['trading_timeframe']
                trend_tf = self.config['trend_timeframe']
                
                # Get data
                trade_data = self.data_handler.get_data(symbol, trade_tf)
                trend_data = self.data_handler.get_data(symbol, trend_tf)
                
                if trade_data is None or trend_data is None:
                    continue
                
                # Detect swing points and BOS
                self.structure_analyzer.detect_swing_points(trade_data, symbol, trade_tf)
                self.structure_analyzer.detect_swing_points(trend_data, symbol, trend_tf)
                self.structure_analyzer.detect_bos(trend_data, symbol, trend_tf)
                
                # Detect FVGs
                self.fvg_detector.detect_fvgs(trade_data, symbol, trade_tf)
                self.fvg_detector.update_fvg_status(trade_data, symbol, trade_tf)
                
                # Check for entry signals
                signal = self.check_entry_signal(symbol, trade_tf, trend_tf)
                
                if signal:
                    self.logger.info(
                        f"{Fore.GREEN}[SIGNAL] {symbol} {signal['direction'].upper()} @ {signal['entry_price']:.5f}{Style.RESET_ALL}"
                    )
                    self.execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}", exc_info=True)
    
    def print_status(self):
        """Print bot status"""
        stats = self.trade_monitor.get_trade_statistics()
        self.risk_manager.update_equity()
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Equity: ${self.risk_manager.current_equity:.2f}")
        print(f"Daily P&L: ${stats['total_pnl']:.2f}")
        print(f"Active Trades: {len(self.trade_monitor.active_trades)}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Trading Status: {'SUSPENDED' if self.risk_manager.is_trading_suspended else 'ACTIVE'}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def run(self):
        """Main run loop"""
        self.logger.info(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}FVG-BOS TRADING BOT STARTED{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        if not self.initialize_mt5():
            return
        
        self.is_running = True
        status_counter = 0
        
        try:
            while self.is_running:
                self.update_cycle()
                
                # Print status every 10 cycles
                status_counter += 1
                if status_counter >= 10:
                    self.print_status()
                    status_counter = 0
                
                # Sleep
                time_module.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            self.logger.info(f"{Fore.YELLOW}Bot stopped by user{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"Critical error: {e}", exc_info=True)
        finally:
            self.shutdown_mt5()
            self.logger.info(f"{Fore.RED}Bot shutdown complete{Style.RESET_ALL}")


# ============================================================================
# CONFIGURATION & ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    # Setup logging
    setup_logging()
    
    # Configuration
    config = {
        # Symbols to trade
        'symbols':  [
                # Majors
    'EURUSD',   # Essential - highest liquidity
    'GBPUSD',   # Volatile, clear FVGs
    'USDJPY',   # Different USD exposure
    'AUDUSD',   # Commodity-linked
    'EURJPY',   # Cross currency
    'GBPJPY',   # High volatility
    'EURGBP',   # Pure European exposure
    'USDCAD',   # Oil correlation
     
    ],


        #'symbols': ['EURUSD', 'GBPUSD', 'USDJPY','XAUUSD','XAUEUR','AUDUSD'] ,
        # Timeframes
        'timeframes': {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
        },
        'trading_timeframe': 'M15',  # Timeframe for entries
        'trend_timeframe': 'H1',      # Higher timeframe for trend
        
        # Strategy parameters
        'swing_lookback': 10,          # Bars for swing detection
        'min_bos_distance': 5,         # Min bars between opposite BOS
        'min_fvg_atr': 0.05,          # Min FVG size (ATR multiplier)
        'risk_reward_ratio': 2.0,      # Risk:Reward ratio
        
        # Risk management
        'risk_per_trade': 1.0,         # % of equity per trade
        'max_open_trades': 10,          # Max simultaneous trades
        'max_exposure': 3.0,           # Max exposure per symbol (%)
        'max_daily_loss_pct': 3.0,    # Max daily loss (%)
        'max_daily_loss_usd': 30.0,  # Max daily loss (USD)
        
        # Trade management
        'cooldown_minutes': 30,        # Minutes after trade close
        'magic_number': 234567,        # EA magic number

         # ============================================================
        # TRAILING STOP CONFIGURATION (NEW - ADD THESE)
        # ============================================================
        'trailing_stop_activation': 1.5,    # Activate trailing at 1.5R profit
        'trailing_stop_distance': 2.0,      # Trail 2 ATRs behind price
        'trailing_stop_min_distance': 50,   # Minimum trail distance in points
        'trailing_update_threshold': 5,     # Update only if movement > 5 points
        
        # Bot settings
        'update_interval_seconds': 60, # Update frequency
    }
    
    # Create and run bot
    bot = FVGBOSTradingBot(config)
    bot.run()


if __name__ == "__main__":
    main() 