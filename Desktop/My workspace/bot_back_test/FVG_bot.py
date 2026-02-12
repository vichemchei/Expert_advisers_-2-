import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gold_bos_fvg_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC_NUMBER = 123456
COMMENT = "BOS-FVG-Bot"
RISK_PERCENT = 0.02
LOOKBACK_BARS = 100
COOLDOWN_HOURS = 1

last_trade_time = None
current_bos = None
current_fvg = None

def initialize_mt5():
    if not mt5.initialize(path=MT5_PATH):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    logger.info("MT5 initialized successfully")
    return True

def get_data(symbol, timeframe, n_candles):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def find_swing_highs_lows(data, lookback=5):
    highs = []
    lows = []
    
    for i in range(lookback, len(data) - lookback):
        is_high = True
        is_low = True
        
        for j in range(1, lookback + 1):
            if data['high'].iloc[i] <= data['high'].iloc[i - j] or data['high'].iloc[i] <= data['high'].iloc[i + j]:
                is_high = False
            if data['low'].iloc[i] >= data['low'].iloc[i - j] or data['low'].iloc[i] >= data['low'].iloc[i + j]:
                is_low = False
        
        if is_high:
            highs.append({'index': i, 'price': data['high'].iloc[i], 'time': data['time'].iloc[i]})
        if is_low:
            lows.append({'index': i, 'price': data['low'].iloc[i], 'time': data['time'].iloc[i]})
    
    return highs, lows

def detect_bos(data):
    if len(data) < 20:
        return None
    
    highs, lows = find_swing_highs_lows(data, lookback=3)
    
    if len(highs) < 2 or len(lows) < 2:
        return None
    
    last_candle = data.iloc[-1]
    prev_candle = data.iloc[-2]
    
    recent_high = highs[-1]['price']
    recent_low = lows[-1]['price']
    
    if last_candle['close'] > recent_high and prev_candle['close'] <= recent_high:
        return {
            'type': 'bullish',
            'break_level': recent_high,
            'time': last_candle['time'],
            'swing_low': lows[-1]['price']
        }
    
    if last_candle['close'] < recent_low and prev_candle['close'] >= recent_low:
        return {
            'type': 'bearish',
            'break_level': recent_low,
            'time': last_candle['time'],
            'swing_high': highs[-1]['price']
        }
    
    return None

def find_fvg(data, bos_type):
    if len(data) < 3:
        return None
    
    fvgs = []
    
    for i in range(2, len(data)):
        candle_1 = data.iloc[i - 2]
        candle_2 = data.iloc[i - 1]
        candle_3 = data.iloc[i]
        
        if bos_type == 'bullish':
            if candle_2['close'] > candle_2['open']:
                gap_bottom = candle_1['high']
                gap_top = candle_3['low']
                
                if gap_bottom < gap_top:
                    fvgs.append({
                        'type': 'bullish',
                        'top': gap_top,
                        'bottom': gap_bottom,
                        'index': i,
                        'time': candle_3['time']
                    })
        
        elif bos_type == 'bearish':
            if candle_2['close'] < candle_2['open']:
                gap_top = candle_1['low']
                gap_bottom = candle_3['high']
                
                if gap_bottom < gap_top:
                    fvgs.append({
                        'type': 'bearish',
                        'top': gap_top,
                        'bottom': gap_bottom,
                        'index': i,
                        'time': candle_3['time']
                    })
    
    if fvgs:
        return fvgs[-1]
    return None

def calculate_fib_levels(bos):
    if bos['type'] == 'bullish':
        swing_high = bos['break_level']
        swing_low = bos['swing_low']
    else:
        swing_high = bos['swing_high']
        swing_low = bos['break_level']
    
    diff = swing_high - swing_low
    
    fib_levels = {
        '0.0': swing_low,
        '0.236': swing_low + diff * 0.236,
        '0.382': swing_low + diff * 0.382,
        '0.5': swing_low + diff * 0.5,
        '0.618': swing_low + diff * 0.618,
        '1.0': swing_high
    }
    
    return fib_levels

def check_price_in_fvg(current_price, fvg):
    if fvg is None:
        return False
    return fvg['bottom'] <= current_price <= fvg['top']

def calculate_lot_size(symbol, account_balance, risk_amount, sl_points):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01
    
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    
    if sl_points == 0 or point == 0:
        return 0.01
    
    sl_value = (sl_points / point) * tick_value
    
    if sl_value == 0:
        return 0.01
    
    lot_size = risk_amount / sl_value
    
    lot_size = max(symbol_info.volume_min, lot_size)
    lot_size = min(symbol_info.volume_max, lot_size)
    
    lot_step = symbol_info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    
    return lot_size

def place_trade(symbol, direction, entry_price, sl, tp):
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        return False
    
    balance = account_info.balance
    risk_amount = balance * RISK_PERCENT
    
    sl_points = abs(entry_price - sl)
    lot_size = calculate_lot_size(symbol, balance, risk_amount, sl_points)
    
    if lot_size < 0.01:
        lot_size = 0.01
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select {symbol}")
            return False
    
    point = symbol_info.point
    
    order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Failed to get tick for {symbol}")
        return False
    
    price = tick.ask if direction == 'buy' else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        logger.error("Order send failed: None result")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.retcode} - {result.comment}")
        return False
    
    logger.info(f"[TRADE EXECUTED] {direction.upper()} at {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Lot: {lot_size}")
    return True

def check_existing_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    
    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            return True
    return False

def is_cooldown_active():
    global last_trade_time
    if last_trade_time is None:
        return False
    
    elapsed = datetime.now() - last_trade_time
    return elapsed < timedelta(hours=COOLDOWN_HOURS)

def main_loop():
    global current_bos, current_fvg, last_trade_time
    
    logger.info("="*60)
    logger.info("Gold BOS-FVG Trading Bot Started")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Timeframe: H1")
    logger.info(f"Risk per trade: {RISK_PERCENT * 100}%")
    logger.info("="*60)
    
    iteration = 0
    
    while True:
        try:
            if not mt5.terminal_info():
                logger.warning("MT5 disconnected, reconnecting...")
                if not initialize_mt5():
                    time.sleep(10)
                    continue
            
            iteration += 1
            
            if check_existing_position(SYMBOL):
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Position already open, monitoring...")
                time.sleep(300)
                continue
            
            if is_cooldown_active():
                remaining = COOLDOWN_HOURS * 3600 - (datetime.now() - last_trade_time).total_seconds()
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Cooldown active, {remaining/60:.1f} minutes remaining")
                time.sleep(300)
                continue
            
            data = get_data(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
            
            if data is None or len(data) < 20:
                logger.warning("Insufficient data")
                time.sleep(60)
                continue
            
            if current_bos is None:
                bos = detect_bos(data)
                
                if bos is not None:
                    current_bos = bos
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] BOS detected ({bos['type'].upper()})")
                    
                    fvg = find_fvg(data, bos['type'])
                    if fvg is not None:
                        current_fvg = fvg
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] FVG identified: {fvg['bottom']:.2f} - {fvg['top']:.2f}")
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for retracement into FVG...")
                    else:
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] No valid FVG found, resetting BOS")
                        current_bos = None
            
            else:
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    time.sleep(60)
                    continue
                
                current_price = tick.bid
                
                if current_fvg is not None and check_price_in_fvg(current_price, current_fvg):
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Price entered FVG zone at {current_price:.2f}")
                    
                    fib_levels = calculate_fib_levels(current_bos)
                    
                    if current_bos['type'] == 'bullish':
                        direction = 'buy'
                        sl = current_bos['swing_low']
                        tp = fib_levels['0.5']
                        entry_price = current_price
                        
                    else:
                        direction = 'sell'
                        sl = current_bos['swing_high']
                        tp = fib_levels['0.5']
                        entry_price = current_price
                    
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Entry triggered at {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                    
                    if place_trade(SYMBOL, direction, entry_price, sl, tp):
                        last_trade_time = datetime.now()
                        current_bos = None
                        current_fvg = None
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Trade placed successfully, entering cooldown")
                    else:
                        logger.error(f"[{datetime.now().strftime('%H:%M:%S')}] Trade execution failed")
                
                else:
                    if iteration % 12 == 0:
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for price to retrace into FVG... Current: {current_price:.2f}")
            
            time.sleep(300)
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)
            continue
    
    mt5.shutdown()
    logger.info("MT5 connection closed")

if __name__ == '__main__':
    if initialize_mt5():
        main_loop()
    else:
        logger.error("Failed to initialize MT5, exiting")