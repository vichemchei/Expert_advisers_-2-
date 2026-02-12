import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# Initialize MT5
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# Parameters (Same as live bot)
symbols = ["XAUUSD", "GBPUSD", "GBPJPY", "EURUSD", "AUDUSD", "USDJPY"]
initial_balance = 10000
risk_per_trade = 0.02
stop_loss_pips = 10
take_profit_pips = 20

# Indicators
rsi_period = 14
ema_fast = 21
ema_slow = 50
ema_trend = 200
stoch_k_period = 14
stoch_d_period = 3
stoch_slowing = 3
bb_period = 20
bb_std = 2

# Thresholds
rsi_oversold = 35
rsi_overbought = 65
stoch_oversold = 25
stoch_overbought = 75

# Backtest Settings
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 1)
TIMEFRAME = mt5.TIMEFRAME_M30


# Indicator Functions
def calculate_rsi(prices, period=14):
    series = pd.Series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(prices, period):
    series = pd.Series(prices)
    ema = series.ewm(span=period, adjust=False).mean()
    return ema


def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3):
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)
    close_series = pd.Series(closes)
    
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    k_percent_slowed = k_percent.rolling(window=slowing).mean()
    d_percent = k_percent_slowed.rolling(window=d_period).mean()
    
    return k_percent_slowed, d_percent


def calculate_bollinger_bands(prices, period=20, std=2):
    series = pd.Series(prices)
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return upper_band, sma, lower_band


def detect_rsi_divergence(prices, rsi_values, lookback=5):
    if len(prices) < lookback * 2 or len(rsi_values) < lookback * 2:
        return None
    
    recent_prices = prices[-lookback:]
    recent_rsi = rsi_values[-lookback:]
    prev_prices = prices[-lookback*2:-lookback]
    prev_rsi = rsi_values[-lookback*2:-lookback]
    
    # Bullish divergence
    if min(recent_prices) < min(prev_prices) and min(recent_rsi) > min(prev_rsi):
        return "bullish_divergence"
    
    # Bearish divergence
    if max(recent_prices) > max(prev_prices) and max(recent_rsi) < max(prev_rsi):
        return "bearish_divergence"
    
    return None


def is_trending_market(ema_fast, ema_slow, ema_trend):
    if ema_fast > ema_slow > ema_trend:
        return "uptrend"
    elif ema_fast < ema_slow < ema_trend:
        return "downtrend"
    else:
        return "ranging"


def get_historical_data(symbol, start_date, end_date, timeframe):
    print(f"Fetching data for {symbol}...")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_lot_size(balance, risk_pct, sl_pips, pip_value=10):
    risk_amount = balance * risk_pct
    lot = (risk_amount / (sl_pips * pip_value)) * 0.1
    return max(round(lot, 2), 0.01)


def backtest_enhanced_strategy(symbol, start_date, end_date):
    # Get historical data
    df = get_historical_data(symbol, start_date, end_date, TIMEFRAME)
    
    if df is None:
        return None
    
    data_needed = max(ema_trend, bb_period, stoch_k_period, rsi_period) + 100
    
    if len(df) < data_needed:
        print(f"Not enough data for {symbol}")
        return None
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'].values, rsi_period)
    df['ema21'] = calculate_ema(df['close'].values, ema_fast)
    df['ema50'] = calculate_ema(df['close'].values, ema_slow)
    df['ema200'] = calculate_ema(df['close'].values, ema_trend)
    
    stoch_k, stoch_d = calculate_stochastic(
        df['high'].values, df['low'].values, df['close'].values,
        stoch_k_period, stoch_d_period, stoch_slowing
    )
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        df['close'].values, bb_period, bb_std
    )
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    # Determine trend
    df['trend'] = df.apply(
        lambda row: is_trending_market(row['ema21'], row['ema50'], row['ema200'])
        if pd.notna(row['ema21']) and pd.notna(row['ema50']) and pd.notna(row['ema200'])
        else 'ranging',
        axis=1
    )
    
    # Detect divergence (simplified for backtesting)
    df['divergence'] = None
    
    # Generate signals
    df['signal'] = 0
    
    # BUY conditions
    buy_condition_1 = (
        (df['stoch_k'] < stoch_oversold) & 
        (df['rsi'] < rsi_oversold) & 
        (df['close'] > df['ema200']) & 
        (df['trend'] == 'uptrend')
    )
    
    buy_condition_2 = (
        (df['close'] <= df['bb_lower'] * 1.002) & 
        (df['ema21'] > df['ema50']) & 
        (df['rsi'] > 35)
    )
    
    buy_condition_3 = (
        (df['ema21'] > df['ema50']) & 
        (df['stoch_k'] < 30) & 
        (df['rsi'] > 40) & 
        (df['close'] > df['ema200'])
    )
    
    # SELL conditions
    sell_condition_1 = (
        (df['stoch_k'] > stoch_overbought) & 
        (df['rsi'] > rsi_overbought) & 
        (df['close'] < df['ema200']) & 
        (df['trend'] == 'downtrend')
    )
    
    sell_condition_2 = (
        (df['close'] >= df['bb_upper'] * 0.998) & 
        (df['ema21'] < df['ema50']) & 
        (df['rsi'] < 65)
    )
    
    sell_condition_3 = (
        (df['ema21'] < df['ema50']) & 
        (df['stoch_k'] > 70) & 
        (df['rsi'] < 60) & 
        (df['close'] < df['ema200'])
    )
    
    df.loc[buy_condition_1 | buy_condition_2 | buy_condition_3, 'signal'] = 1
    df.loc[sell_condition_1 | sell_condition_2 | sell_condition_3, 'signal'] = -1
    
    # Remove NaN values
    df = df.dropna()
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Cannot get symbol info for {symbol}")
        return None
    
    point = symbol_info.point
    
    # Simulate trades
    trades = []
    balance = initial_balance
    balance_history = [balance]
    in_position = False
    position_type = None
    entry_price = 0
    entry_time = None
    stop_loss = 0
    take_profit = 0
    lot_size_used = 0
    
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        current_high = current_row['high']
        current_low = current_row['low']
        current_time = current_row['time']
        current_signal = current_row['signal']
        
        # Check if we should close position
        if in_position:
            if position_type == 'buy':
                if current_high >= take_profit:
                    pnl = (take_profit - entry_price) * (lot_size_used * 100000) / point
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'buy',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win',
                        'exit_reason': 'tp'
                    })
                    in_position = False
                elif current_low <= stop_loss:
                    pnl = (stop_loss - entry_price) * (lot_size_used * 100000) / point
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'buy',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'loss',
                        'exit_reason': 'sl'
                    })
                    in_position = False
            
            elif position_type == 'sell':
                if current_low <= take_profit:
                    pnl = (entry_price - take_profit) * (lot_size_used * 100000) / point
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'sell',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win',
                        'exit_reason': 'tp'
                    })
                    in_position = False
                elif current_high >= stop_loss:
                    pnl = (entry_price - stop_loss) * (lot_size_used * 100000) / point
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'sell',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'loss',
                        'exit_reason': 'sl'
                    })
                    in_position = False
        
        # Open new position if signal and not in position
        if not in_position and current_signal != 0:
            lot_size_used = calculate_lot_size(balance, risk_per_trade, stop_loss_pips)
            entry_price = current_price
            entry_time = current_time
            
            if current_signal == 1:  # Buy
                position_type = 'buy'
                stop_loss = entry_price - (stop_loss_pips * point)
                take_profit = entry_price + (take_profit_pips * point)
                in_position = True
            
            elif current_signal == -1:  # Sell
                position_type = 'sell'
                stop_loss = entry_price + (stop_loss_pips * point)
                take_profit = entry_price - (take_profit_pips * point)
                in_position = True
        
        balance_history.append(balance)
    
    # Close any remaining open positions
    if in_position:
        exit_price = df.iloc[-1]['close']
        if position_type == 'buy':
            pnl = (exit_price - entry_price) * (lot_size_used * 100000) / point
        else:
            pnl = (entry_price - exit_price) * (lot_size_used * 100000) / point
        
        balance += pnl
        trades.append({
            'symbol': symbol,
            'type': position_type,
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['time'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'lot_size': lot_size_used,
            'pnl': pnl,
            'result': 'win' if pnl > 0 else 'loss',
            'exit_reason': 'end_of_test'
        })
    
    return {
        'symbol': symbol,
        'trades': trades,
        'final_balance': balance,
        'balance_history': balance_history,
        'df': df
    }


def calculate_metrics(results):
    all_trades = []
    for result in results:
        if result and result['trades']:
            all_trades.extend(result['trades'])
    
    if not all_trades:
        print("No trades to analyze")
        return None, None
    
    trades_df = pd.DataFrame(all_trades)
    
    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['result'] == 'win'])
    losing_trades = len(trades_df[trades_df['result'] == 'loss'])
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    net_profit = trades_df['pnl'].sum()
    
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # Calculate drawdown
    balance_history = results[0]['balance_history'] if results else [initial_balance]
    cumulative_balance = pd.Series(balance_history)
    running_max = cumulative_balance.cummax()
    drawdown = (cumulative_balance - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Return on investment
    final_balance = results[0]['final_balance'] if results else initial_balance
    roi = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Exit reason stats
    tp_exits = len(trades_df[trades_df['exit_reason'] == 'tp'])
    sl_exits = len(trades_df[trades_df['exit_reason'] == 'sl'])
    
    metrics = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Net Profit': round(net_profit, 2),
        'Total Profit': round(total_profit, 2),
        'Total Loss': round(total_loss, 2),
        'Profit Factor': round(profit_factor, 2),
        'Average Win': round(avg_win, 2),
        'Average Loss': round(avg_loss, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'ROI (%)': round(roi, 2),
        'Initial Balance': initial_balance,
        'Final Balance': round(final_balance, 2),
        'TP Exits': tp_exits,
        'SL Exits': sl_exits
    }
    
    return metrics, trades_df


def plot_results(results, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Enhanced Multi-Indicator Bot - Backtest Results', 
                 fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    if results[0] and results[0]['balance_history']:
        axes[0, 0].plot(results[0]['balance_history'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=initial_balance, color='red', linestyle='--', 
                          label='Initial Balance', linewidth=1)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Time Period (15-min bars)')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(range(len(results[0]['balance_history'])), 
                                initial_balance, results[0]['balance_history'], 
                                alpha=0.2, color='blue')
    
    # 2. Win/Loss Distribution
    all_pnls = []
    for result in results:
        if result and result['trades']:
            all_pnls.extend([trade['pnl'] for trade in result['trades']])
    
    if all_pnls:
        wins = [pnl for pnl in all_pnls if pnl > 0]
        losses = [pnl for pnl in all_pnls if pnl < 0]
        axes[0, 1].hist([wins, losses], bins=25, label=['Wins', 'Losses'], 
                       color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Profit/Loss Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # 3. Metrics Table
    axes[1, 0].axis('off')
    metrics_text = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_title('Performance Metrics', fontweight='bold')
    
    # 4. Trades by Symbol
    symbol_data = {}
    for result in results:
        if result and result['trades']:
            symbol = result['symbol']
            wins = len([t for t in result['trades'] if t['result'] == 'win'])
            losses = len([t for t in result['trades'] if t['result'] == 'loss'])
            symbol_data[symbol] = {'wins': wins, 'losses': losses}
    
    if symbol_data:
        symbols_list = list(symbol_data.keys())
        wins_list = [symbol_data[s]['wins'] for s in symbols_list]
        losses_list = [symbol_data[s]['losses'] for s in symbols_list]
        
        x = np.arange(len(symbols_list))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, wins_list, width, label='Wins', color='green', alpha=0.7)
        axes[1, 1].bar(x + width/2, losses_list, width, label='Losses', color='red', alpha=0.7)
        axes[1, 1].set_title('Wins vs Losses by Symbol', fontweight='bold')
        axes[1, 1].set_xlabel('Symbol')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(symbols_list, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest_enhanced_bot.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'backtest_enhanced_bot.png'")
    plt.show()


def run_backtest():
    print("=" * 70)
    print("ENHANCED MULTI-INDICATOR BOT BACKTEST")
    print("=" * 70)
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timeframe: M15 (15-minute charts)")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"\nIndicators:")
    print(f"  - RSI (period {rsi_period}): Oversold={rsi_oversold}, Overbought={rsi_overbought}")
    print(f"  - EMA: {ema_fast}, {ema_slow}, {ema_trend}")
    print(f"  - Stochastic: K={stoch_k_period}, D={stoch_d_period}")
    print(f"  - Bollinger Bands: Period={bb_period}, StdDev={bb_std}")
    print("=" * 70)
    
    results = []
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        result = backtest_enhanced_strategy(symbol, START_DATE, END_DATE)
        if result:
            results.append(result)
            print(f"Completed: {symbol} - {len(result['trades'])} trades")
        else:
            print(f"Failed: {symbol}")
    
    if not results:
        print("\nNo results to display")
        return
    
    print("\n" + "=" * 70)
    print("CALCULATING PERFORMANCE METRICS...")
    print("=" * 70)
    
    metrics, trades_df = calculate_metrics(results)
    
    if metrics:
        print("\nBACKTEST RESULTS:\n")
        for key, value in metrics.items():
            print(f"{key:.<50} {value}")
        
        # Save detailed trade log
        if trades_df is not None and not trades_df.empty:
            trades_df.to_csv('backtest_enhanced_trades.csv', index=False)
            print(f"\nDetailed trade log saved as 'backtest_enhanced_trades.csv'")
        
        # Plot results
        print("\nGenerating charts...")
        plot_results(results, metrics)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
    mt5.shutdown()