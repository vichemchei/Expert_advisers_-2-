import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Backtest Parameters ---------
symbols = ["XAUUSD", "GBPUSD", "GBPJPY", "EURUSD", "AUDUSD", "USDJPY"]
initial_balance = 1000
risk_per_trade = 0.20
stop_loss_pips = 100
take_profit_pips = 150
rsi_period = 14
macd_fast = 8
macd_slow = 21
macd_signal = 5
stoch_k_period = 14
stoch_d_period = 3
stoch_slowing = 3

# Trailing Stop Parameters
TRAILING_STOP_ENABLED = True
TRAILING_START_PIPS = 50
TRAILING_STOP_DISTANCE_PIPS = 25

# Backtest Settings
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 1)
TIMEFRAME = mt5.TIMEFRAME_H1


# --------- Helper Functions ---------
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    series = pd.Series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast_period=8, slow_period=21, signal_period=5):
    """Calculate MACD line, Signal line, and Histogram"""
    series = pd.Series(prices)
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3):
    """Calculate Stochastic Oscillator %K and %D"""
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)
    close_series = pd.Series(closes)
    
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    k_percent_slowed = k_percent.rolling(window=slowing).mean()
    d_percent = k_percent_slowed.rolling(window=d_period).mean()
    
    return k_percent_slowed, d_percent


def get_historical_data(symbol, start_date, end_date, timeframe):
    """Fetch historical data from MT5"""
    print(f"Fetching data for {symbol}...")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df


def calculate_lot_size(balance, risk_pct, sl_pips, pip_value=10):
    """Calculate position size based on risk"""
    risk_amount = balance * risk_pct
    lot = (risk_amount / (sl_pips * pip_value)) * 0.1
    return max(round(lot, 2), 0.01)


class Position:
    """Class to track open positions with trailing stop logic"""
    def __init__(self, symbol, position_type, entry_price, entry_time, lot_size, sl, tp, point):
        self.symbol = symbol
        self.position_type = position_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.lot_size = lot_size
        self.sl = sl
        self.tp = tp
        self.point = point
        self.highest_price = entry_price if position_type == 'buy' else entry_price
        self.lowest_price = entry_price if position_type == 'sell' else entry_price
        self.trailing_active = False
    
    def update_trailing_stop(self, current_price):
        """Update trailing stop if conditions are met"""
        if not TRAILING_STOP_ENABLED:
            return False
        
        if self.position_type == 'buy':
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            profit_pips = (current_price - self.entry_price) / self.point
            
            if profit_pips >= TRAILING_START_PIPS:
                new_sl = current_price - (TRAILING_STOP_DISTANCE_PIPS * self.point)
                
                if new_sl > self.sl:
                    self.sl = new_sl
                    self.trailing_active = True
                    return True
        
        else:  # sell position
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            
            profit_pips = (self.entry_price - current_price) / self.point
            
            if profit_pips >= TRAILING_START_PIPS:
                new_sl = current_price + (TRAILING_STOP_DISTANCE_PIPS * self.point)
                
                if self.sl == 0 or new_sl < self.sl:
                    self.sl = new_sl
                    self.trailing_active = True
                    return True
        
        return False
    
    def check_exit(self, current_high, current_low):
        """Check if position should be closed (TP or SL hit)"""
        if self.position_type == 'buy':
            if current_high >= self.tp:
                return 'tp', self.tp
            elif current_low <= self.sl:
                return 'sl', self.sl
        else:  # sell
            if current_low <= self.tp:
                return 'tp', self.tp
            elif current_high >= self.sl:
                return 'sl', self.sl
        
        return None, None
    
    def calculate_pnl(self, exit_price):
        """Calculate P&L for the position"""
        if self.position_type == 'buy':
            pnl = (exit_price - self.entry_price) * (self.lot_size * 100000) / self.point
        else:
            pnl = (self.entry_price - exit_price) * (self.lot_size * 100000) / self.point
        
        return pnl


def backtest_strategy(symbol, start_date, end_date):
    """Backtest the RSI-MACD-Stochastic strategy with trailing stops"""
    
    # Get historical data
    df = get_historical_data(symbol, start_date, end_date, TIMEFRAME)
    
    if df is None or len(df) < max(macd_slow, stoch_k_period, rsi_period) + 50:
        return None
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'].values, rsi_period)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(
        df['close'].values, macd_fast, macd_slow, macd_signal
    )
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(
        df['high'].values, df['low'].values, df['close'].values,
        stoch_k_period, stoch_d_period, stoch_slowing
    )
    
    # Generate signals
    df['signal'] = 0
    
    # Buy signal: Stochastic %K < 30 (oversold) + RSI > 50 + MACD uptrend
    buy_condition = (
        (df['stoch_k'] < 30) & 
        (df['rsi'] > 50) & 
        (df['macd_line'] > df['macd_signal']) & 
        (df['macd_hist'] > 0)
    )
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell signal: Stochastic %K > 70 (overbought) + RSI < 50 + MACD downtrend
    sell_condition = (
        (df['stoch_k'] > 70) & 
        (df['rsi'] < 50) & 
        (df['macd_line'] < df['macd_signal']) & 
        (df['macd_hist'] < 0)
    )
    df.loc[sell_condition, 'signal'] = -1
    
    # Remove NaN values
    df = df.dropna()
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Cannot get symbol info for {symbol}")
        return None
    
    point = symbol_info.point
    
    # Simulation variables
    trades = []
    balance = initial_balance
    balance_history = [balance]
    open_positions = []
    
    # Simulate trading
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        current_high = current_row['high']
        current_low = current_row['low']
        current_time = current_row['time']
        current_signal = current_row['signal']
        
        # Update and check existing positions
        positions_to_close = []
        
        for pos in open_positions:
            # Update trailing stop
            pos.update_trailing_stop(current_price)
            
            # Check if TP or SL hit during this candle
            exit_reason, exit_price = pos.check_exit(current_high, current_low)
            
            if exit_reason:
                pnl = pos.calculate_pnl(exit_price)
                balance += pnl
                
                trades.append({
                    'symbol': symbol,
                    'type': pos.position_type,
                    'entry_time': pos.entry_time,
                    'exit_time': current_time,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'lot_size': pos.lot_size,
                    'pnl': pnl,
                    'result': 'win' if pnl > 0 else 'loss',
                    'exit_reason': exit_reason,
                    'trailing_used': pos.trailing_active
                })
                
                positions_to_close.append(pos)
        
        # Remove closed positions
        for pos in positions_to_close:
            open_positions.remove(pos)
        
        # Open new position if signal and no existing position for this symbol
        if current_signal != 0:
            has_position = any(pos.symbol == symbol for pos in open_positions)
            
            if not has_position:
                lot_size = calculate_lot_size(balance, risk_per_trade, stop_loss_pips)
                
                if current_signal == 1:  # Buy
                    entry_price = current_price
                    sl = entry_price - (stop_loss_pips * point)
                    tp = entry_price + (take_profit_pips * point)
                    
                    position = Position(symbol, 'buy', entry_price, current_time, 
                                      lot_size, sl, tp, point)
                    open_positions.append(position)
                
                elif current_signal == -1:  # Sell
                    entry_price = current_price
                    sl = entry_price + (stop_loss_pips * point)
                    tp = entry_price - (take_profit_pips * point)
                    
                    position = Position(symbol, 'sell', entry_price, current_time,
                                      lot_size, sl, tp, point)
                    open_positions.append(position)
        
        balance_history.append(balance)
    
    # Close any remaining open positions at the end
    for pos in open_positions:
        exit_price = df.iloc[-1]['close']
        pnl = pos.calculate_pnl(exit_price)
        balance += pnl
        
        trades.append({
            'symbol': symbol,
            'type': pos.position_type,
            'entry_time': pos.entry_time,
            'exit_time': df.iloc[-1]['time'],
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'lot_size': pos.lot_size,
            'pnl': pnl,
            'result': 'win' if pnl > 0 else 'loss',
            'exit_reason': 'end_of_test',
            'trailing_used': pos.trailing_active
        })
    
    return {
        'symbol': symbol,
        'trades': trades,
        'final_balance': balance,
        'balance_history': balance_history,
        'df': df
    }


def calculate_metrics(results):
    """Calculate comprehensive performance metrics"""
    
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
    
    # Trailing stop stats
    trailing_trades = len(trades_df[trades_df['trailing_used'] == True])
    trailing_wins = len(trades_df[(trades_df['trailing_used'] == True) & (trades_df['result'] == 'win')])
    
    # Calculate drawdown
    balance_history = results[0]['balance_history'] if results else [initial_balance]
    cumulative_balance = pd.Series(balance_history)
    running_max = cumulative_balance.cummax()
    drawdown = (cumulative_balance - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Return on investment
    final_balance = results[0]['final_balance'] if results else initial_balance
    roi = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Risk-Reward Ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
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
        'Risk-Reward Ratio': round(risk_reward, 2),
        'Expectancy': round(expectancy, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'ROI (%)': round(roi, 2),
        'Initial Balance': initial_balance,
        'Final Balance': round(final_balance, 2),
        'Trailing Stop Trades': trailing_trades,
        'Trailing Stop Wins': trailing_wins,
    }
    
    return metrics, trades_df


def plot_results(results, metrics):
    """Plot backtest results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'RSI-MACD-Stochastic Bot Backtest Results (Trailing: {TRAILING_STOP_ENABLED})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    if results[0] and results[0]['balance_history']:
        axes[0, 0].plot(results[0]['balance_history'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=initial_balance, color='red', linestyle='--', 
                          label='Initial Balance', linewidth=1)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Time Period (5-min bars)')
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
        axes[0, 1].hist([wins, losses], bins=30, label=['Wins', 'Losses'], 
                       color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Profit/Loss Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # 3. Metrics Table
    axes[0, 2].axis('off')
    metrics_text = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    axes[0, 2].text(0.1, 0.5, metrics_text, fontsize=9, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[0, 2].set_title('Performance Metrics', fontweight='bold')
    
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
        
        axes[1, 0].bar(x - width/2, wins_list, width, label='Wins', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, losses_list, width, label='Losses', color='red', alpha=0.7)
        axes[1, 0].set_title('Wins vs Losses by Symbol', fontweight='bold')
        axes[1, 0].set_xlabel('Symbol')
        axes[1, 0].set_ylabel('Number of Trades')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(symbols_list, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative P&L over time
    if all_pnls:
        cumulative_pnl = np.cumsum(all_pnls)
        axes[1, 1].plot(cumulative_pnl, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Cumulative P&L', fontweight='bold')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative P&L ($)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                                alpha=0.2, color='purple')
    
    # 6. Exit Reasons Pie Chart
    all_trades_list = []
    for result in results:
        if result and result['trades']:
            all_trades_list.extend(result['trades'])
    
    if all_trades_list:
        exit_reasons = {}
        for trade in all_trades_list:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        labels = list(exit_reasons.keys())
        sizes = list(exit_reasons.values())
        colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen']
        
        axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 2].set_title('Exit Reasons Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('backtest_results_stoch_macd.png', dpi=300, bbox_inches='tight')
    print("\n Chart saved as 'backtest_results_stoch_macd.png'")
    plt.show()


# --------- Main Backtest Execution ---------
def run_backtest():
    print("=" * 80)
    print(" RSI-MACD-STOCHASTIC STRATEGY BACKTEST WITH TRAILING STOPS")
    print("=" * 80)
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timeframe: 5 Minutes (M5)")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"\nIndicators:")
    print(f"  - RSI Period: {rsi_period}")
    print(f"  - MACD: Fast={macd_fast}, Slow={macd_slow}, Signal={macd_signal}")
    print(f"  - Stochastic: K={stoch_k_period}, D={stoch_d_period}, Slowing={stoch_slowing}")
    print(f"\nSignal Logic:")
    print(f"  - BUY: Stoch %K < 30 + RSI > 50 + MACD uptrend")
    print(f"  - SELL: Stoch %K > 70 + RSI < 50 + MACD downtrend")
    print(f"\n Trailing Stop: {'ENABLED' if TRAILING_STOP_ENABLED else 'DISABLED'}")
    if TRAILING_STOP_ENABLED:
        print(f"   Start: {TRAILING_START_PIPS} pips | Distance: {TRAILING_STOP_DISTANCE_PIPS} pips")
    print("=" * 80)
    
    results = []
    
    for symbol in symbols:
        print(f"\n Backtesting {symbol}...")
        result = backtest_strategy(symbol, START_DATE, END_DATE)
        if result:
            results.append(result)
            trailing_count = sum(1 for t in result['trades'] if t.get('trailing_used', False))
            print(f" {symbol}: {len(result['trades'])} trades ({trailing_count} with trailing)")
        else:
            print(f" {symbol}: Backtest failed")
    
    if not results:
        print("\n No results to display")
        return
    
    print("\n" + "=" * 80)
    print("  CALCULATING PERFORMANCE METRICS...")
    print("=" * 80)
    
    metrics, trades_df = calculate_metrics(results)
    
    if metrics:
        print("\n BACKTEST RESULTS:\n")
        for key, value in metrics.items():
            print(f"{key:.<50} {value}")
        
        # Save detailed trade log
        if trades_df is not None and not trades_df.empty:
            trades_df.to_csv('backtest_trades_stoch_macd_detailed.csv', index=False)
            print(f"\n Detailed trade log saved as 'backtest_trades_stoch_macd_detailed.csv'")
        
        # Additional Analysis
        print(f"\n" + "=" * 80)
        print("  ADDITIONAL STATISTICS")
        print("=" * 80)
        
        # Trades per symbol
        print("\n Trades per Symbol:")
        symbol_trades = trades_df.groupby('symbol').size()
        for sym, count in symbol_trades.items():
            print(f"  {sym}: {count} trades")
        
        # Win rate per symbol
        print("\n Win Rate per Symbol:")
        for sym in symbol_trades.index:
            sym_trades = trades_df[trades_df['symbol'] == sym]
            sym_wins = len(sym_trades[sym_trades['result'] == 'win'])
            sym_wr = (sym_wins / len(sym_trades)) * 100 if len(sym_trades) > 0 else 0
            print(f"  {sym}: {sym_wr:.2f}%")
        
        # Plot results
        print("\n Generating charts...")
        plot_results(results, metrics)
    
    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_backtest()
    mt5.shutdown()