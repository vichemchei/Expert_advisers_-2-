import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pytz


# --------- Initialize MT5 ---------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print("MT5 initialization failed:", mt5.last_error())
    quit()


# --------- Backtest Parameters ---------
symbols = ["XAUUSD","XAUEUR"]
initial_balance = 1000
risk_per_trade = 0.2  # 2% risk per trade
stop_loss_pips = 80
take_profit_pips = 100
rsi_period = 14
ma_period = 50

# Realistic Trading Costs
SPREAD_PIPS = 3  # Typical gold spread
COMMISSION_PER_LOT = 0  # Set if broker charges commission
SLIPPAGE_PIPS = 1  # Average slippage

# Trailing Stop Parameters
ENABLE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_PIPS = 30  # Start trailing after 30 pips profit
TRAILING_STOP_DISTANCE_PIPS = 20    # Keep stop 20 pips behind current price

# Backtest Settings
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 9, 1)
TIMEFRAME = mt5.TIMEFRAME_M15


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


def calculate_ma(prices, period=50):
    """Calculate Moving Average"""
    return pd.Series(prices).rolling(window=period).mean()


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


def get_pip_value(symbol_info, lot_size):
    """
    Calculate correct pip value for given lot size
    For gold: 1 pip = 0.1 price movement
    """
    # For XAUUSD specifically
    # 1 standard lot (1.0) = $10 per pip
    # So 0.01 lot = $0.10 per pip
    
    # Get contract size (usually 100 for gold)
    contract_size = symbol_info.trade_contract_size
    
    # For gold, 1 pip = 0.1 movement
    # Pip value = (0.1 / exchange_rate) * contract_size * lot_size
    # For gold traded in USD: exchange_rate = 1
    
    pip_value = (0.1 * contract_size * lot_size) / 100
    
    return pip_value


def calculate_lot_size(balance, risk_pct, sl_pips, symbol_info):
    """
    Calculate position size based on risk - CORRECTED
    """
    # For gold: 1 standard lot (1.0) = $10 per pip
    # So pip value per lot = $10
    pip_value_per_standard_lot = 10.0
    
    # Calculate risk amount in dollars
    risk_amount = balance * risk_pct
    
    # Calculate lot size needed
    # If risking $20 on 100 pips: lot = 20 / (100 * 10) = 0.02 lots
    lot = risk_amount / (sl_pips * pip_value_per_standard_lot)
    
    # Apply constraints
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    
    # Round to volume step (usually 0.01)
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    
    return round(lot, 2)


def calculate_trade_costs(lot_size, symbol_info):
    """Calculate realistic trading costs"""
    # Spread cost
    spread_cost = SPREAD_PIPS * get_pip_value(symbol_info, lot_size)
    
    # Commission
    commission = COMMISSION_PER_LOT * lot_size
    
    # Slippage cost (average)
    slippage_cost = SLIPPAGE_PIPS * get_pip_value(symbol_info, lot_size)
    
    return spread_cost + commission + slippage_cost


class Position:
    """Class to track open positions"""
    def __init__(self, symbol, position_type, entry_price, entry_time, lot_size, sl, tp, point, symbol_info):
        self.symbol = symbol
        self.position_type = position_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.lot_size = lot_size
        self.sl = sl
        self.tp = tp
        self.point = point
        self.symbol_info = symbol_info
    
    def check_exit(self, current_high, current_low):
        """Check if position should be closed (TP or SL hit)"""
        if self.position_type == 'buy':
            if current_high >= self.tp:
                return 'tp'
            elif current_low <= self.sl:
                return 'sl'
        else:  # sell
            if current_low <= self.tp:
                return 'tp'
            elif current_high >= self.sl:
                return 'sl'
        
        return None
    
    def calculate_pnl(self, exit_price):
        """
        FIXED: Calculate P&L using correct pip value calculation
        """
        # Calculate pips gained/lost
        if self.position_type == 'buy':
            pips_gained = (exit_price - self.entry_price) / self.point
        else:
            pips_gained = (self.entry_price - exit_price) / self.point
        
        # Get correct pip value for this lot size
        pip_value = get_pip_value(self.symbol_info, self.lot_size)
        
        # Calculate P&L
        pnl = pips_gained * pip_value
        
        return pnl


def backtest_strategy(symbol, start_date, end_date):
    """Backtest the RSI-MA strategy"""
    
    # Get symbol info first
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Cannot get symbol info for {symbol}")
        return None
    
    point = symbol_info.point
    
    # Get historical data
    df = get_historical_data(symbol, start_date, end_date, TIMEFRAME)
    
    if df is None or len(df) < ma_period + rsi_period:
        return None
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'].values, rsi_period)
    df['ma'] = calculate_ma(df['close'].values, ma_period)
    
    # Generate signals
    df['signal'] = 0
    
    # Buy signal: RSI > 50 AND price > MA
    df.loc[(df['rsi'] > 50) & (df['close'] > df['ma']), 'signal'] = 1
    
    # Sell signal: RSI < 50 AND price < MA
    df.loc[(df['rsi'] < 50) & (df['close'] < df['ma']), 'signal'] = -1
    
    # Remove NaN values
    df = df.dropna()
    
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
        
        # Check existing positions
        positions_to_close = []
        
        for pos in open_positions:
            exit_reason = pos.check_exit(current_high, current_low)
            
            if exit_reason:
                # Determine exit price with slippage
                if exit_reason == 'tp':
                    if pos.position_type == 'buy':
                        exit_price = pos.tp - (SLIPPAGE_PIPS * point)
                    else:
                        exit_price = pos.tp + (SLIPPAGE_PIPS * point)
                else:  # sl
                    if pos.position_type == 'buy':
                        exit_price = pos.sl - (SLIPPAGE_PIPS * point)
                    else:
                        exit_price = pos.sl + (SLIPPAGE_PIPS * point)
                
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
                    'exit_reason': exit_reason
                })
                
                positions_to_close.append(pos)
        
        # Remove closed positions
        for pos in positions_to_close:
            open_positions.remove(pos)
        
        # Open new position if signal and no existing position for this symbol
        if current_signal != 0:
            has_position = any(pos.symbol == symbol for pos in open_positions)
            
            if not has_position:
                lot_size = calculate_lot_size(balance, risk_per_trade, stop_loss_pips, symbol_info)
                
                # Deduct trading costs
                trade_costs = calculate_trade_costs(lot_size, symbol_info)
                balance -= trade_costs
                
                if current_signal == 1:  # Buy
                    # Account for spread on entry
                    entry_price = current_price + (SPREAD_PIPS * point)
                    sl = entry_price - (stop_loss_pips * point)
                    tp = entry_price + (take_profit_pips * point)
                    
                    position = Position(symbol, 'buy', entry_price, current_time, 
                                      lot_size, sl, tp, point, symbol_info)
                    open_positions.append(position)
                
                elif current_signal == -1:  # Sell
                    # Account for spread on entry
                    entry_price = current_price - (SPREAD_PIPS * point)
                    sl = entry_price + (stop_loss_pips * point)
                    tp = entry_price - (take_profit_pips * point)
                    
                    position = Position(symbol, 'sell', entry_price, current_time,
                                      lot_size, sl, tp, point, symbol_info)
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
    """Calculate comprehensive performance metrics"""
    
    all_trades = []
    
    # Merge all trades
    for result in results:
        if result and result['trades']:
            all_trades.extend(result['trades'])
    
    if not all_trades:
        print("No trades to analyze")
        return None, None, None
    
    # Sort trades by time to create proper equity curve
    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values('exit_time')
    
    # Calculate cumulative balance
    balance = initial_balance
    balance_history = [balance]
    for _, trade in trades_df.iterrows():
        balance += trade['pnl']
        balance_history.append(balance)
    
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
    cumulative_balance = pd.Series(balance_history)
    running_max = cumulative_balance.cummax()
    drawdown = (cumulative_balance - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Return on investment
    final_balance = balance_history[-1]
    roi = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Exit statistics
    tp_exits = len(trades_df[trades_df['exit_reason'] == 'tp'])
    sl_exits = len(trades_df[trades_df['exit_reason'] == 'sl'])
    
    # Risk-reward ratio
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Sharpe Ratio
    returns = trades_df['pnl'] / initial_balance
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    metrics = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Net Profit ($)': round(net_profit, 2),
        'Total Profit ($)': round(total_profit, 2),
        'Total Loss ($)': round(total_loss, 2),
        'Profit Factor': round(profit_factor, 2),
        'Average Win ($)': round(avg_win, 2),
        'Average Loss ($)': round(avg_loss, 2),
        'Avg Risk:Reward': round(avg_rr, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'ROI (%)': round(roi, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Initial Balance ($)': initial_balance,
        'Final Balance ($)': round(final_balance, 2),
        'TP Exits': tp_exits,
        'SL Exits': sl_exits
    }
    
    return metrics, trades_df, balance_history


def plot_results(results, metrics, balance_history):
    """Plot backtest results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Multi-Symbol RSI-MA Bot - Realistic Backtest Results', 
                 fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    if balance_history:
        axes[0, 0].plot(balance_history, linewidth=2, color='blue')
        axes[0, 0].axhline(y=initial_balance, color='red', linestyle='--', 
                          label='Initial Balance', linewidth=1)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Win/Loss Distribution
    all_pnls = []
    for result in results:
        if result and result['trades']:
            all_pnls.extend([trade['pnl'] for trade in result['trades']])
    
    if all_pnls:
        wins = [pnl for pnl in all_pnls if pnl > 0]
        losses = [pnl for pnl in all_pnls if pnl < 0]
        axes[0, 1].hist([wins, losses], bins=20, label=['Wins', 'Losses'], 
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
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=9, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
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
    plt.savefig('backtest_multi_symbol_realistic.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'backtest_multi_symbol_realistic.png'")
    plt.show()


# --------- Main Backtest Execution ---------
def run_backtest():
    print("=" * 70)
    print("RSI-MA STRATEGY BACKTEST - REALISTIC VERSION")
    print("=" * 70)
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"RSI Period: {rsi_period} | MA Period: {ma_period}")
    print(f"\nTrading Costs:")
    print(f"  - Spread: {SPREAD_PIPS} pips")
    print(f"  - Slippage: {SLIPPAGE_PIPS} pips")
    print(f"  - Commission: ${COMMISSION_PER_LOT} per lot")
    print("=" * 70)
    
    results = []
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        result = backtest_strategy(symbol, START_DATE, END_DATE)
        if result:
            results.append(result)
            print(f"{symbol}: {len(result['trades'])} trades")
        else:
            print(f"{symbol}: Backtest failed")
    
    if not results:
        print("\nNo results to display")
        return
    
    print("\n" + "=" * 70)
    print("CALCULATING PERFORMANCE METRICS...")
    print("=" * 70)
    
    metrics, trades_df, balance_history = calculate_metrics(results)
    
    if metrics:
        print("\nBACKTEST RESULTS:\n")
        for key, value in metrics.items():
            print(f"{key:.<50} {value}")
        
        # Save detailed trade log with error handling
        if trades_df is not None and not trades_df.empty:
            try:
                trades_df.to_csv('backtest_realistic_trades.csv', index=False)
                print(f"\nDetailed trade log saved as 'backtest_realistic_trades.csv'")
            except PermissionError:
                # Try alternative filename if file is open
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                alt_filename = f'backtest_realistic_trades_{timestamp}.csv'
                trades_df.to_csv(alt_filename, index=False)
                print(f"\nOriginal file was locked. Saved as '{alt_filename}' instead")
            except Exception as e:
                print(f"\nWarning: Could not save trade log: {e}")
        
        # Plot results
        print("\nGenerating charts...")
        plot_results(results, metrics, balance_history)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
    mt5.shutdown()