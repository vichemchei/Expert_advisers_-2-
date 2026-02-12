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


# Parameters
symbols = ["XAUUSD","XAUEUR"]
initial_balance = 1000
risk_per_trade = 0.20  # 2% risk per trade
stop_loss_pips = 80
take_profit_pips = 100
rsi_period = 14
ema_period = 200
stoch_k_period = 14
stoch_d_period = 3
stoch_slowing = 3

# Realistic Trading Costs
SPREAD_PIPS = 3  # Typical gold spread (adjust based on your broker)
COMMISSION_PER_LOT = 0  # Set if your broker charges commission
SLIPPAGE_PIPS = 1  # Average slippage on entry/exit

# Trailing Stop Parameters
ENABLE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_PIPS = 30  # Start trailing after 30 pips profit
TRAILING_STOP_DISTANCE_PIPS = 20    # Keep stop 20 pips behind current price

# Backtest Settings
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 9, 1)
TIMEFRAME = mt5.TIMEFRAME_M15


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


def get_historical_data(symbol, start_date, end_date, timeframe):
    print(f"Fetching data for {symbol}...")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def get_pip_value(symbol_info, lot_size):
    """
    Calculate the correct pip value for a given lot size
    For gold: 1 pip = $0.10 movement in price
    """
    # For gold (XAUUSD), 1 pip = 0.1 price movement
    # Standard lot (1.0) = $10 per pip
    # Mini lot (0.1) = $1 per pip
    # Micro lot (0.01) = $0.10 per pip
    
    tick_value = symbol_info.trade_tick_value  # Value of 1 tick
    tick_size = symbol_info.trade_tick_size    # Size of 1 tick
    
    # Calculate pip value per 1.0 lot
    pip_value_per_standard_lot = tick_value * (0.1 / tick_size)
    
    # Scale by actual lot size
    return pip_value_per_standard_lot * lot_size


def calculate_lot_size(balance, risk_pct, sl_pips, symbol_info):
    """
    Calculate proper lot size based on risk management
    """
    # Get pip value for 1 standard lot
    contract_size = symbol_info.trade_contract_size
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    
    # For gold: pip value per 1.0 lot
    pip_value_per_lot = tick_value * (0.1 / tick_size)
    
    # Calculate risk amount in dollars
    risk_amount = balance * risk_pct
    
    # Calculate lot size needed
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    
    # Apply constraints
    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    
    # Round to volume step
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    
    return round(lot, 2)


def calculate_trade_costs(lot_size, symbol_info):
    """
    Calculate realistic trading costs
    """
    # Spread cost
    spread_cost = SPREAD_PIPS * get_pip_value(symbol_info, lot_size)
    
    # Commission
    commission = COMMISSION_PER_LOT * lot_size
    
    # Slippage cost (average)
    slippage_cost = SLIPPAGE_PIPS * get_pip_value(symbol_info, lot_size)
    
    return spread_cost + commission + slippage_cost


def backtest_rsi_stoch_ema_strategy(symbol, start_date, end_date):
    # Get symbol info first
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Cannot get symbol info for {symbol}")
        return None
    
    # Get historical data
    df = get_historical_data(symbol, start_date, end_date, TIMEFRAME)
    
    if df is None:
        return None
    
    data_needed = max(ema_period, stoch_k_period, rsi_period) + 50
    
    if len(df) < data_needed:
        print(f"Not enough data for {symbol}")
        return None
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'].values, rsi_period)
    df['ema200'] = calculate_ema(df['close'].values, ema_period)
    
    stoch_k, stoch_d = calculate_stochastic(
        df['high'].values, df['low'].values, df['close'].values,
        stoch_k_period, stoch_d_period, stoch_slowing
    )
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    # Generate signals
    df['signal'] = 0
    
    # BUY Signal: Stoch %K < 20 AND RSI > 50 AND Price > EMA200
    buy_condition = (df['stoch_k'] < 20) & (df['rsi'] > 50) & (df['close'] > df['ema200'])
    
    # SELL Signal: Stoch %K > 80 AND RSI < 50 AND Price < EMA200
    sell_condition = (df['stoch_k'] > 80) & (df['rsi'] < 50) & (df['close'] < df['ema200'])
    
    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1
    
    # Remove NaN values
    df = df.dropna()
    
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
    trailing_activated = False
    highest_price_since_entry = 0
    lowest_price_since_entry = 0
    
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        current_high = current_row['high']
        current_low = current_row['low']
        current_time = current_row['time']
        current_signal = current_row['signal']
        
        # Update trailing stop logic
        if in_position and ENABLE_TRAILING_STOP:
            if position_type == 'buy':
                if current_high > highest_price_since_entry:
                    highest_price_since_entry = current_high
                
                profit_pips = (highest_price_since_entry - entry_price) / point
                
                if profit_pips >= TRAILING_STOP_ACTIVATION_PIPS and not trailing_activated:
                    trailing_activated = True
                
                if trailing_activated:
                    new_stop = highest_price_since_entry - (TRAILING_STOP_DISTANCE_PIPS * point)
                    if new_stop > stop_loss:
                        stop_loss = new_stop
            
            elif position_type == 'sell':
                if current_low < lowest_price_since_entry:
                    lowest_price_since_entry = current_low
                
                profit_pips = (entry_price - lowest_price_since_entry) / point
                
                if profit_pips >= TRAILING_STOP_ACTIVATION_PIPS and not trailing_activated:
                    trailing_activated = True
                
                if trailing_activated:
                    new_stop = lowest_price_since_entry + (TRAILING_STOP_DISTANCE_PIPS * point)
                    if new_stop < stop_loss:
                        stop_loss = new_stop
        
        # Check if we should close position
        if in_position:
            if position_type == 'buy':
                if current_high >= take_profit:
                    # Take profit hit (with slippage)
                    exit_price = take_profit - (SLIPPAGE_PIPS * point)
                    pips_gained = (exit_price - entry_price) / point
                    pip_value = get_pip_value(symbol_info, lot_size_used)
                    pnl = pips_gained * pip_value
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'buy',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win',
                        'exit_reason': 'tp',
                        'trailing_activated': trailing_activated
                    })
                    in_position = False
                    trailing_activated = False
                elif current_low <= stop_loss:
                    # Stop loss hit (with slippage)
                    exit_price = stop_loss - (SLIPPAGE_PIPS * point)
                    pips_gained = (exit_price - entry_price) / point
                    pip_value = get_pip_value(symbol_info, lot_size_used)
                    pnl = pips_gained * pip_value
                    balance += pnl
                    exit_reason = 'trailing_sl' if trailing_activated else 'sl'
                    trades.append({
                        'symbol': symbol,
                        'type': 'buy',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win' if pnl > 0 else 'loss',
                        'exit_reason': exit_reason,
                        'trailing_activated': trailing_activated
                    })
                    in_position = False
                    trailing_activated = False
            
            elif position_type == 'sell':
                if current_low <= take_profit:
                    # Take profit hit (with slippage)
                    exit_price = take_profit + (SLIPPAGE_PIPS * point)
                    pips_gained = (entry_price - exit_price) / point
                    pip_value = get_pip_value(symbol_info, lot_size_used)
                    pnl = pips_gained * pip_value
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'type': 'sell',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win',
                        'exit_reason': 'tp',
                        'trailing_activated': trailing_activated
                    })
                    in_position = False
                    trailing_activated = False
                elif current_high >= stop_loss:
                    # Stop loss hit (with slippage)
                    exit_price = stop_loss + (SLIPPAGE_PIPS * point)
                    pips_gained = (entry_price - exit_price) / point
                    pip_value = get_pip_value(symbol_info, lot_size_used)
                    pnl = pips_gained * pip_value
                    balance += pnl
                    exit_reason = 'trailing_sl' if trailing_activated else 'sl'
                    trades.append({
                        'symbol': symbol,
                        'type': 'sell',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'lot_size': lot_size_used,
                        'pnl': pnl,
                        'result': 'win' if pnl > 0 else 'loss',
                        'exit_reason': exit_reason,
                        'trailing_activated': trailing_activated
                    })
                    in_position = False
                    trailing_activated = False
        
        # Open new position if signal and not in position
        if not in_position and current_signal != 0:
            lot_size_used = calculate_lot_size(balance, risk_per_trade, stop_loss_pips, symbol_info)
            
            # Apply trading costs
            trade_costs = calculate_trade_costs(lot_size_used, symbol_info)
            balance -= trade_costs  # Deduct costs immediately
            
            entry_price = current_price
            entry_time = current_time
            trailing_activated = False
            
            if current_signal == 1:  # Buy
                # Account for spread on entry
                entry_price = current_price + (SPREAD_PIPS * point)
                position_type = 'buy'
                stop_loss = entry_price - (stop_loss_pips * point)
                take_profit = entry_price + (take_profit_pips * point)
                highest_price_since_entry = current_high
                in_position = True
            
            elif current_signal == -1:  # Sell
                # Account for spread on entry
                entry_price = current_price - (SPREAD_PIPS * point)
                position_type = 'sell'
                stop_loss = entry_price + (stop_loss_pips * point)
                take_profit = entry_price - (take_profit_pips * point)
                lowest_price_since_entry = current_low
                in_position = True
        
        balance_history.append(balance)
    
    # Close any remaining open positions
    if in_position:
        exit_price = df.iloc[-1]['close']
        if position_type == 'buy':
            pips_gained = (exit_price - entry_price) / point
        else:
            pips_gained = (entry_price - exit_price) / point
        
        pip_value = get_pip_value(symbol_info, lot_size_used)
        pnl = pips_gained * pip_value
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
            'exit_reason': 'end_of_test',
            'trailing_activated': trailing_activated
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
    trailing_sl_exits = len(trades_df[trades_df['exit_reason'] == 'trailing_sl'])
    
    # Trailing stop stats
    trailing_activated_count = len(trades_df[trades_df['trailing_activated'] == True])
    
    # Risk-reward ratio
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Calculate Sharpe Ratio (simplified, assuming risk-free rate = 0)
    returns = trades_df['pnl'] / initial_balance
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
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
        'Avg Risk:Reward': round(avg_rr, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'ROI (%)': round(roi, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Initial Balance': initial_balance,
        'Final Balance': round(final_balance, 2),
        'TP Exits': tp_exits,
        'SL Exits': sl_exits,
        'Trailing SL Exits': trailing_sl_exits,
        'Trailing Activated': trailing_activated_count
    }
    
    return metrics, trades_df


def plot_results(results, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RSI-Stochastic-EMA Bot - Realistic Backtest Results', 
                 fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    if results[0] and results[0]['balance_history']:
        axes[0, 0].plot(results[0]['balance_history'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=initial_balance, color='red', linestyle='--', 
                          label='Initial Balance', linewidth=1)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Time Period (M5 bars)')
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
    
    # 4. Exit Reasons Breakdown
    if results and results[0]['trades']:
        trades_df = pd.DataFrame(results[0]['trades'])
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        colors_map = {
            'tp': 'green',
            'sl': 'red',
            'trailing_sl': 'orange',
            'end_of_test': 'gray'
        }
        colors = [colors_map.get(reason, 'blue') for reason in exit_reasons.index]
        
        axes[1, 1].bar(exit_reasons.index, exit_reasons.values, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Exit Reasons Breakdown', fontweight='bold')
        axes[1, 1].set_xlabel('Exit Reason')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest_realistic.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'backtest_realistic.png'")
    plt.show()


def run_backtest():
    print("=" * 70)
    print("RSI-STOCHASTIC-EMA BOT - REALISTIC BACKTEST")
    print("=" * 70)
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timeframe: M5 (5-minute charts)")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Stop Loss: {stop_loss_pips} pips | Take Profit: {take_profit_pips} pips")
    print(f"\nTrading Costs:")
    print(f"  - Spread: {SPREAD_PIPS} pips")
    print(f"  - Slippage: {SLIPPAGE_PIPS} pips")
    print(f"  - Commission: ${COMMISSION_PER_LOT} per lot")
    print(f"\nTrailing Stop Settings:")
    print(f"  - Enabled: {ENABLE_TRAILING_STOP}")
    print(f"  - Activation: {TRAILING_STOP_ACTIVATION_PIPS} pips profit")
    print(f"  - Distance: {TRAILING_STOP_DISTANCE_PIPS} pips behind price")
    print(f"\nStrategy Logic:")
    print(f"  BUY: Stoch %K < 20 AND RSI > 50 AND Price > EMA200")
    print(f"  SELL: Stoch %K > 80 AND RSI < 50 AND Price < EMA200")
    print(f"\nIndicators:")
    print(f"  - RSI Period: {rsi_period}")
    print(f"  - EMA Period: {ema_period}")
    print(f"  - Stochastic: K={stoch_k_period}, D={stoch_d_period}, Slowing={stoch_slowing}")
    print("=" * 70)
    
    results = []
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        result = backtest_rsi_stoch_ema_strategy(symbol, START_DATE, END_DATE)
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
            trades_df.to_csv('backtest_realistic_trades.csv', index=False)
            print(f"\nDetailed trade log saved as 'backtest_realistic_trades.csv'")
        
        # Plot results
        print("\nGenerating charts...")
        plot_results(results, metrics)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
    mt5.shutdown()