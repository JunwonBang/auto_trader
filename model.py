import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class RuleBasedTrader:
    def __init__(self, initial_balance=100, leverage=10, trading_fee=0.0006):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.trading_fee = trading_fee
        self.quantity = 0.0001
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.equity_curve = []
        
    def calculate_indicators(self, df):
        """Calculate enhanced technical indicators with improved signals"""
        # 1. Trend Indicators
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # 2. Momentum Indicators
        # RSI with optimized parameters
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-9)  # Add small value to avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume-Weighted MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 3. Volatility Indicators
        # ATR with dynamic period
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.ewm(alpha=1/14, adjust=False).mean()
        
        # SuperTrend
        df['upper_band'] = (df['high'] + df['low']) / 2 + (2 * df['atr'])
        df['lower_band'] = (df['high'] + df['low']) / 2 - (2 * df['atr'])
        df['in_uptrend'] = True
        
        for current in range(1, len(df)):
            previous = current - 1
            
            if df['close'].iloc[current] > df['upper_band'].iloc[previous]:
                df['in_uptrend'].iloc[current] = True
            elif df['close'].iloc[current] < df['lower_band'].iloc[previous]:
                df['in_uptrend'].iloc[current] = False
            else:
                df['in_uptrend'].iloc[current] = df['in_uptrend'].iloc[previous]
                
                if df['in_uptrend'].iloc[current] and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]:
                    df['lower_band'].iloc[current] = df['lower_band'].iloc[previous]
                
                if not df['in_uptrend'].iloc[current] and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]:
                    df['upper_band'].iloc[current] = df['upper_band'].iloc[previous]
        
        # 4. Volume Analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
        
        # 5. Price Action
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['upper_bb'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
        df['lower_bb'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)
        
        # Clean up and forward fill
        df = df.ffill().dropna()
        return df
        
    def calculate_signals(self, df):
        """Generate enhanced trading signals with improved entry/exit conditions"""
        # Calculate indicators first
        df = self.calculate_indicators(df)
        
        # Initialize signals and position management columns
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['trailing_stop'] = np.nan
        df['position_size'] = 0.0
        
        # Strategy parameters
        params = {
            'rsi_overbought': 70,  # More conservative RSI levels
            'rsi_oversold': 30,
            'adx_threshold': 25,    # Minimum trend strength
            'atr_multiplier': 1.5,  # For stop loss/take profit
            'risk_reward_ratio': 2.5,  # Increased reward:risk ratio
            'min_volume_ratio': 1.2,  # Minimum volume ratio for confirmation
            'trend_confirmation_bars': 3  # Number of bars for trend confirmation
        }
        
        # Generate signals with improved conditions
        for i in range(2, len(df)):
            prev2 = df.iloc[i-2]
            prev = df.iloc[i-1]
            current = df.iloc[i]
            
            # Skip if we don't have enough data
            required_cols = ['rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'volume_ratio']
            if any(pd.isna(df[col].iloc[i]) for col in required_cols):
                continue
                
            # Trend confirmation (price above/below key moving averages)
            uptrend = current['close'] > current['ema_50'] > current['ema_200']
            downtrend = current['close'] < current['ema_50'] < current['ema_200']
            
            # Volume confirmation
            volume_ok = current['volume_ratio'] > params['min_volume_ratio']
            
            # Long conditions
            long_conditions = [
                current['rsi'] < params['rsi_oversold'],
                current['macd'] > current['macd_signal'],
                prev['macd'] <= prev['macd_signal'],  # MACD crossover up
                uptrend,  # Price above key MAs
                current['in_uptrend'],  # SuperTrend confirmation
                current['close'] > current['vwap'],  # Price above VWAP
                volume_ok,  # Volume confirmation
                # Trend confirmation (price making higher highs)
                current['close'] > prev['close'] > prev2['close']
            ]
            
            # Short conditions
            short_conditions = [
                current['rsi'] > params['rsi_overbought'],
                current['macd'] < current['macd_signal'],
                prev['macd'] >= prev['macd_signal'],  # MACD crossover down
                downtrend,  # Price below key MAs
                not current['in_uptrend'],  # SuperTrend confirmation
                current['close'] < current['vwap'],  # Price below VWAP
                volume_ok,  # Volume confirmation
                # Trend confirmation (price making lower lows)
                current['close'] < prev['close'] < prev2['close']
            ]
            
            # Check long entry with multiple confirmations
            if sum(long_conditions) >= len(long_conditions) - 1:  # Allow 1 condition to be false
                # Only enter if we have a clear stop loss level
                atr = current['atr']
                stop_loss = min(current['low'], current['lower_band'])
                risk = current['close'] - stop_loss
                
                if risk > 0:  # Only take trades with valid risk
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = current['close'] + (risk * params['risk_reward_ratio'])
            
            # Check short entry with multiple confirmations
            elif sum(short_conditions) >= len(short_conditions) - 1:  # Allow 1 condition to be false
                # Only enter if we have a clear stop loss level
                atr = current['atr']
                stop_loss = max(current['high'], current['upper_band'])
                risk = stop_loss - current['close']
                
                if risk > 0:  # Only take trades with valid risk
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = current['close'] - (risk * params['risk_reward_ratio'])
                
        return df
    
    def execute_trades(self, df):
        """Execute trades with enhanced position management"""
        for i, row in df.iterrows():
            current_price = row['close']
            
            # Update trailing stop for open positions
            if self.position == 'long' and not pd.isna(row['trailing_stop']):
                # Move trailing stop up if price increases
                new_trailing_stop = current_price - (row['atr'] * 1.5)
                if new_trailing_stop > row['trailing_stop']:
                    df.loc[i, 'trailing_stop'] = new_trailing_stop
                
                # Check if price hit trailing stop
                if current_price <= row['trailing_stop']:
                    self._close_position(current_price, i)
                    continue
            
            # Close position conditions
            if self.position == 'long':
                # Check stop loss, take profit, or exit signal
                if (not pd.isna(row['stop_loss']) and current_price <= row['stop_loss']) or \
                   (not pd.isna(row['take_profit']) and current_price >= row['take_profit']) or \
                   (row['signal'] == -1):
                    self._close_position(current_price, i)
            
            elif self.position == 'short':
                # Check stop loss, take profit, or exit signal
                if (not pd.isna(row['stop_loss']) and current_price >= row['stop_loss']) or \
                   (not pd.isna(row['take_profit']) and current_price <= row['take_profit']) or \
                   (row['signal'] == 1):
                    self._close_position(current_price, i)
            
            # Open new position if no position is open and we have a valid signal
            if self.position is None and row['signal'] != 0:
                # Calculate position size based on ATR and risk per trade (1% of capital)
                risk_per_trade = self.balance * 0.01
                atr = row['atr']
                
                # Skip if we don't have ATR data
                if pd.isna(atr) or atr == 0:
                    continue
                    
                # Calculate position size
                risk_amount = abs(current_price - row['stop_loss'])
                if risk_amount > 0:
                    self.quantity = min(risk_per_trade / risk_amount, self.balance / current_price)
                    self._open_position(row['signal'], current_price, i)
                    
                    # Initialize trailing stop
                    if row['signal'] == 1:  # Long position
                        df.loc[i, 'trailing_stop'] = current_price - (atr * 1.5)
                    else:  # Short position
                        df.loc[i, 'trailing_stop'] = current_price + (atr * 1.5)
            
            self.equity_curve.append(self.balance)
    
    def _open_position(self, signal, price, timestamp):
        """Open a new position"""
        self.position = 'long' if signal == 1 else 'short'
        self.entry_price = price
        fee = price * self.quantity * self.leverage * self.trading_fee
        self.balance -= fee
    
    def _close_position(self, price, timestamp):
        """Close the current position"""
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.quantity * self.leverage
        else:  # short
            pnl = (self.entry_price - price) * self.quantity * self.leverage
        
        fee = price * self.quantity * self.leverage * self.trading_fee
        self.balance += pnl - fee
        
        self.position = None
        self.entry_price = None
    
    def analyze_performance(self):
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate performance metrics
        total_return = (self.balance / self.initial_balance - 1) * 100
        
        # Plot equity curve
        plt.figure(figsize=(12, 8))
        
        plt.plot(self.equity_curve)
        plt.title(f'Equity Curve (Total Return: {total_return:.2f}%)')
        plt.xlabel('Trade #')
        plt.ylabel('Balance ($)')
        plt.grid(True)
        
        # Save the plot
        plot_filename = f'backtest_results/equity_curve_{timestamp}.png'
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"\nPerformance results saved to backtest_results/ with timestamp: {timestamp}")

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('./dataset/data_20250701_20250801.csv')
    
    # Initialize and run trader
    trader = RuleBasedTrader()
    df = trader.calculate_signals(df)
    trader.execute_trades(df)
    
    # Analyze performance
    trader.analyze_performance()