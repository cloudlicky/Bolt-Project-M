"""Paper trading simulator for strategy testing"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class PaperTrading:
    """Class to handle paper trading simulation"""
    
    def __init__(self):
        self.initial_capital = 100000  # ₹1 Lakh default
        self.position_size = 0.02      # 2% per trade default
        
        # Initialize session state for trades
        if 'paper_trades' not in st.session_state:
            st.session_state.paper_trades = []
        if 'portfolio_value' not in st.session_state:
            st.session_state.portfolio_value = self.initial_capital
        if 'active_positions' not in st.session_state:
            st.session_state.active_positions = {}
    
    def render_paper_trading(self, selected_stocks, selected_timeframes, 
                            start_date, end_date, data_fetcher, technical_analysis):
        """Render paper trading interface"""
        
        st.markdown("""
        🧪 **Paper Trading Simulator** lets you test your trading strategies with virtual money. 
        Build custom strategies, simulate trades, and track performance without any financial risk.
        """)
        
        # Tabs for paper trading features
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Strategy Builder", 
            "📊 Current Portfolio", 
            "📈 Performance Analysis",
            "📋 Trade History"
        ])
        
        with tab1:
            self.render_strategy_builder(selected_stocks, selected_timeframes, 
                                       start_date, end_date, data_fetcher, technical_analysis)
        
        with tab2:
            self.render_current_portfolio()
        
        with tab3:
            self.render_performance_analysis()
        
        with tab4:
            self.render_trade_history()
    
    def render_strategy_builder(self, selected_stocks, selected_timeframes, 
                               start_date, end_date, data_fetcher, technical_analysis):
        """Render strategy building interface"""
        
        st.markdown("### 🎯 Custom Strategy Builder")
        
        # Strategy configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📝 Strategy Rules**")
            
            # Entry conditions
            st.markdown("**Entry Conditions (All must be met):**")
            
            entry_conditions = []
            
            # EMA condition
            use_ema = st.checkbox("EMA Crossover", value=True)
            if use_ema:
                ema_condition = st.selectbox(
                    "EMA Condition",
                    ["EMA 10 > EMA 21", "EMA 21 > EMA 50", "EMA 10 crosses above EMA 21"]
                )
                entry_conditions.append(("EMA", ema_condition))
            
            # RSI condition
            use_rsi = st.checkbox("RSI Filter", value=True)
            if use_rsi:
                rsi_operator = st.selectbox("RSI Operator", ["<", ">", "between"])
                if rsi_operator == "between":
                    rsi_low = st.number_input("RSI Low", value=30, min_value=0, max_value=100)
                    rsi_high = st.number_input("RSI High", value=70, min_value=0, max_value=100)
                    rsi_condition = f"RSI between {rsi_low} and {rsi_high}"
                else:
                    rsi_value = st.number_input("RSI Value", value=30 if rsi_operator == "<" else 70, 
                                              min_value=0, max_value=100)
                    rsi_condition = f"RSI {rsi_operator} {rsi_value}"
                entry_conditions.append(("RSI", rsi_condition))
            
            # Volume condition
            use_volume = st.checkbox("Volume Filter", value=True)
            if use_volume:
                volume_multiplier = st.number_input("Volume > Average by", value=1.5, step=0.1)
                volume_condition = f"Volume > {volume_multiplier}x average"
                entry_conditions.append(("Volume", volume_condition))
            
            # MACD condition
            use_macd = st.checkbox("MACD Signal")
            if use_macd:
                macd_condition = st.selectbox(
                    "MACD Condition",
                    ["MACD > Signal", "MACD crosses above Signal", "MACD > 0"]
                )
                entry_conditions.append(("MACD", macd_condition))
        
        with col2:
            st.markdown("**⚙️ Strategy Settings**")
            
            strategy_name = st.text_input("Strategy Name", value="Custom Strategy")
            
            # Risk management
            st.markdown("**Risk Management:**")
            stop_loss_pct = st.number_input("Stop Loss %", value=2.0, step=0.1)
            take_profit_pct = st.number_input("Take Profit %", value=6.0, step=0.1)
            position_size_pct = st.number_input("Position Size %", value=2.0, step=0.1)
            
            # Strategy timeframe
            strategy_timeframe = st.selectbox(
                "Strategy Timeframe",
                selected_timeframes if selected_timeframes else ["1D"]
            )
            
            # Backtesting period
            backtest_days = st.number_input("Backtest Days", value=90, min_value=30, max_value=365)
        
        # Display strategy summary
        if entry_conditions:
            st.markdown("### 📋 Strategy Summary")
            st.markdown(f"**Strategy Name:** {strategy_name}")
            st.markdown(f"**Timeframe:** {strategy_timeframe}")
            st.markdown("**Entry Conditions:**")
            for condition_type, condition in entry_conditions:
                st.write(f"- {condition_type}: {condition}")
            
            st.markdown("**Risk Management:**")
            st.write(f"- Stop Loss: {stop_loss_pct}%")
            st.write(f"- Take Profit: {take_profit_pct}%") 
            st.write(f"- Position Size: {position_size_pct}%")
            
            # Backtest button
            if st.button("🚀 Run Strategy Backtest", type="primary"):
                if not selected_stocks:
                    st.error("Please select at least one stock from the sidebar.")
                    return
                
                self.run_strategy_backtest(
                    strategy_name, entry_conditions, strategy_timeframe,
                    stop_loss_pct, take_profit_pct, position_size_pct,
                    selected_stocks, backtest_days, data_fetcher, technical_analysis
                )
    
    def run_strategy_backtest(self, strategy_name, entry_conditions, timeframe,
                             stop_loss_pct, take_profit_pct, position_size_pct,
                             stocks, backtest_days, data_fetcher, technical_analysis):
        """Run backtest simulation for the custom strategy"""
        
        st.markdown("### 📊 Strategy Backtest Results")
        
        with st.spinner("Running backtest simulation..."):
            backtest_start = datetime.now() - timedelta(days=backtest_days)
            backtest_end = datetime.now()
            
            all_trades = []
            portfolio_value = self.initial_capital
            
            progress_bar = st.progress(0)
            
            for idx, stock in enumerate(stocks):
                # Get historical data
                data = data_fetcher.get_stock_data(stock, timeframe, 
                                                 backtest_start.date(), backtest_end.date())
                
                if data is None or len(data) < 50:
                    continue
                
                # Calculate indicators
                indicators = technical_analysis.calculate_all_indicators(data)
                
                # Simulate trading on this stock
                stock_trades = self.simulate_strategy_on_stock(
                    stock, data, indicators, entry_conditions,
                    stop_loss_pct, take_profit_pct, position_size_pct, portfolio_value
                )
                
                all_trades.extend(stock_trades)
                progress_bar.progress((idx + 1) / len(stocks))
            
            progress_bar.empty()
            
            if all_trades:
                self.display_backtest_results(strategy_name, all_trades, portfolio_value)
            else:
                st.warning("No trades generated by this strategy. Try adjusting the conditions.")
    
    def simulate_strategy_on_stock(self, stock, data, indicators, entry_conditions,
                                  stop_loss_pct, take_profit_pct, position_size_pct, 
                                  initial_capital):
        """Simulate strategy on individual stock"""
        
        trades = []
        position = None
        
        for i in range(50, len(data)):  # Start after indicators are calculated
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Check if we have a position
            if position is None:
                # Check entry conditions
                if self.check_entry_conditions(indicators, i, entry_conditions):
                    # Open position
                    position_value = initial_capital * (position_size_pct / 100)
                    shares = position_value / current_price
                    
                    position = {
                        'stock': stock,
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'shares': shares,
                        'stop_loss': current_price * (1 - stop_loss_pct / 100),
                        'take_profit': current_price * (1 + take_profit_pct / 100),
                        'position_value': position_value
                    }
            
            else:
                # Check exit conditions
                exit_reason = None
                
                # Stop loss hit
                if current_price <= position['stop_loss']:
                    exit_reason = "Stop Loss"
                
                # Take profit hit
                elif current_price >= position['take_profit']:
                    exit_reason = "Take Profit"
                
                # Time-based exit (optional - exit after 10 days)
                elif (current_date - position['entry_date']).days >= 10:
                    exit_reason = "Time Exit"
                
                if exit_reason:
                    # Close position
                    exit_value = position['shares'] * current_price
                    pnl = exit_value - position['position_value']
                    pnl_pct = (pnl / position['position_value']) * 100
                    
                    trade = {
                        'stock': stock,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'days_held': (current_date - position['entry_date']).days
                    }
                    
                    trades.append(trade)
                    position = None
        
        return trades
    
    def check_entry_conditions(self, indicators, index, entry_conditions):
        """Check if all entry conditions are met"""
        
        for condition_type, condition in entry_conditions:
            if condition_type == "EMA":
                if not self.check_ema_condition(indicators, index, condition):
                    return False
            
            elif condition_type == "RSI":
                if not self.check_rsi_condition(indicators, index, condition):
                    return False
            
            elif condition_type == "Volume":
                if not self.check_volume_condition(indicators, index, condition):
                    return False
            
            elif condition_type == "MACD":
                if not self.check_macd_condition(indicators, index, condition):
                    return False
        
        return True
    
    def check_ema_condition(self, indicators, index, condition):
        """Check EMA-based conditions"""
        
        if index < 1:  # Need at least 2 points for crossover
            return False
        
        if 'EMA_10' not in indicators or 'EMA_21' not in indicators:
            return False
        
        ema_10_current = indicators['EMA_10'][index]
        ema_21_current = indicators['EMA_21'][index]
        
        if condition == "EMA 10 > EMA 21":
            return ema_10_current > ema_21_current
        
        elif condition == "EMA 10 crosses above EMA 21":
            ema_10_prev = indicators['EMA_10'][index-1]
            ema_21_prev = indicators['EMA_21'][index-1]
            return (ema_10_prev <= ema_21_prev and ema_10_current > ema_21_current)
        
        elif condition == "EMA 21 > EMA 50":
            if 'EMA_50' not in indicators:
                return False
            ema_50_current = indicators['EMA_50'][index]
            return ema_21_current > ema_50_current
        
        return False
    
    def check_rsi_condition(self, indicators, index, condition):
        """Check RSI-based conditions"""
        
        if 'RSI' not in indicators or index >= len(indicators['RSI']):
            return False
        
        rsi_current = indicators['RSI'][index]
        
        if "between" in condition:
            # Extract values from "RSI between X and Y"
            parts = condition.split()
            low_val = float(parts[2])
            high_val = float(parts[4])
            return low_val <= rsi_current <= high_val
        
        elif "<" in condition:
            threshold = float(condition.split("<")[1].strip())
            return rsi_current < threshold
        
        elif ">" in condition:
            threshold = float(condition.split(">")[1].strip())
            return rsi_current > threshold
        
        return False
    
    def check_volume_condition(self, indicators, index, condition):
        """Check volume-based conditions"""
        
        if 'VOLUME_RATIO' not in indicators or index >= len(indicators['VOLUME_RATIO']):
            return False
        
        volume_ratio = indicators['VOLUME_RATIO'][index]
        
        # Extract multiplier from "Volume > Xx average"
        multiplier = float(condition.split(">")[1].replace("x average", "").strip())
        return volume_ratio > multiplier
    
    def check_macd_condition(self, indicators, index, condition):
        """Check MACD-based conditions"""
        
        if not all(key in indicators for key in ['MACD', 'MACD_signal']):
            return False
        
        if index >= len(indicators['MACD']):
            return False
        
        macd_current = indicators['MACD'][index]
        signal_current = indicators['MACD_signal'][index]
        
        if condition == "MACD > Signal":
            return macd_current > signal_current
        
        elif condition == "MACD crosses above Signal":
            if index < 1:
                return False
            macd_prev = indicators['MACD'][index-1]
            signal_prev = indicators['MACD_signal'][index-1]
            return (macd_prev <= signal_prev and macd_current > signal_current)
        
        elif condition == "MACD > 0":
            return macd_current > 0
        
        return False
    
    def display_backtest_results(self, strategy_name, trades, initial_capital):
        """Display comprehensive backtest results"""
        
        if not trades:
            st.warning("No trades were executed during the backtest period.")
            return
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        total_return_pct = (total_pnl / initial_capital) * 100
        
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        # Display metrics
        st.markdown(f"### 🎯 {strategy_name} - Backtest Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.metric("Total Return", f"₹{total_pnl:,.0f}", delta=f"{total_return_pct:+.2f}%")
            st.metric("Winning Trades", winning_trades)
        
        with col3:
            st.metric("Average Win", f"₹{avg_win:,.0f}")
            st.metric("Losing Trades", losing_trades)
        
        with col4:
            st.metric("Average Loss", f"₹{avg_loss:,.0f}")
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # Performance breakdown
        st.markdown("### 📊 Performance Details")
        
        # Create DataFrame for trades
        df_trades = pd.DataFrame(trades)
        
        # Best and worst trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Best Trades**")
            best_trades = df_trades.nlargest(5, 'pnl_pct')[['stock', 'entry_date', 'pnl_pct', 'exit_reason']]
            best_trades['pnl_pct'] = best_trades['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(best_trades, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**📉 Worst Trades**")
            worst_trades = df_trades.nsmallest(5, 'pnl_pct')[['stock', 'entry_date', 'pnl_pct', 'exit_reason']]
            worst_trades['pnl_pct'] = worst_trades['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(worst_trades, hide_index=True, use_container_width=True)
        
        # Stock-wise performance
        st.markdown("**📈 Stock-wise Performance**")
        stock_performance = df_trades.groupby('stock').agg({
            'pnl': ['sum', 'count'],
            'pnl_pct': 'mean'
        }).round(2)
        
        stock_performance.columns = ['Total PnL', 'Trades', 'Avg Return %']
        stock_performance = stock_performance.sort_values('Total PnL', ascending=False)
        st.dataframe(stock_performance, use_container_width=True)
        
        # Save strategy button
        if st.button("💾 Save Strategy"):
            strategy_data = {
                'name': strategy_name,
                'trades': trades,
                'performance': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_return': total_return_pct,
                    'profit_factor': profit_factor
                },
                'created_date': datetime.now().isoformat()
            }
            
            # Add to session state
            if 'saved_strategies' not in st.session_state:
                st.session_state.saved_strategies = []
            
            st.session_state.saved_strategies.append(strategy_data)
            st.success(f"Strategy '{strategy_name}' saved successfully!")
    
    def render_current_portfolio(self):
        """Render current portfolio overview"""
        
        st.markdown("### 📊 Current Portfolio Overview")
        
        # Portfolio summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"₹{st.session_state.portfolio_value:,.0f}")
        
        with col2:
            pnl = st.session_state.portfolio_value - self.initial_capital
            pnl_pct = (pnl / self.initial_capital) * 100
            st.metric("Total P&L", f"₹{pnl:,.0f}", delta=f"{pnl_pct:+.2f}%")
        
        with col3:
            active_positions = len(st.session_state.active_positions)
            st.metric("Active Positions", active_positions)
        
        with col4:
            total_trades = len(st.session_state.paper_trades)
            st.metric("Total Trades", total_trades)
        
        # Active positions
        if st.session_state.active_positions:
            st.markdown("### 🎯 Active Positions")
            
            positions_data = []
            for symbol, position in st.session_state.active_positions.items():
                positions_data.append({
                    'Stock': symbol,
                    'Entry Price': f"₹{position['entry_price']:.2f}",
                    'Current Price': f"₹{position.get('current_price', 0):.2f}",
                    'Quantity': position['quantity'],
                    'P&L': f"₹{position.get('unrealized_pnl', 0):,.0f}",
                    'P&L %': f"{position.get('unrealized_pnl_pct', 0):+.2f}%"
                })
            
            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, hide_index=True, use_container_width=True)
        
        else:
            st.info("No active positions. Use the Strategy Builder to create and test trading strategies.")
        
        # Manual trading section
        st.markdown("### 🎮 Manual Trading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            manual_stock = st.selectbox("Select Stock", ["RELIANCE", "TCS", "HDFCBANK"])
            manual_action = st.selectbox("Action", ["BUY", "SELL"])
            manual_quantity = st.number_input("Quantity", value=100, min_value=1)
            manual_price = st.number_input("Price per Share", value=100.0, step=0.1)
        
        with col2:
            st.markdown("**Trade Summary**")
            trade_value = manual_quantity * manual_price
            st.write(f"Trade Value: ₹{trade_value:,.2f}")
            
            if st.button("📝 Execute Trade", type="primary"):
                self.execute_manual_trade(manual_stock, manual_action, manual_quantity, manual_price)
    
    def execute_manual_trade(self, stock, action, quantity, price):
        """Execute a manual trade"""
        
        trade_value = quantity * price
        
        if action == "BUY":
            if trade_value <= st.session_state.portfolio_value:
                # Add to active positions
                if stock in st.session_state.active_positions:
                    # Average down/up
                    current_pos = st.session_state.active_positions[stock]
                    total_value = (current_pos['quantity'] * current_pos['entry_price']) + trade_value
                    total_quantity = current_pos['quantity'] + quantity
                    avg_price = total_value / total_quantity
                    
                    st.session_state.active_positions[stock] = {
                        'entry_price': avg_price,
                        'quantity': total_quantity,
                        'current_price': price
                    }
                else:
                    st.session_state.active_positions[stock] = {
                        'entry_price': price,
                        'quantity': quantity,
                        'current_price': price
                    }
                
                st.session_state.portfolio_value -= trade_value
                
                # Record trade
                trade_record = {
                    'date': datetime.now(),
                    'stock': stock,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value
                }
                st.session_state.paper_trades.append(trade_record)
                
                st.success(f"✅ Bought {quantity} shares of {stock} at ₹{price:.2f}")
            else:
                st.error("Insufficient funds for this trade.")
        
        elif action == "SELL":
            if stock in st.session_state.active_positions:
                position = st.session_state.active_positions[stock]
                if quantity <= position['quantity']:
                    # Partial or full sell
                    pnl = quantity * (price - position['entry_price'])
                    
                    if quantity == position['quantity']:
                        # Full sell
                        del st.session_state.active_positions[stock]
                    else:
                        # Partial sell
                        st.session_state.active_positions[stock]['quantity'] -= quantity
                    
                    st.session_state.portfolio_value += trade_value
                    
                    # Record trade
                    trade_record = {
                        'date': datetime.now(),
                        'stock': stock,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'value': trade_value,
                        'pnl': pnl
                    }
                    st.session_state.paper_trades.append(trade_record)
                    
                    st.success(f"✅ Sold {quantity} shares of {stock} at ₹{price:.2f}. P&L: ₹{pnl:+,.0f}")
                else:
                    st.error(f"You only have {position['quantity']} shares of {stock}.")
            else:
                st.error(f"You don't have any position in {stock}.")
    
    def render_performance_analysis(self):
        """Render performance analysis dashboard"""
        
        st.markdown("### 📈 Performance Analysis")
        
        if not st.session_state.paper_trades:
            st.info("No trades executed yet. Start trading to see performance analysis.")
            return
        
        # Convert trades to DataFrame
        df_trades = pd.DataFrame(st.session_state.paper_trades)
        
        # Performance metrics
        total_trades = len(df_trades)
        buy_trades = len(df_trades[df_trades['action'] == 'BUY'])
        sell_trades = len(df_trades[df_trades['action'] == 'SELL'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Buy Trades", buy_trades)
        
        with col3:
            st.metric("Sell Trades", sell_trades)
        
        with col4:
            if 'pnl' in df_trades.columns:
                total_realized_pnl = df_trades['pnl'].sum()
                st.metric("Realized P&L", f"₹{total_realized_pnl:+,.0f}")
        
        # Trade history chart
        if len(df_trades) > 1:
            st.markdown("**📊 Portfolio Value Over Time**")
            
            # Calculate cumulative portfolio value
            portfolio_history = [self.initial_capital]
            current_value = self.initial_capital
            
            for _, trade in df_trades.iterrows():
                if trade['action'] == 'BUY':
                    current_value -= trade['value']
                else:
                    current_value += trade['value']
                    if 'pnl' in trade:
                        current_value += trade['pnl']
                
                portfolio_history.append(current_value)
            
            # Create chart data
            chart_data = pd.DataFrame({
                'Date': [datetime.now() - timedelta(days=len(portfolio_history)-i) 
                        for i in range(len(portfolio_history))],
                'Portfolio Value': portfolio_history
            })
            
            st.line_chart(chart_data.set_index('Date')['Portfolio Value'])
        
        # Stock-wise performance
        if sell_trades > 0:
            st.markdown("**📈 Stock-wise P&L**")
            
            sell_trades_df = df_trades[df_trades['action'] == 'SELL']
            if 'pnl' in sell_trades_df.columns:
                stock_pnl = sell_trades_df.groupby('stock')['pnl'].sum().sort_values(ascending=False)
                
                pnl_data = pd.DataFrame({
                    'Stock': stock_pnl.index,
                    'Realized P&L': [f"₹{pnl:+,.0f}" for pnl in stock_pnl.values]
                })
                
                st.dataframe(pnl_data, hide_index=True, use_container_width=True)
    
    def render_trade_history(self):
        """Render complete trade history"""
        
        st.markdown("### 📋 Complete Trade History")
        
        if not st.session_state.paper_trades:
            st.info("No trades executed yet.")
            return
        
        # Convert to DataFrame and format
        df_trades = pd.DataFrame(st.session_state.paper_trades)
        
        # Format for display
        display_df = df_trades.copy()
        display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Value'] = display_df['value'].apply(lambda x: f"₹{x:,.0f}")
        
        if 'pnl' in display_df.columns:
            display_df['P&L'] = display_df['pnl'].apply(lambda x: f"₹{x:+,.0f}" if pd.notna(x) else "N/A")
        
        # Select columns for display
        display_columns = ['Date', 'stock', 'action', 'quantity', 'Price', 'Value']
        if 'P&L' in display_df.columns:
            display_columns.append('P&L')
        
        final_df = display_df[display_columns].rename(columns={
            'stock': 'Stock',
            'action': 'Action',
            'quantity': 'Quantity'
        })
        
        st.dataframe(final_df, hide_index=True, use_container_width=True)
        
        # Export trades
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("📥 Export Trades"):
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        # Clear history
        with col1:
            if st.button("🗑️ Clear Trade History", type="secondary"):
                if st.button("⚠️ Confirm Clear History"):
                    st.session_state.paper_trades = []
                    st.session_state.active_positions = {}
                    st.session_state.portfolio_value = self.initial_capital
                    st.success("Trade history cleared!")
                    st.rerun()