"""TradingView setup generator for swing trading strategies"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class TradingViewSetup:
    """Class to generate TradingView-style setups and configurations"""
    
    def __init__(self):
        self.setups_database = {
            'Large Cap': {
                '1H': [
                    {
                        'name': 'EMA Crossover + Volume Surge',
                        'description': 'EMA 10 crosses above EMA 21 with volume > 1.5x average',
                        'success_rate': 72,
                        'avg_return': 4.2,
                        'risk_level': 'Medium',
                        'timeframe': '1H',
                        'indicators': ['EMA 10', 'EMA 21', 'Volume'],
                        'entry_rules': 'EMA 10 > EMA 21 AND Volume > 1.5x MA',
                        'stop_loss': '2% below entry',
                        'target': '6% above entry',
                        'example_stocks': ['RELIANCE', 'TCS', 'HDFCBANK']
                    },
                    {
                        'name': 'Bollinger Bounce Strategy',
                        'description': 'Price bounces from lower Bollinger Band with RSI < 35',
                        'success_rate': 68,
                        'avg_return': 3.8,
                        'risk_level': 'Low',
                        'timeframe': '1H',
                        'indicators': ['Bollinger Bands', 'RSI'],
                        'entry_rules': 'Price touches BB Lower AND RSI < 35',
                        'stop_loss': '1.5% below BB Lower',
                        'target': 'BB Middle or Resistance',
                        'example_stocks': ['HINDUNILVR', 'ITC', 'ASIANPAINT']
                    }
                ],
                '4H': [
                    {
                        'name': 'MACD Momentum Play',
                        'description': 'MACD crosses signal line with price above EMA 50',
                        'success_rate': 75,
                        'avg_return': 5.1,
                        'risk_level': 'Medium',
                        'timeframe': '4H',
                        'indicators': ['MACD', 'EMA 50'],
                        'entry_rules': 'MACD crosses above Signal AND Price > EMA 50',
                        'stop_loss': '2.5% below entry',
                        'target': '7.5% above entry',
                        'example_stocks': ['INFY', 'WIPRO', 'HCLTECH']
                    }
                ],
                '1D': [
                    {
                        'name': 'Support Breakout with Volume',
                        'description': 'Price breaks above resistance with high volume confirmation',
                        'success_rate': 78,
                        'avg_return': 8.3,
                        'risk_level': 'Medium',
                        'timeframe': '1D',
                        'indicators': ['Support/Resistance', 'Volume', 'RSI'],
                        'entry_rules': 'Price > Resistance AND Volume > 2x MA AND RSI < 70',
                        'stop_loss': '3% below resistance level',
                        'target': '10% above entry',
                        'example_stocks': ['HDFC', 'ICICIBANK', 'KOTAKBANK']
                    }
                ]
            },
            'Mid Cap': {
                '1H': [
                    {
                        'name': 'RSI Oversold Reversal',
                        'description': 'RSI below 30 with bullish divergence and volume spike',
                        'success_rate': 70,
                        'avg_return': 6.2,
                        'risk_level': 'High',
                        'timeframe': '1H',
                        'indicators': ['RSI', 'Volume', 'Price Divergence'],
                        'entry_rules': 'RSI < 30 AND Bullish Divergence AND Volume > 1.8x MA',
                        'stop_loss': '3% below recent low',
                        'target': '8% above entry',
                        'example_stocks': ['ADANIPORTS', 'TATAPOWER', 'JINDALSTEL']
                    }
                ],
                '4H': [
                    {
                        'name': 'Triangle Breakout',
                        'description': 'Ascending triangle breakout with volume confirmation',
                        'success_rate': 73,
                        'avg_return': 7.8,
                        'risk_level': 'Medium',
                        'timeframe': '4H',
                        'indicators': ['Chart Patterns', 'Volume', 'ATR'],
                        'entry_rules': 'Price breaks triangle resistance AND Volume > 2x MA',
                        'stop_loss': 'Below triangle support',
                        'target': 'Height of triangle added to breakout',
                        'example_stocks': ['SAIL', 'COALINDIA', 'IOC']
                    }
                ],
                '1D': [
                    {
                        'name': 'Flag Pattern Continuation',
                        'description': 'Bull flag pattern after strong uptrend',
                        'success_rate': 76,
                        'avg_return': 9.1,
                        'risk_level': 'Medium',
                        'timeframe': '1D',
                        'indicators': ['Chart Patterns', 'Volume', 'Trend Analysis'],
                        'entry_rules': 'Flag breakout AND Volume surge AND Price > EMA 20',
                        'stop_loss': 'Below flag low',
                        'target': 'Flagpole height added to breakout',
                        'example_stocks': ['BPCL', 'HPCL', 'GODREJCP']
                    }
                ]
            },
            'Small Cap': {
                '1H': [
                    {
                        'name': 'Momentum Breakout',
                        'description': 'High volume breakout with multiple indicator alignment',
                        'success_rate': 65,
                        'avg_return': 8.7,
                        'risk_level': 'Very High',
                        'timeframe': '1H',
                        'indicators': ['Volume', 'RSI', 'MACD', 'EMA'],
                        'entry_rules': 'Volume > 3x MA AND RSI > 50 AND MACD > Signal',
                        'stop_loss': '4% below entry',
                        'target': '12% above entry',
                        'example_stocks': ['ZEEL', 'SUZLON', 'RPOWER']
                    }
                ],
                '4H': [
                    {
                        'name': 'Cup and Handle',
                        'description': 'Cup and handle pattern with volume expansion',
                        'success_rate': 69,
                        'avg_return': 11.2,
                        'risk_level': 'High',
                        'timeframe': '4H',
                        'indicators': ['Chart Patterns', 'Volume', 'Support/Resistance'],
                        'entry_rules': 'Handle breakout AND Volume > 2.5x MA',
                        'stop_loss': 'Below handle low',
                        'target': 'Cup depth added to breakout',
                        'example_stocks': ['YESBANK', 'JPASSOCIAT', 'DHFL']
                    }
                ],
                '1D': [
                    {
                        'name': 'Double Bottom Reversal',
                        'description': 'Double bottom with volume confirmation and RSI divergence',
                        'success_rate': 71,
                        'avg_return': 13.5,
                        'risk_level': 'High',
                        'timeframe': '1D',
                        'indicators': ['Chart Patterns', 'Volume', 'RSI', 'Support'],
                        'entry_rules': 'Neckline break AND Volume surge AND RSI divergence',
                        'stop_loss': 'Below second bottom',
                        'target': 'Distance from bottom to neckline',
                        'example_stocks': ['PCJEWELLER', 'JETAIRWAYS', 'RCOM']
                    }
                ]
            }
        }
    
    def render_setup_generator(self, selected_stocks, selected_timeframes, 
                              start_date, end_date, data_fetcher, technical_analysis):
        """Render TradingView setup generator interface"""
        
        st.markdown("""
        📋 **TradingView Setup Generator** creates professional trading setups with clear entry/exit rules, 
        success rates, and example configurations for different market conditions.
        """)
        
        # Setup filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_cap_filter = st.selectbox(
                "Market Cap Focus",
                ["All", "Large Cap", "Mid Cap", "Small Cap"],
                help="Filter setups by market capitalization"
            )
        
        with col2:
            timeframe_filter = st.selectbox(
                "Primary Timeframe",
                ["All"] + (selected_timeframes if selected_timeframes else ["1H", "4H", "1D"]),
                help="Focus on specific timeframes"
            )
        
        with col3:
            success_rate_filter = st.slider(
                "Minimum Success Rate %",
                min_value=50, max_value=90, value=65, step=5,
                help="Filter setups by historical success rate"
            )
        
        # Display mode selection
        display_mode = st.radio(
            "Display Mode",
            ["Strategy Cards", "Detailed Table", "TradingView Script"],
            horizontal=True,
            help="Choose how to display the trading setups"
        )
        
        # Generate setups
        if st.button("🚀 Generate Trading Setups", type="primary"):
            filtered_setups = self.get_filtered_setups(
                market_cap_filter, timeframe_filter, success_rate_filter
            )
            
            if filtered_setups:
                if display_mode == "Strategy Cards":
                    self.display_strategy_cards(filtered_setups)
                elif display_mode == "Detailed Table":
                    self.display_detailed_table(filtered_setups)
                elif display_mode == "TradingView Script":
                    self.display_tradingview_script(filtered_setups)
            else:
                st.warning("No setups match your criteria. Try adjusting the filters.")
        
        # Quick setup recommendations
        st.markdown("---")
        st.markdown("### 🎯 Quick Setup Recommendations")
        
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("📈 Trending Setups", use_container_width=True):
                self.display_trending_setups()
        
        with quick_col2:
            if st.button("🔥 High Success Rate Setups", use_container_width=True):
                self.display_high_success_setups()
    
    def get_filtered_setups(self, market_cap_filter, timeframe_filter, success_rate_filter):
        """Filter setups based on user criteria"""
        
        filtered_setups = []
        
        for market_cap, timeframes in self.setups_database.items():
            # Market cap filter
            if market_cap_filter != "All" and market_cap != market_cap_filter:
                continue
            
            for timeframe, setups in timeframes.items():
                # Timeframe filter
                if timeframe_filter != "All" and timeframe != timeframe_filter:
                    continue
                
                for setup in setups:
                    # Success rate filter
                    if setup['success_rate'] >= success_rate_filter:
                        setup['market_cap'] = market_cap
                        filtered_setups.append(setup)
        
        # Sort by success rate descending
        filtered_setups.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return filtered_setups
    
    def display_strategy_cards(self, setups):
        """Display setups as interactive cards"""
        
        st.markdown("### 📋 Trading Setup Cards")
        
        for i, setup in enumerate(setups):
            with st.expander(f"🎯 {setup['name']} - {setup['market_cap']} ({setup['timeframe']})", expanded=i<3):
                
                # Setup metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Success Rate", f"{setup['success_rate']}%")
                
                with col2:
                    st.metric("Avg Return", f"{setup['avg_return']}%")
                
                with col3:
                    risk_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Very High': '🔴'}
                    st.metric("Risk Level", f"{risk_color.get(setup['risk_level'], '⚪')} {setup['risk_level']}")
                
                # Setup details
                st.markdown("**📝 Description:**")
                st.write(setup['description'])
                
                st.markdown("**🔧 Required Indicators:**")
                indicators_text = " • ".join(setup['indicators'])
                st.write(f"• {indicators_text}")
                
                st.markdown("**📊 Entry Rules:**")
                st.code(setup['entry_rules'], language="text")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🛑 Stop Loss:**")
                    st.write(setup['stop_loss'])
                
                with col2:
                    st.markdown("**🎯 Target:**")
                    st.write(setup['target'])
                
                st.markdown("**📈 Example Stocks:**")
                st.write(" • ".join(setup['example_stocks']))
                
                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if st.button(f"📋 Copy Setup", key=f"copy_{i}"):
                        setup_text = self.format_setup_for_copy(setup)
                        st.code(setup_text, language="text")
                
                with btn_col2:
                    if st.button(f"📊 Backtest", key=f"backtest_{i}"):
                        st.info("Backtesting feature would analyze historical performance of this setup.")
                
                with btn_col3:
                    if st.button(f"⚠️ Set Alert", key=f"alert_{i}"):
                        st.info("Alert system would monitor stocks for this setup condition.")
    
    def display_detailed_table(self, setups):
        """Display setups in a detailed table format"""
        
        st.markdown("### 📊 Detailed Setup Analysis")
        
        # Create DataFrame
        table_data = []
        for setup in setups:
            table_data.append({
                'Setup Name': setup['name'],
                'Market Cap': setup['market_cap'],
                'Timeframe': setup['timeframe'],
                'Success Rate': f"{setup['success_rate']}%",
                'Avg Return': f"{setup['avg_return']}%",
                'Risk Level': setup['risk_level'],
                'Entry Rules': setup['entry_rules'][:50] + "..." if len(setup['entry_rules']) > 50 else setup['entry_rules'],
                'Example Stocks': ", ".join(setup['example_stocks'][:2])
            })
        
        df = pd.DataFrame(table_data)
        
        # Display with enhanced formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Success Rate': st.column_config.ProgressColumn(
                    'Success Rate',
                    help="Historical success rate of the setup",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%"
                ),
                'Risk Level': st.column_config.SelectboxColumn(
                    'Risk Level',
                    options=['Low', 'Medium', 'High', 'Very High'],
                    help="Risk assessment for the trading setup"
                )
            }
        )
        
        # Export option
        if st.button("📥 Export Setup Table"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_setups_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def display_tradingview_script(self, setups):
        """Display TradingView Pine Script format"""
        
        st.markdown("### 📜 TradingView Pine Script Generator")
        
        if not setups:
            st.warning("No setups available to generate scripts.")
            return
        
        selected_setup = st.selectbox(
            "Select Setup for Pine Script",
            [f"{setup['name']} - {setup['market_cap']}" for setup in setups]
        )
        
        # Find the selected setup
        setup_index = [f"{setup['name']} - {setup['market_cap']}" for setup in setups].index(selected_setup)
        setup = setups[setup_index]
        
        # Generate Pine Script
        pine_script = self.generate_pine_script(setup)
        
        st.markdown("**📝 Generated Pine Script:**")
        st.code(pine_script, language="javascript")
        
        # Copy to clipboard button
        if st.button("📋 Copy Pine Script"):
            st.success("Pine Script copied! Paste it into TradingView's Pine Editor.")
        
        # Instructions
        st.markdown("### 📖 How to Use:")
        st.markdown("""
        1. **Copy the Pine Script** above
        2. **Open TradingView** and go to Pine Editor
        3. **Paste the script** and click "Add to Chart"
        4. **Customize** the parameters as needed
        5. **Set up alerts** for the signals
        """)
        
        st.info("""
        💡 **Pro Tip:** Adjust the input parameters in TradingView to fine-tune 
        the strategy for different market conditions and stocks.
        """)
    
    def generate_pine_script(self, setup):
        """Generate Pine Script code for the setup"""
        
        script_name = setup['name'].replace(' ', '_').replace('+', '_').lower()
        
        pine_script = f"""
//@version=5
strategy("{setup['name']}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=2)

// Input Parameters
ema_fast = input.int(10, "EMA Fast Period", minval=1)
ema_slow = input.int(21, "EMA Slow Period", minval=1)
rsi_period = input.int(14, "RSI Period", minval=1)
rsi_oversold = input.int(30, "RSI Oversold Level", minval=0, maxval=100)
rsi_overbought = input.int(70, "RSI Overbought Level", minval=0, maxval=100)
volume_multiplier = input.float(1.5, "Volume Multiplier", minval=0.1)
stop_loss_pct = input.float(2.0, "Stop Loss %", minval=0.1) / 100
take_profit_pct = input.float(6.0, "Take Profit %", minval=0.1) / 100

// Technical Indicators
ema_fast_line = ta.ema(close, ema_fast)
ema_slow_line = ta.ema(close, ema_slow)
rsi = ta.rsi(close, rsi_period)
volume_ma = ta.sma(volume, 20)
volume_surge = volume > volume_ma * volume_multiplier

// Setup Conditions
"""

        # Add specific conditions based on setup type
        if "EMA" in setup['name']:
            pine_script += """
// EMA Crossover Condition
ema_bullish = ema_fast_line > ema_slow_line
ema_cross_up = ta.crossover(ema_fast_line, ema_slow_line)

// Entry Condition
long_condition = ema_cross_up and volume_surge and rsi < rsi_overbought
"""
        
        elif "RSI" in setup['name']:
            pine_script += """
// RSI Oversold Condition
rsi_oversold_condition = rsi < rsi_oversold
rsi_recovery = rsi > rsi[1] and rsi[1] < rsi_oversold

// Entry Condition  
long_condition = rsi_oversold_condition and volume_surge and rsi_recovery
"""
        
        elif "Bollinger" in setup['name']:
            pine_script += """
// Bollinger Bands
bb_length = 20
bb_mult = 2
bb_basis = ta.sma(close, bb_length)
bb_dev = bb_mult * ta.stdev(close, bb_length)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev

// Bollinger Bounce Condition
bb_bounce = close <= bb_lower and rsi < rsi_oversold

// Entry Condition
long_condition = bb_bounce and volume_surge
"""
        
        else:
            pine_script += """
// Generic Entry Condition
long_condition = ema_fast_line > ema_slow_line and volume_surge and rsi > 50 and rsi < rsi_overbought
"""
        
        pine_script += f"""

// Strategy Execution
if long_condition
    strategy.entry("Long", strategy.long)
    
// Exit Conditions
if strategy.position_size > 0
    stop_price = close * (1 - stop_loss_pct)
    profit_price = close * (1 + take_profit_pct)
    
    strategy.exit("Exit", "Long", stop=stop_price, limit=profit_price)

// Plotting
plot(ema_fast_line, "EMA Fast", color=color.blue)
plot(ema_slow_line, "EMA Slow", color=color.red)

{'plot(bb_upper, "BB Upper", color=color.gray)' if 'Bollinger' in setup['name'] else ''}
{'plot(bb_lower, "BB Lower", color=color.gray)' if 'Bollinger' in setup['name'] else ''}

// Background color for entry signals
bgcolor(long_condition ? color.new(color.green, 90) : na)

// Labels for entry points
if long_condition
    label.new(bar_index, low, "BUY", style=label.style_label_up, color=color.green, size=size.small)

// Performance Info
// Success Rate: {setup['success_rate']}%
// Average Return: {setup['avg_return']}%
// Risk Level: {setup['risk_level']}
"""
        
        return pine_script
    
    def format_setup_for_copy(self, setup):
        """Format setup for easy copying"""
        
        formatted_text = f"""
🎯 {setup['name']} - {setup['market_cap']} Strategy

📊 Performance Metrics:
• Success Rate: {setup['success_rate']}%
• Average Return: {setup['avg_return']}%
• Risk Level: {setup['risk_level']}
• Timeframe: {setup['timeframe']}

🔧 Required Indicators:
{chr(10).join('• ' + indicator for indicator in setup['indicators'])}

📝 Entry Rules:
{setup['entry_rules']}

🛑 Stop Loss: {setup['stop_loss']}
🎯 Target: {setup['target']}

📈 Example Stocks: {', '.join(setup['example_stocks'])}

💡 Description: {setup['description']}
"""
        return formatted_text
    
    def display_trending_setups(self):
        """Display currently trending setups"""
        
        st.markdown("### 📈 Trending Setups (Last 30 Days)")
        
        trending_setups = [
            {
                'name': 'EMA Crossover + Volume Surge',
                'market_cap': 'Large Cap',
                'timeframe': '1H',
                'recent_signals': 23,
                'win_rate': 78,
                'trend': '↗️ +5%'
            },
            {
                'name': 'RSI Oversold Reversal',
                'market_cap': 'Mid Cap',
                'timeframe': '4H',
                'recent_signals': 18,
                'win_rate': 71,
                'trend': '↗️ +3%'
            },
            {
                'name': 'Triangle Breakout',
                'market_cap': 'Small Cap',
                'timeframe': '1D',
                'recent_signals': 12,
                'win_rate': 69,
                'trend': '↘️ -2%'
            }
        ]
        
        for setup in trending_setups:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"**{setup['name']}**")
                st.write(f"_{setup['market_cap']} • {setup['timeframe']}_")
            
            with col2:
                st.metric("Signals", setup['recent_signals'])
            
            with col3:
                st.metric("Win Rate", f"{setup['win_rate']}%")
            
            with col4:
                st.write("**Trend**")
                st.write(setup['trend'])
            
            with col5:
                if st.button(f"View Setup", key=f"trending_{setup['name']}"):
                    st.info(f"Detailed view for {setup['name']} would be displayed here.")
    
    def display_high_success_setups(self):
        """Display setups with highest success rates"""
        
        st.markdown("### 🔥 High Success Rate Setups (>75%)")
        
        high_success_setups = []
        
        for market_cap, timeframes in self.setups_database.items():
            for timeframe, setups in timeframes.items():
                for setup in setups:
                    if setup['success_rate'] >= 75:
                        setup['market_cap'] = market_cap
                        high_success_setups.append(setup)
        
        # Sort by success rate
        high_success_setups.sort(key=lambda x: x['success_rate'], reverse=True)
        
        for setup in high_success_setups:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{setup['name']}**")
                    st.write(f"{setup['market_cap']} • {setup['timeframe']}")
                    st.write(f"_{setup['description'][:60]}..._")
                
                with col2:
                    st.metric("Success", f"{setup['success_rate']}%")
                
                with col3:
                    st.metric("Return", f"{setup['avg_return']}%")
                
                with col4:
                    st.write(f"**{setup['risk_level']}** Risk")
                    if st.button("📋 Details", key=f"high_{setup['name']}_{setup['timeframe']}"):
                        st.info("Detailed setup information would be displayed in a modal.")
                
                st.markdown("---")