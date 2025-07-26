"""
Swing Trading Analysis - Indian Stocks
A comprehensive technical analysis app for Indian retail traders
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalysis
from src.pattern_detection import PatternDetection
from src.excel_exporter import ExcelExporter
from src.chart_viewer import ChartViewer
from src.paper_trading import PaperTrading
from src.tradingview_setup import TradingViewSetup
from src.config import INDIAN_STOCKS, TIMEFRAMES, INDICES

# Page configuration
st.set_page_config(
    page_title="Swing Trading Analysis - Indian Stocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
    }
    .warning-signal {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
    .danger-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #dc3545;
    }
    .stSelectbox > div > div > select {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.title("📈 Swing Trading Analysis - Indian Stocks")
    st.markdown("**Complete Technical Analysis & Pattern Detection for NSE/BSE Stocks**")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("🎯 Global Settings")
        
        # Stock selection method
        selection_method = st.radio(
            "Stock Selection Method",
            ["Manual Selection", "Index Selection"],
            help="Choose how to select stocks for analysis"
        )
        
        if selection_method == "Manual Selection":
            # Manual stock selection
            selected_stocks = st.multiselect(
                "Select Stocks",
                options=list(INDIAN_STOCKS.keys()),
                default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC"],
                help="Choose stocks from our database of 100 Indian stocks"
            )
        else:
            # Index-based selection
            selected_index = st.selectbox(
                "Select Index",
                options=list(INDICES.keys()),
                help="Choose from predefined indices"
            )
            selected_stocks = INDICES[selected_index]
            st.write(f"**Selected {len(selected_stocks)} stocks from {selected_index}:**")
            st.write(", ".join(selected_stocks[:10]) + ("..." if len(selected_stocks) > 10 else ""))
        
        # Timeframe selection
        selected_timeframes = st.multiselect(
            "Select Timeframes",
            options=list(TIMEFRAMES.keys()),
            default=["1D", "4H", "1H"],
            help="Multiple timeframes for comprehensive analysis"
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=90),
                help="Historical data start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Historical data end date"
            )
        
        # Market cap filter
        market_cap_filter = st.selectbox(
            "Market Cap Filter",
            ["All", "Large Cap", "Mid Cap", "Small Cap"],
            help="Filter stocks by market capitalization"
        )
        
        # Apply market cap filter
        if market_cap_filter != "All":
            from src.config import MARKET_CAPS
            filtered_stocks = [stock for stock in selected_stocks if stock in MARKET_CAPS.get(market_cap_filter, [])]
            selected_stocks = filtered_stocks
        
        st.markdown("---")
        st.markdown("**📊 Quick Stats**")
        st.metric("Selected Stocks", len(selected_stocks))
        st.metric("Timeframes", len(selected_timeframes))
        st.metric("Analysis Period", f"{(end_date - start_date).days} days")
        
        # Quick actions
        st.markdown("---")
        st.markdown("**⚡ Quick Actions**")
        
        if st.button("🔄 Clear Cache", help="Clear data cache to fetch fresh data"):
            if 'data_fetcher' in st.session_state:
                st.session_state.data_fetcher.clear_cache()
        
        if st.button("📊 Market Overview", help="Show overall market status"):
            show_market_overview()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Technical Analysis", 
        "📈 Chart Inspector", 
        "📋 TradingView Setup",
        "🧪 Paper Trading",
        "📑 Excel Reports",
        "🔍 Pattern Scanner"
    ])
    
    # Initialize components
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher()
    if 'technical_analysis' not in st.session_state:
        st.session_state.technical_analysis = TechnicalAnalysis()
    if 'pattern_detection' not in st.session_state:
        st.session_state.pattern_detection = PatternDetection()
    if 'excel_exporter' not in st.session_state:
        st.session_state.excel_exporter = ExcelExporter()
    if 'chart_viewer' not in st.session_state:
        st.session_state.chart_viewer = ChartViewer()
    if 'paper_trading' not in st.session_state:
        st.session_state.paper_trading = PaperTrading()
    if 'tradingview_setup' not in st.session_state:
        st.session_state.tradingview_setup = TradingViewSetup()

    with tab1:
        st.header("📊 Technical Analysis Dashboard")
        
        if not selected_stocks:
            st.warning("Please select at least one stock from the sidebar.")
            return
        
        # Analysis controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Overview", "Detailed Signals", "Pattern Detection", "Breakout Analysis"]
            )
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Signal Strength", "Volume", "Price Change", "RSI"]
            )
        with col3:
            max_stocks = st.slider("Max Stocks to Analyze", 5, 50, 20)
        
        # Run analysis
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing stocks... This may take a few moments."):
                analysis_results = []
                
                progress_bar = st.progress(0)
                total_stocks = min(len(selected_stocks), max_stocks)
                
                for idx, stock in enumerate(selected_stocks[:max_stocks]):
                    try:
                        # Fetch data for each timeframe
                        stock_results = {
                            'stock': stock,
                            'company': INDIAN_STOCKS.get(stock, stock),
                            'timeframes': {}
                        }
                        
                        for timeframe in selected_timeframes:
                            # Get stock data
                            data = st.session_state.data_fetcher.get_stock_data(
                                stock, timeframe, start_date, end_date
                            )
                            
                            if data is not None and len(data) > 50:  # Minimum data points
                                # Technical analysis
                                indicators = st.session_state.technical_analysis.calculate_all_indicators(data)
                                
                                # Pattern detection
                                patterns = st.session_state.pattern_detection.detect_all_patterns(data)
                                
                                # Generate signals
                                signals = st.session_state.technical_analysis.generate_signals(data, indicators)
                                
                                stock_results['timeframes'][timeframe] = {
                                    'data': data,
                                    'indicators': indicators,
                                    'patterns': patterns,
                                    'signals': signals
                                }
                        
                        analysis_results.append(stock_results)
                        progress_bar.progress((idx + 1) / total_stocks)
                        
                    except Exception as e:
                        st.error(f"Error analyzing {stock}: {str(e)}")
                        continue
                
                progress_bar.empty()
                
                # Display results
                if analysis_results:
                    display_analysis_results(analysis_results, analysis_type)
                else:
                    st.error("No data available for the selected stocks and timeframes.")

    with tab2:
        st.header("📈 Interactive Chart Inspector")
        st.session_state.chart_viewer.render_chart_inspector(
            selected_stocks, selected_timeframes, start_date, end_date,
            st.session_state.data_fetcher, st.session_state.technical_analysis, 
            st.session_state.pattern_detection
        )

    with tab3:
        st.header("📋 TradingView Setup Generator")
        st.session_state.tradingview_setup.render_setup_generator(
            selected_stocks, selected_timeframes, start_date, end_date,
            st.session_state.data_fetcher, st.session_state.technical_analysis
        )

    with tab4:
        st.header("🧪 Paper Trading Simulator")
        st.session_state.paper_trading.render_paper_trading(
            selected_stocks, selected_timeframes, start_date, end_date,
            st.session_state.data_fetcher, st.session_state.technical_analysis
        )

    with tab5:
        st.header("📑 Excel Report Generator")
        st.session_state.excel_exporter.render_excel_exporter(
            selected_stocks, selected_timeframes, start_date, end_date,
            st.session_state.data_fetcher, st.session_state.technical_analysis, 
            st.session_state.pattern_detection
        )

    with tab6:
        st.header("🔍 Real-time Pattern Scanner")
        render_pattern_scanner(
            selected_stocks, selected_timeframes,
            st.session_state.data_fetcher, st.session_state.technical_analysis, 
            st.session_state.pattern_detection
        )

def display_analysis_results(results, analysis_type):
    """Display technical analysis results in organized format"""
    
    if analysis_type == "Overview":
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_signals = sum(
            len(stock['timeframes'].get(tf, {}).get('signals', {}))
            for stock in results
            for tf in stock['timeframes']
        )
        
        buy_signals = sum(
            1 for stock in results
            for tf in stock['timeframes']
            for signal_type, signal in stock['timeframes'][tf].get('signals', {}).items()
            if signal.get('action') == 'BUY'
        )
        
        sell_signals = sum(
            1 for stock in results
            for tf in stock['timeframes']
            for signal_type, signal in stock['timeframes'][tf].get('signals', {}).items()
            if signal.get('action') == 'SELL'
        )
        
        total_patterns = sum(
            len(stock['timeframes'].get(tf, {}).get('patterns', []))
            for stock in results
            for tf in stock['timeframes']
        )
        
        with col1:
            st.metric("Total Signals", total_signals)
        with col2:
            st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/max(total_signals,1)*100:.1f}%")
        with col3:
            st.metric("Sell Signals", sell_signals, delta=f"{sell_signals/max(total_signals,1)*100:.1f}%")
        with col4:
            st.metric("Patterns Found", total_patterns)
        
        # Quick signals table
        st.subheader("🎯 Quick Signals Overview")
        signals_data = []
        
        for stock_result in results:
            stock = stock_result['stock']
            company = stock_result['company']
            
            for timeframe, tf_data in stock_result['timeframes'].items():
                signals = tf_data.get('signals', {})
                patterns = tf_data.get('patterns', [])
                
                # Get strongest signal
                strongest_signal = get_strongest_signal(signals)
                pattern_count = len(patterns)
                
                # Get current price and change
                data = tf_data.get('data')
                if data is not None and len(data) > 1:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    price_change = ((current_price - prev_price) / prev_price) * 100
                else:
                    current_price = 0
                    price_change = 0
                
                signals_data.append({
                    'Stock': stock,
                    'Company': company[:30] + "..." if len(company) > 30 else company,
                    'Timeframe': timeframe,
                    'Current Price': f"₹{current_price:.2f}",
                    'Change %': f"{price_change:+.2f}%",
                    'Signal': strongest_signal['action'] if strongest_signal else 'HOLD',
                    'Strength': strongest_signal['strength'] if strongest_signal else 'Neutral',
                    'Patterns': pattern_count,
                    'Entry Zone': f"₹{strongest_signal.get('entry_price', 0):.2f}" if strongest_signal and strongest_signal.get('entry_price') else 'N/A'
                })
        
        if signals_data:
            df = pd.DataFrame(signals_data)
            
            # Color code the signals
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'SELL':
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return 'background-color: #fff3cd; color: #856404'
            
            styled_df = df.style.applymap(color_signal, subset=['Signal'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    elif analysis_type == "Detailed Signals":
        st.subheader("📋 Detailed Signal Analysis")
        
        for stock_result in results:
            stock = stock_result['stock']
            company = stock_result['company']
            
            with st.expander(f"📈 {stock} - {company}", expanded=False):
                for timeframe, tf_data in stock_result['timeframes'].items():
                    st.markdown(f"**{timeframe} Timeframe:**")
                    
                    signals = tf_data.get('signals', {})
                    if signals:
                        for signal_type, signal in signals.items():
                            action = signal.get('action', 'HOLD')
                            strength = signal.get('strength', 'Weak')
                            description = signal.get('description', '')
                            
                            if action == 'BUY':
                                st.success(f"**{signal_type}**: {action} ({strength})")
                            elif action == 'SELL':
                                st.error(f"**{signal_type}**: {action} ({strength})")
                            else:
                                st.info(f"**{signal_type}**: {action} ({strength})")
                            
                            st.write(f"📝 {description}")
                            
                            if signal.get('entry_price'):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"💰 Entry: ₹{signal['entry_price']:.2f}")
                                with col2:
                                    st.write(f"🛑 Stop Loss: ₹{signal.get('stop_loss', 0):.2f}")
                                with col3:
                                    st.write(f"🎯 Target: ₹{signal.get('target', 0):.2f}")
                    else:
                        st.write("No signals detected for this timeframe.")
                    
                    st.markdown("---")
    
    elif analysis_type == "Pattern Detection":
        st.subheader("🔍 Pattern Detection Results")
        
        all_patterns = []
        for stock_result in results:
            stock = stock_result['stock']
            for timeframe, tf_data in stock_result['timeframes'].items():
                patterns = tf_data.get('patterns', [])
                for pattern in patterns:
                    pattern['stock'] = stock
                    pattern['timeframe'] = timeframe
                    all_patterns.append(pattern)
        
        if all_patterns:
            # Sort by date (most recent first)
            all_patterns.sort(key=lambda x: x.get('date', datetime.min), reverse=True)
            
            pattern_data = []
            for pattern in all_patterns[:20]:  # Show top 20 patterns
                pattern_data.append({
                    'Stock': pattern.get('stock', 'Unknown'),
                    'Timeframe': pattern.get('timeframe', 'Unknown'),
                    'Pattern': pattern.get('name', 'Unknown'),
                    'Type': pattern.get('type', 'Unknown'),
                    'Direction': pattern.get('direction', 'Neutral'),
                    'Strength': pattern.get('strength', 'Weak'),
                    'Success Rate': f"{pattern.get('success_rate', 0)}%",
                    'Date': pattern.get('date', 'N/A')
                })
            
            df_patterns = pd.DataFrame(pattern_data)
            
            # Color code by direction
            def color_direction(val):
                if val == 'Bullish':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Bearish':
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return 'background-color: #fff3cd; color: #856404'
            
            styled_df = df_patterns.style.applymap(color_direction, subset=['Direction'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No patterns detected in the selected stocks and timeframes.")
    
    elif analysis_type == "Breakout Analysis":
        st.subheader("🚀 Breakout Analysis")
        
        breakout_data = []
        for stock_result in results:
            stock = stock_result['stock']
            
            for timeframe, tf_data in stock_result['timeframes'].items():
                data = tf_data.get('data')
                indicators = tf_data.get('indicators', {})
                
                if data is not None and len(data) > 1:
                    current_price = data['Close'].iloc[-1]
                    
                    # Check for breakouts
                    if 'RESISTANCE' in indicators and 'SUPPORT' in indicators:
                        resistance = indicators['RESISTANCE'][-1]
                        support = indicators['SUPPORT'][-1]
                        
                        breakout_type = "None"
                        if current_price > resistance * 1.01:
                            breakout_type = "Resistance Breakout"
                        elif current_price < support * 0.99:
                            breakout_type = "Support Breakdown"
                        
                        if breakout_type != "None":
                            volume_ratio = indicators.get('VOLUME_RATIO', [1])[-1]
                            
                            breakout_data.append({
                                'Stock': stock,
                                'Timeframe': timeframe,
                                'Type': breakout_type,
                                'Current Price': f"₹{current_price:.2f}",
                                'Key Level': f"₹{resistance:.2f}" if "Resistance" in breakout_type else f"₹{support:.2f}",
                                'Volume Ratio': f"{volume_ratio:.1f}x",
                                'Strength': 'Strong' if volume_ratio > 1.5 else 'Weak'
                            })
        
        if breakout_data:
            df_breakouts = pd.DataFrame(breakout_data)
            
            # Color code by type
            def color_breakout(val):
                if 'Breakout' in val:
                    return 'background-color: #d4edda; color: #155724'
                elif 'Breakdown' in val:
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return ''
            
            styled_df = df_breakouts.style.applymap(color_breakout, subset=['Type'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant breakouts detected in the selected stocks and timeframes.")

def get_strongest_signal(signals):
    """Get the strongest trading signal from all available signals"""
    if not signals:
        return None
    
    signal_priorities = {'BUY': 3, 'SELL': 2, 'HOLD': 1}
    strength_priorities = {'Very Strong': 5, 'Strong': 4, 'Moderate': 3, 'Weak': 2, 'Very Weak': 1}
    
    strongest = None
    max_score = 0
    
    for signal_type, signal in signals.items():
        action_score = signal_priorities.get(signal.get('action', 'HOLD'), 0)
        strength_score = strength_priorities.get(signal.get('strength', 'Weak'), 0)
        total_score = action_score * strength_score
        
        if total_score > max_score:
            max_score = total_score
            strongest = signal
    
    return strongest

def render_pattern_scanner(selected_stocks, selected_timeframes, 
                          data_fetcher, technical_analysis, pattern_detection):
    """Render real-time pattern scanner"""
    
    st.markdown("""
    🔍 **Pattern Scanner** monitors your selected stocks for emerging patterns and breakouts.
    Set up alerts and get notified when your favorite setups appear.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_type = st.selectbox(
            "Scan Type",
            ["All Patterns", "Breakout Patterns", "Reversal Patterns", "Continuation Patterns"]
        )
        
        pattern_filters = st.multiselect(
            "Pattern Filters",
            ["Bullish Engulfing", "Bearish Engulfing", "Hammer", "Doji", 
             "Triangle", "Wedge", "Flag", "Head & Shoulders", "Cup & Handle"],
            default=["Bullish Engulfing", "Triangle", "Flag"]
        )
    
    with col2:
        min_volume = st.number_input("Min Volume (Cr)", value=1.0, step=0.5)
        min_price_change = st.number_input("Min Price Change %", value=2.0, step=0.5)
        
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        if st.button("🔍 Start Scan", type="primary"):
            run_pattern_scan(selected_stocks, selected_timeframes, scan_type, 
                            pattern_filters, min_volume, min_price_change,
                            data_fetcher, technical_analysis, pattern_detection)

def run_pattern_scan(stocks, timeframes, scan_type, pattern_filters, 
                    min_volume, min_price_change, data_fetcher, 
                    technical_analysis, pattern_detection):
    """Run pattern scanning on selected stocks"""
    
    with st.spinner("Scanning for patterns..."):
        scan_results = []
        
        progress_bar = st.progress(0)
        total_combinations = len(stocks) * len(timeframes)
        current_progress = 0
        
        for stock in stocks:
            for timeframe in timeframes:
                try:
                    # Fetch recent data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)  # Last 30 days
                    
                    data = data_fetcher.get_stock_data(stock, timeframe, 
                                                     start_date.date(), end_date.date())
                    
                    if data is not None and len(data) > 20:
                        # Detect patterns
                        patterns = pattern_detection.detect_all_patterns(data)
                        
                        # Filter patterns based on criteria
                        for pattern in patterns:
                            pattern_name = pattern.get('name', '')
                            
                            # Apply pattern filters
                            if pattern_filters and not any(pf in pattern_name for pf in pattern_filters):
                                continue
                            
                            # Apply scan type filter
                            if scan_type == "Breakout Patterns" and pattern.get('type') != 'Breakout':
                                continue
                            elif scan_type == "Reversal Patterns" and 'Reversal' not in pattern.get('description', ''):
                                continue
                            elif scan_type == "Continuation Patterns" and 'Continuation' not in pattern.get('description', ''):
                                continue
                            
                            # Check volume and price change criteria
                            current_price = data['Close'].iloc[-1]
                            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                            price_change = abs((current_price - prev_price) / prev_price * 100)
                            
                            current_volume = data['Volume'].iloc[-1] / 10000000  # Convert to Cr
                            
                            if current_volume >= min_volume and price_change >= min_price_change:
                                scan_results.append({
                                    'stock': stock,
                                    'timeframe': timeframe,
                                    'pattern': pattern,
                                    'current_price': current_price,
                                    'price_change': price_change,
                                    'volume': current_volume
                                })
                
                except Exception as e:
                    st.warning(f"Error scanning {stock} - {timeframe}: {str(e)}")
                
                current_progress += 1
                progress_bar.progress(current_progress / total_combinations)
        
        progress_bar.empty()
        
        # Display scan results
        if scan_results:
            st.success(f"✅ Found {len(scan_results)} patterns matching your criteria!")
            
            # Create results table
            results_data = []
            for result in scan_results:
                pattern = result['pattern']
                results_data.append({
                    'Stock': result['stock'],
                    'Timeframe': result['timeframe'],
                    'Pattern': pattern.get('name', 'Unknown'),
                    'Direction': pattern.get('direction', 'Neutral'),
                    'Strength': pattern.get('strength', 'Weak'),
                    'Success Rate': f"{pattern.get('success_rate', 0)}%",
                    'Current Price': f"₹{result['current_price']:.2f}",
                    'Price Change': f"{result['price_change']:+.2f}%",
                    'Volume (Cr)': f"{result['volume']:.1f}",
                    'Description': pattern.get('description', '')[:50] + "..."
                })
            
            df_results = pd.DataFrame(results_data)
            
            # Color code by direction
            def color_direction(val):
                if val == 'Bullish':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Bearish':
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return 'background-color: #fff3cd; color: #856404'
            
            styled_df = df_results.style.applymap(color_direction, subset=['Direction'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Export scan results
            if st.button("📥 Export Scan Results"):
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"pattern_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No patterns found matching your criteria. Try adjusting the filters.")

def show_market_overview():
    """Show overall market overview"""
    
    st.markdown("### 📊 Market Overview")
    
    # This would typically fetch real market data
    # For demo purposes, showing sample data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("NIFTY 50", "19,674.25", "+127.35 (+0.65%)")
    
    with col2:
        st.metric("SENSEX", "66,023.69", "+431.02 (+0.66%)")
    
    with col3:
        st.metric("BANK NIFTY", "44,856.70", "+298.45 (+0.67%)")
    
    with col4:
        st.metric("NIFTY IT", "30,245.15", "+156.80 (+0.52%)")
    
    st.info("💡 Market data is for demonstration purposes. In production, this would fetch real-time data from market APIs.")

if __name__ == "__main__":
    main()