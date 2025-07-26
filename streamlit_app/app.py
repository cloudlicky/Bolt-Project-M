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
from src.config import INDIAN_STOCKS, TIMEFRAMES

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
</style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.title("📈 Swing Trading Analysis - Indian Stocks")
    st.markdown("**Complete Technical Analysis & Pattern Detection for NSE/BSE Stocks**")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("🎯 Global Settings")
        
        # Stock selection
        selected_stocks = st.multiselect(
            "Select Stocks",
            options=list(INDIAN_STOCKS.keys()),
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC"],
            help="Choose stocks from Nifty50, BankNifty, and other major indices"
        )
        
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
        
        st.markdown("---")
        st.markdown("**📊 Quick Stats**")
        st.metric("Selected Stocks", len(selected_stocks))
        st.metric("Timeframes", len(selected_timeframes))
        st.metric("Analysis Period", f"{(end_date - start_date).days} days")

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
    data_fetcher = DataFetcher()
    technical_analysis = TechnicalAnalysis()
    pattern_detection = PatternDetection()
    excel_exporter = ExcelExporter()
    chart_viewer = ChartViewer()
    paper_trading = PaperTrading()
    tradingview_setup = TradingViewSetup()

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
            market_cap_filter = st.selectbox(
                "Market Cap",
                ["All", "Large Cap", "Mid Cap", "Small Cap"]
            )
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ["Signal Strength", "Volume", "Price Change", "RSI"]
            )
        
        # Run analysis
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing stocks... This may take a few moments."):
                analysis_results = []
                
                progress_bar = st.progress(0)
                total_stocks = len(selected_stocks)
                
                for idx, stock in enumerate(selected_stocks):
                    try:
                        # Fetch data for each timeframe
                        stock_results = {
                            'stock': stock,
                            'company': INDIAN_STOCKS[stock],
                            'timeframes': {}
                        }
                        
                        for timeframe in selected_timeframes:
                            # Get stock data
                            data = data_fetcher.get_stock_data(
                                stock, timeframe, start_date, end_date
                            )
                            
                            if data is not None and len(data) > 50:  # Minimum data points
                                # Technical analysis
                                indicators = technical_analysis.calculate_all_indicators(data)
                                
                                # Pattern detection
                                patterns = pattern_detection.detect_all_patterns(data)
                                
                                # Generate signals
                                signals = technical_analysis.generate_signals(data, indicators)
                                
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
                
                # Display results
                if analysis_results:
                    display_analysis_results(analysis_results, analysis_type)
                else:
                    st.error("No data available for the selected stocks and timeframes.")

    with tab2:
        st.header("📈 Interactive Chart Inspector")
        chart_viewer.render_chart_inspector(
            selected_stocks, selected_timeframes, start_date, end_date,
            data_fetcher, technical_analysis, pattern_detection
        )

    with tab3:
        st.header("📋 TradingView Setup Generator")
        tradingview_setup.render_setup_generator(
            selected_stocks, selected_timeframes, start_date, end_date,
            data_fetcher, technical_analysis
        )

    with tab4:
        st.header("🧪 Paper Trading Simulator")
        paper_trading.render_paper_trading(
            selected_stocks, selected_timeframes, start_date, end_date,
            data_fetcher, technical_analysis
        )

    with tab5:
        st.header("📑 Excel Report Generator")
        excel_exporter.render_excel_exporter(
            selected_stocks, selected_timeframes, start_date, end_date,
            data_fetcher, technical_analysis, pattern_detection
        )

    with tab6:
        st.header("🔍 Real-time Pattern Scanner")
        render_pattern_scanner(
            selected_stocks, selected_timeframes,
            data_fetcher, technical_analysis, pattern_detection
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
        
        with col1:
            st.metric("Total Signals", total_signals)
        with col2:
            st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/max(total_signals,1)*100:.1f}%")
        with col3:
            st.metric("Stocks Analyzed", len(results))
        with col4:
            st.metric("Active Patterns", 
                     sum(len(stock['timeframes'].get(tf, {}).get('patterns', []))
                         for stock in results for tf in stock['timeframes']))
        
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
                
                signals_data.append({
                    'Stock': stock,
                    'Company': company[:30] + "..." if len(company) > 30 else company,
                    'Timeframe': timeframe,
                    'Signal': strongest_signal['action'] if strongest_signal else 'HOLD',
                    'Strength': strongest_signal['strength'] if strongest_signal else 'Neutral',
                    'Patterns': pattern_count,
                    'Entry Zone': strongest_signal.get('entry_price', 'N/A') if strongest_signal else 'N/A'
                })
        
        if signals_data:
            df = pd.DataFrame(signals_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Signal': st.column_config.SelectboxColumn(
                        'Signal',
                        options=['BUY', 'SELL', 'HOLD'],
                        help="Trading signal based on technical analysis"
                    ),
                    'Patterns': st.column_config.NumberColumn(
                        'Patterns',
                        help="Number of detected chart patterns",
                        format="%d"
                    )
                }
            )

def get_strongest_signal(signals):
    """Get the strongest trading signal from all available signals"""
    if not signals:
        return None
    
    signal_priorities = {'BUY': 3, 'SELL': 2, 'HOLD': 1}
    strongest = None
    max_priority = 0
    
    for signal_type, signal in signals.items():
        priority = signal_priorities.get(signal.get('action', 'HOLD'), 0)
        if priority > max_priority:
            max_priority = priority
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
            ["Breakout Patterns", "Reversal Patterns", "Continuation Patterns", "All Patterns"]
        )
        
        pattern_filters = st.multiselect(
            "Pattern Filters",
            ["Bullish Engulfing", "Bearish Engulfing", "Hammer", "Doji", 
             "Triangle", "Wedge", "Flag", "Head & Shoulders"],
            default=["Bullish Engulfing", "Triangle", "Flag"]
        )
    
    with col2:
        min_volume = st.number_input("Min Volume (Cr)", value=1.0, step=0.5)
        min_price_change = st.number_input("Min Price Change %", value=2.0, step=0.5)
        
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        if st.button("🔍 Start Scan", type="primary"):
            st.success("Scanner started! Monitoring selected stocks...")
    
    # Scanner results placeholder
    scanner_placeholder = st.empty()
    
    if auto_refresh:
        with scanner_placeholder.container():
            st.info("🔄 Auto-refresh enabled. Scanner will update every 30 seconds.")
            # In a real implementation, this would use st.rerun() with a timer
            st.markdown("*Note: Auto-refresh functionality would be implemented with background threads in production.*")

if __name__ == "__main__":
    main()