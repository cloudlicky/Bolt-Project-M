"""Interactive chart viewer with technical indicators and pattern highlighting"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ChartViewer:
    """Class to handle interactive chart visualization"""
    
    def __init__(self):
        self.colors = {
            'bullish': '#26A69A',
            'bearish': '#EF5350', 
            'background': '#FAFAFA',
            'grid': '#E0E0E0',
            'text': '#37474F',
            'volume': '#42A5F5'
        }
    
    def render_chart_inspector(self, selected_stocks, selected_timeframes, 
                              start_date, end_date, data_fetcher, 
                              technical_analysis, pattern_detection):
        """Render the interactive chart inspector interface"""
        
        st.markdown("""
        📈 **Chart Inspector** provides interactive visualization of stock prices with 
        technical indicators, pattern detection, and detailed analysis tools.
        """)
        
        if not selected_stocks:
            st.warning("Please select at least one stock from the sidebar.")
            return
        
        # Chart controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            chart_stock = st.selectbox(
                "Select Stock for Chart",
                options=selected_stocks,
                help="Choose stock to display in chart"
            )
        
        with col2:
            chart_timeframe = st.selectbox(
                "Chart Timeframe",
                options=selected_timeframes,
                help="Select timeframe for chart analysis"
            )
        
        with col3:
            chart_style = st.selectbox(
                "Chart Style",
                ["Candlestick", "OHLC", "Line", "Area"],
                help="Choose chart visualization style"
            )
        
        # Indicator selection
        st.markdown("**🔧 Technical Indicators**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_ema = st.checkbox("EMA (10, 21, 50)", value=True)
            show_sma = st.checkbox("SMA (20, 50)", value=False)
            show_bb = st.checkbox("Bollinger Bands", value=True)
        
        with col2:
            show_rsi = st.checkbox("RSI", value=True)
            show_macd = st.checkbox("MACD", value=True)
            show_stoch = st.checkbox("Stochastic", value=False)
        
        with col3:
            show_volume = st.checkbox("Volume", value=True)
            show_vwap = st.checkbox("VWAP", value=False)
            show_support_resistance = st.checkbox("Support/Resistance", value=True)
        
        with col4:
            highlight_patterns = st.checkbox("Highlight Patterns", value=True)
            show_signals = st.checkbox("Show Signals", value=True)
            show_annotations = st.checkbox("Pattern Annotations", value=True)
        
        # Generate chart
        if st.button("📊 Generate Interactive Chart", type="primary"):
            self.display_interactive_chart(
                chart_stock, chart_timeframe, start_date, end_date,
                data_fetcher, technical_analysis, pattern_detection,
                chart_style, {
                    'ema': show_ema, 'sma': show_sma, 'bb': show_bb,
                    'rsi': show_rsi, 'macd': show_macd, 'stoch': show_stoch,
                    'volume': show_volume, 'vwap': show_vwap, 
                    'support_resistance': show_support_resistance,
                    'patterns': highlight_patterns, 'signals': show_signals,
                    'annotations': show_annotations
                }
            )
    
    def display_interactive_chart(self, stock, timeframe, start_date, end_date,
                                 data_fetcher, technical_analysis, pattern_detection,
                                 chart_style, indicators):
        """Display the interactive chart with selected indicators"""
        
        with st.spinner("Loading chart data and calculating indicators..."):
            # Fetch data
            data = data_fetcher.get_stock_data(stock, timeframe, start_date, end_date)
            
            if data is None or len(data) < 20:
                st.error(f"Insufficient data for {stock} in {timeframe} timeframe.")
                return
            
            # Calculate indicators
            tech_indicators = technical_analysis.calculate_all_indicators(data)
            signals = technical_analysis.generate_signals(data, tech_indicators)
            patterns = pattern_detection.detect_all_patterns(data)
            
            # Create chart
            fig = self.create_comprehensive_chart(
                data, tech_indicators, signals, patterns, 
                stock, timeframe, chart_style, indicators
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Chart analysis summary
            self.display_chart_analysis(data, tech_indicators, signals, patterns, stock)
    
    def create_comprehensive_chart(self, data, indicators, signals, patterns, 
                                  stock, timeframe, chart_style, show_indicators):
        """Create comprehensive chart with all selected indicators"""
        
        # Determine subplot configuration
        subplot_count = 1  # Main price chart
        subplot_heights = [0.6]  # Main chart height ratio
        
        if show_indicators['volume']:
            subplot_count += 1
            subplot_heights.append(0.15)
        
        if show_indicators['rsi']:
            subplot_count += 1
            subplot_heights.append(0.125)
        
        if show_indicators['macd']:
            subplot_count += 1
            subplot_heights.append(0.125)
        
        # Normalize heights
        total_height = sum(subplot_heights)
        subplot_heights = [h/total_height for h in subplot_heights]
        
        # Create subplots
        subplot_titles = ['Price Chart']
        if show_indicators['volume']:
            subplot_titles.append('Volume')
        if show_indicators['rsi']:
            subplot_titles.append('RSI')
        if show_indicators['macd']:
            subplot_titles.append('MACD')
        
        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=subplot_heights
        )
        
        current_row = 1
        
        # Main price chart
        self.add_price_chart(fig, data, chart_style, current_row)
        
        # Technical indicators on main chart
        if show_indicators['ema']:
            self.add_ema_lines(fig, indicators, current_row)
        
        if show_indicators['sma']:
            self.add_sma_lines(fig, indicators, current_row)
        
        if show_indicators['bb']:
            self.add_bollinger_bands(fig, indicators, current_row)
        
        if show_indicators['vwap']:
            self.add_vwap_line(fig, indicators, current_row)
        
        if show_indicators['support_resistance']:
            self.add_support_resistance(fig, indicators, current_row)
        
        # Pattern highlighting
        if show_indicators['patterns']:
            self.add_pattern_highlights(fig, patterns, data, current_row)
        
        # Signal markers
        if show_indicators['signals']:
            self.add_signal_markers(fig, signals, data, current_row)
        
        current_row += 1
        
        # Volume chart
        if show_indicators['volume']:
            self.add_volume_chart(fig, data, indicators, current_row)
            current_row += 1
        
        # RSI chart
        if show_indicators['rsi']:
            self.add_rsi_chart(fig, indicators, current_row)
            current_row += 1
        
        # MACD chart
        if show_indicators['macd']:
            self.add_macd_chart(fig, indicators, current_row)
            current_row += 1
        
        # Update layout
        fig.update_layout(
            title=f'{stock} - {timeframe} Chart Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor='white'
        )
        
        # Update x-axis for all subplots
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            rangeslider_visible=False
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            showgrid=True
        )
        
        return fig
    
    def add_price_chart(self, fig, data, chart_style, row):
        """Add main price chart"""
        
        if chart_style == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.colors['bullish'],
                    decreasing_line_color=self.colors['bearish']
                ),
                row=row, col=1
            )
        
        elif chart_style == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=row, col=1
            )
        
        elif chart_style == "Line":
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.colors['bullish'], width=2)
                ),
                row=row, col=1
            )
        
        elif chart_style == "Area":
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    fill='tonexty',
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.colors['bullish'], width=2),
                    fillcolor=f'rgba(38, 166, 154, 0.3)'
                ),
                row=row, col=1
            )
    
    def add_ema_lines(self, fig, indicators, row):
        """Add EMA lines to chart"""
        
        if 'EMA_10' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['EMA_10']))),
                    y=indicators['EMA_10'],
                    mode='lines',
                    name='EMA 10',
                    line=dict(color='#FF9800', width=1.5)
                ),
                row=row, col=1
            )
        
        if 'EMA_21' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['EMA_21']))),
                    y=indicators['EMA_21'],
                    mode='lines',
                    name='EMA 21',
                    line=dict(color='#9C27B0', width=1.5)
                ),
                row=row, col=1
            )
        
        if 'EMA_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['EMA_50']))),
                    y=indicators['EMA_50'],
                    mode='lines',
                    name='EMA 50',
                    line=dict(color='#2196F3', width=2)
                ),
                row=row, col=1
            )
    
    def add_sma_lines(self, fig, indicators, row):
        """Add SMA lines to chart"""
        
        if 'SMA_20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['SMA_20']))),
                    y=indicators['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#FFC107', width=1.5, dash='dash')
                ),
                row=row, col=1
            )
        
        if 'SMA_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['SMA_50']))),
                    y=indicators['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#795548', width=2, dash='dash')
                ),
                row=row, col=1
            )
    
    def add_bollinger_bands(self, fig, indicators, row):
        """Add Bollinger Bands to chart"""
        
        if all(key in indicators for key in ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']):
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['BB_UPPER']))),
                    y=indicators['BB_UPPER'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(158, 158, 158, 0.8)', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Middle band (SMA 20)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['BB_MIDDLE']))),
                    y=indicators['BB_MIDDLE'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='#9E9E9E', width=1.5, dash='dot')
                ),
                row=row, col=1
            )
            
            # Lower band with fill
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['BB_LOWER']))),
                    y=indicators['BB_LOWER'],
                    mode='lines',
                    name='Bollinger Bands',
                    line=dict(color='rgba(158, 158, 158, 0.8)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(158, 158, 158, 0.1)'
                ),
                row=row, col=1
            )
    
    def add_volume_chart(self, fig, data, indicators, row):
        """Add volume chart"""
        
        # Color volume bars based on price movement
        colors = []
        for i in range(len(data)):
            if i == 0:
                colors.append(self.colors['volume'])
            else:
                if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=1
        )
        
        # Add volume moving average if available
        if 'VOLUME_MA' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['VOLUME_MA'],
                    mode='lines',
                    name='Volume MA',
                    line=dict(color='#FF5722', width=2)
                ),
                row=row, col=1
            )
    
    def add_rsi_chart(self, fig, indicators, row):
        """Add RSI chart"""
        
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['RSI']))),
                    y=indicators['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#9C27B0', width=2)
                ),
                row=row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold", row=row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                         annotation_text="Midline", row=row, col=1)
    
    def add_macd_chart(self, fig, indicators, row):
        """Add MACD chart"""
        
        if all(key in indicators for key in ['MACD', 'MACD_signal', 'MACD_hist']):
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['MACD']))),
                    y=indicators['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#2196F3', width=2)
                ),
                row=row, col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['MACD_signal']))),
                    y=indicators['MACD_signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='#FF9800', width=2)
                ),
                row=row, col=1
            )
            
            # Histogram
            colors = ['green' if val >= 0 else 'red' for val in indicators['MACD_hist']]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(indicators['MACD_hist']))),
                    y=indicators['MACD_hist'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=row, col=1
            )
    
    def add_pattern_highlights(self, fig, patterns, data, row):
        """Add pattern highlights to chart"""
        
        for pattern in patterns[-10:]:  # Show last 10 patterns
            pattern_date = pattern.get('date')
            if pattern_date and pattern_date in data.index:
                idx = data.index.get_loc(pattern_date)
                price = data['Close'].iloc[idx]
                
                # Add pattern marker
                color = '#4CAF50' if pattern.get('direction') == 'Bullish' else '#F44336'
                
                fig.add_trace(
                    go.Scatter(
                        x=[pattern_date],
                        y=[price],
                        mode='markers',
                        name=f'{pattern.get("name", "Pattern")}',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False
                    ),
                    row=row, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=pattern_date,
                    y=price * 1.05,
                    text=f'{pattern.get("name", "Pattern")}<br>{pattern.get("direction", "")}',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    arrowwidth=2,
                    bgcolor=color,
                    bordercolor='white',
                    font=dict(color='white', size=10),
                    row=row, col=1
                )
    
    def add_signal_markers(self, fig, signals, data, row):
        """Add trading signal markers"""
        
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1]
        
        for signal_type, signal in signals.items():
            action = signal.get('action', '')
            
            if action in ['BUY', 'SELL']:
                color = '#4CAF50' if action == 'BUY' else '#F44336'
                symbol = 'triangle-up' if action == 'BUY' else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_date],
                        y=[current_price],
                        mode='markers',
                        name=f'{action} Signal',
                        marker=dict(
                            symbol=symbol,
                            size=15,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False
                    ),
                    row=row, col=1
                )
    
    def add_support_resistance(self, fig, indicators, row):
        """Add support and resistance levels"""
        
        if 'SUPPORT' in indicators and 'RESISTANCE' in indicators:
            # Support line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['SUPPORT']))),
                    y=indicators['SUPPORT'],
                    mode='lines',
                    name='Support',
                    line=dict(color='green', width=2, dash='dash'),
                    opacity=0.7
                ),
                row=row, col=1
            )
            
            # Resistance line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['RESISTANCE']))),
                    y=indicators['RESISTANCE'],
                    mode='lines',
                    name='Resistance',
                    line=dict(color='red', width=2, dash='dash'),
                    opacity=0.7
                ),
                row=row, col=1
            )
    
    def add_vwap_line(self, fig, indicators, row):
        """Add VWAP line"""
        
        if 'VWAP' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(indicators['VWAP']))),
                    y=indicators['VWAP'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='#673AB7', width=2, dash='dot')
                ),
                row=row, col=1
            )
    
    def display_chart_analysis(self, data, indicators, signals, patterns, stock):
        """Display analysis summary below the chart"""
        
        st.markdown("### 📊 Chart Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            st.metric(
                "Current Price", 
                f"₹{current_price:.2f}",
                delta=f"{price_change:+.2f}%"
            )
        
        with col2:
            if 'RSI' in indicators:
                rsi_current = indicators['RSI'][-1]
                rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
                st.metric(
                    "RSI", 
                    f"{rsi_current:.1f}",
                    delta=rsi_status
                )
        
        with col3:
            st.metric(
                "Active Patterns", 
                len(patterns),
                delta="Recent activity"
            )
        
        with col4:
            signal_count = len(signals)
            buy_signals = sum(1 for s in signals.values() if s.get('action') == 'BUY')
            signal_bias = "Bullish" if buy_signals > signal_count/2 else "Bearish" if buy_signals < signal_count/2 else "Neutral"
            
            st.metric(
                "Signal Count", 
                signal_count,
                delta=signal_bias
            )
        
        # Recent patterns table
        if patterns:
            st.markdown("**🔍 Recent Patterns Detected**")
            
            pattern_data = []
            for pattern in patterns[-5:]:  # Last 5 patterns
                pattern_data.append({
                    'Pattern': pattern.get('name', 'Unknown'),
                    'Type': pattern.get('type', 'Unknown'),
                    'Direction': pattern.get('direction', 'Neutral'),
                    'Strength': pattern.get('strength', 'Weak'),
                    'Success Rate': f"{pattern.get('success_rate', 0)}%",
                    'Date': pattern.get('date', 'N/A')
                })
            
            if pattern_data:
                df_patterns = pd.DataFrame(pattern_data)
                st.dataframe(df_patterns, use_container_width=True, hide_index=True)
        
        # Trading signals
        if signals:
            st.markdown("**🎯 Current Trading Signals**")
            
            for signal_type, signal in signals.items():
                action = signal.get('action', 'HOLD')
                strength = signal.get('strength', 'Weak')
                description = signal.get('description', 'No description available')
                
                if action == 'BUY':
                    st.success(f"**{signal_type}**: {action} - {strength}")
                    st.write(f"📝 {description}")
                    if 'entry_price' in signal:
                        st.write(f"💰 Entry: ₹{signal['entry_price']:.2f} | Stop Loss: ₹{signal.get('stop_loss', 0):.2f} | Target: ₹{signal.get('target', 0):.2f}")
                elif action == 'SELL':
                    st.error(f"**{signal_type}**: {action} - {strength}")
                    st.write(f"📝 {description}")
                    if 'entry_price' in signal:
                        st.write(f"💰 Entry: ₹{signal['entry_price']:.2f} | Stop Loss: ₹{signal.get('stop_loss', 0):.2f} | Target: ₹{signal.get('target', 0):.2f}")
                else:
                    st.info(f"**{signal_type}**: {action} - {strength}")
                    st.write(f"📝 {description}")