"""Data fetching module for Indian stocks using Yahoo Finance"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import warnings
warnings.filterwarnings('ignore')

class DataFetcher:
    """Class to handle data fetching from various sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def get_stock_data(self, symbol, timeframe, start_date, end_date, use_cache=True):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS')
            timeframe (str): Timeframe ('1d', '1h', '15m', etc.)
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            use_cache (bool): Whether to use cached data
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return data
        
        try:
            # Ensure symbol has .NS suffix for NSE stocks
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            # Create yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Determine period for intraday data
            if timeframe in ['15m', '30m', '45m', '1h', '4h']:
                # For intraday data, limit to last 60 days
                max_start = datetime.now() - timedelta(days=60)
                if start_date < max_start.date():
                    start_date = max_start.date()
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=timeframe,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                st.warning(f"No data available for {symbol} in timeframe {timeframe}")
                return None
            
            # Clean and validate data
            data = self._clean_data(data)
            
            # Cache the data
            if use_cache:
                self.cache[cache_key] = (time.time(), data)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _clean_data(self, data):
        """Clean and validate the fetched data"""
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure positive values
        data = data[data['High'] > 0]
        data = data[data['Low'] > 0]
        data = data[data['Close'] > 0]
        data = data[data['Volume'] >= 0]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def get_multiple_stocks_data(self, symbols, timeframe, start_date, end_date):
        """Fetch data for multiple stocks"""
        
        all_data = {}
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            data = self.get_stock_data(symbol, timeframe, start_date, end_date)
            if data is not None:
                all_data[symbol] = data
            
            progress_bar.progress((i + 1) / len(symbols))
        
        progress_bar.empty()
        return all_data
    
    def get_stock_info(self, symbol):
        """Get detailed stock information"""
        
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'current_price': info.get('currentPrice', 0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache = {}
        st.success("Data cache cleared successfully!")