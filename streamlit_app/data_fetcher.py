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
    
    def get_nse_data(self, symbol, timeframe="1d"):
        """
        Alternative method to fetch NSE data directly
        (Backup method if Yahoo Finance fails)
        """
        try:
            # This would typically use NSE API or web scraping
            # For now, fallback to Yahoo Finance
            return self.get_stock_data(f"{symbol}.NS", timeframe, 
                                     datetime.now() - timedelta(days=365),
                                     datetime.now())
        except Exception as e:
            st.warning(f"NSE data fetch failed for {symbol}: {str(e)}")
            return None
    
    def get_sector_data(self, sector="NIFTY50"):
        """Get sector-wise data"""
        
        sector_symbols = {
            "NIFTY50": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HDFC.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS"
            ],
            "BANKNIFTY": [
                "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS",
                "AXISBANK.NS", "INDUSINDBK.NS", "BANDHANBNK.NS"
            ],
            "NIFTYIT": [
                "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
            ]
        }
        
        if sector not in sector_symbols:
            return {}
        
        return self.get_multiple_stocks_data(
            sector_symbols[sector], 
            "1d",
            datetime.now() - timedelta(days=90),
            datetime.now()
        )
    
    def get_market_overview(self):
        """Get overall market overview"""
        
        indices = {
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN", 
            "BANKNIFTY": "^NSEBANK",
            "NIFTYIT": "^CNXIT"
        }
        
        overview = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1d")
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    
                    overview[name] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct
                    }
            except Exception:
                continue
        
        return overview
    
    def validate_symbol(self, symbol):
        """Validate if a stock symbol exists and has data"""
        
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we can get basic info
            if 'regularMarketPrice' in info or 'currentPrice' in info:
                return True
            else:
                return False
                
        except Exception:
            return False
    
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