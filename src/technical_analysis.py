"""Technical Analysis module with all indicators and signal generation"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysis:
    """Class to handle all technical analysis calculations"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators for the given data"""
        
        if len(data) < 50:  # Need minimum data points
            return {}
        
        indicators = {}
        
        # Price data
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values
        open_price = data['Open'].values
        
        try:
            # Moving Averages
            indicators['EMA_10'] = ta.EMA(close, timeperiod=10)
            indicators['EMA_21'] = ta.EMA(close, timeperiod=21)
            indicators['EMA_50'] = ta.EMA(close, timeperiod=50)
            indicators['SMA_20'] = ta.SMA(close, timeperiod=20)
            indicators['SMA_50'] = ta.SMA(close, timeperiod=50)
            
            # Momentum Indicators
            indicators['RSI'] = ta.RSI(close, timeperiod=14)
            indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = ta.MACD(close)
            indicators['STOCH_K'], indicators['STOCH_D'] = ta.STOCH(high, low, close)
            indicators['CCI'] = ta.CCI(high, low, close, timeperiod=14)
            indicators['ADX'] = ta.ADX(high, low, close, timeperiod=14)
            indicators['AROON_UP'], indicators['AROON_DOWN'] = ta.AROON(high, low, timeperiod=14)
            
            # Volatility Indicators
            indicators['BB_UPPER'], indicators['BB_MIDDLE'], indicators['BB_LOWER'] = ta.BBANDS(close)
            indicators['ATR'] = ta.ATR(high, low, close, timeperiod=14)
            
            # Volume Indicators
            indicators['OBV'] = ta.OBV(close, volume)
            indicators['AD'] = ta.AD(high, low, close, volume)
            indicators['ADOSC'] = ta.ADOSC(high, low, close, volume)
            
            # Trend Indicators
            indicators['SAR'] = ta.SAR(high, low)
            indicators['TEMA'] = ta.TEMA(close, timeperiod=30)
            
            # Custom indicators
            indicators['VWAP'] = self.calculate_vwap(data)
            indicators['PRICE_CHANGE'] = self.calculate_price_change(close)
            indicators['VOLUME_MA'] = ta.SMA(volume, timeperiod=20)
            indicators['VOLUME_RATIO'] = volume / indicators['VOLUME_MA']
            
            # Support and Resistance levels
            indicators['SUPPORT'], indicators['RESISTANCE'] = self.calculate_support_resistance(data)
            
            # Average Daily Range (ADR)
            indicators['ADR'] = self.calculate_adr(data)
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
        
        return indicators
    
    def calculate_vwap(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap.values
    
    def calculate_price_change(self, close: np.ndarray) -> np.ndarray:
        """Calculate percentage price change"""
        
        return np.append([0], np.diff(close) / close[:-1] * 100)
    
    def calculate_adr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average Daily Range"""
        
        daily_range = data['High'] - data['Low']
        adr = daily_range.rolling(window=period).mean()
        return adr.values
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dynamic support and resistance levels"""
        
        high = data['High'].values
        low = data['Low'].values
        
        support = np.zeros(len(data))
        resistance = np.zeros(len(data))
        
        for i in range(window, len(data)):
            # Support: lowest low in the window
            support[i] = np.min(low[i-window:i])
            # Resistance: highest high in the window  
            resistance[i] = np.max(high[i-window:i])
        
        return support, resistance
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Dict]:
        """Generate trading signals based on technical indicators"""
        
        if not indicators:
            return {}
        
        signals = {}
        current_idx = -1  # Latest data point
        
        try:
            current_price = data['Close'].iloc[current_idx]
            current_volume = data['Volume'].iloc[current_idx]
            
            # EMA Crossover Signals
            if len(indicators.get('EMA_10', [])) > 1 and len(indicators.get('EMA_21', [])) > 1:
                ema_10_current = indicators['EMA_10'][current_idx]
                ema_21_current = indicators['EMA_21'][current_idx]
                ema_10_prev = indicators['EMA_10'][current_idx-1]
                ema_21_prev = indicators['EMA_21'][current_idx-1]
                
                if ema_10_prev <= ema_21_prev and ema_10_current > ema_21_current:
                    signals['EMA_CROSS'] = {
                        'action': 'BUY',
                        'strength': 'Strong',
                        'description': 'EMA 10 crossed above EMA 21 - Bullish Breakout Confirmed',
                        'entry_price': current_price,
                        'stop_loss': current_price * 0.98,
                        'target': current_price * 1.06
                    }
                elif ema_10_prev >= ema_21_prev and ema_10_current < ema_21_current:
                    signals['EMA_CROSS'] = {
                        'action': 'SELL',
                        'strength': 'Strong', 
                        'description': 'EMA 10 crossed below EMA 21 - Bearish Breakdown Confirmed',
                        'entry_price': current_price,
                        'stop_loss': current_price * 1.02,
                        'target': current_price * 0.94
                    }
            
            # RSI Signals
            if 'RSI' in indicators:
                rsi_current = indicators['RSI'][current_idx]
                
                if rsi_current < 30:
                    signals['RSI_OVERSOLD'] = {
                        'action': 'BUY',
                        'strength': 'Moderate',
                        'description': f'RSI at {rsi_current:.1f} - Strong Buy Setup (Oversold)',
                        'entry_price': current_price,
                        'stop_loss': current_price * 0.97,
                        'target': current_price * 1.08
                    }
                elif rsi_current > 70:
                    signals['RSI_OVERBOUGHT'] = {
                        'action': 'SELL',
                        'strength': 'Moderate',
                        'description': f'RSI at {rsi_current:.1f} - Strong Sell Setup (Overbought)',
                        'entry_price': current_price,
                        'stop_loss': current_price * 1.03,
                        'target': current_price * 0.92
                    }
            
            # MACD Signals
            if all(key in indicators for key in ['MACD', 'MACD_signal']):
                macd_current = indicators['MACD'][current_idx]
                signal_current = indicators['MACD_signal'][current_idx]
                macd_prev = indicators['MACD'][current_idx-1] 
                signal_prev = indicators['MACD_signal'][current_idx-1]
                
                if macd_prev <= signal_prev and macd_current > signal_current:
                    signals['MACD_CROSS'] = {
                        'action': 'BUY',
                        'strength': 'Strong',
                        'description': 'MACD Bullish Crossover - Momentum Building',
                        'entry_price': current_price,
                        'stop_loss': current_price * 0.98,
                        'target': current_price * 1.05
                    }
                elif macd_prev >= signal_prev and macd_current < signal_current:
                    signals['MACD_CROSS'] = {
                        'action': 'SELL',
                        'strength': 'Strong',
                        'description': 'MACD Bearish Crossover - Momentum Weakening',
                        'entry_price': current_price,
                        'stop_loss': current_price * 1.02,
                        'target': current_price * 0.95
                    }
            
            # Bollinger Bands Signals
            if all(key in indicators for key in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE']):
                bb_upper = indicators['BB_UPPER'][current_idx]
                bb_lower = indicators['BB_LOWER'][current_idx]
                bb_middle = indicators['BB_MIDDLE'][current_idx]
                
                if current_price <= bb_lower:
                    signals['BB_OVERSOLD'] = {
                        'action': 'BUY',
                        'strength': 'Moderate',
                        'description': 'Price at Lower Band - Potential Bounce Setup',
                        'entry_price': current_price,
                        'stop_loss': bb_lower * 0.98,
                        'target': bb_middle
                    }
                elif current_price >= bb_upper:
                    signals['BB_OVERBOUGHT'] = {
                        'action': 'SELL',
                        'strength': 'Moderate',
                        'description': 'Price at Upper Band - Potential Reversal Setup',
                        'entry_price': current_price,
                        'stop_loss': bb_upper * 1.02,
                        'target': bb_middle
                    }
            
            # Volume Analysis
            if 'VOLUME_RATIO' in indicators:
                volume_ratio = indicators['VOLUME_RATIO'][current_idx]
                
                if volume_ratio > 1.5:  # High volume
                    volume_signal = 'Volume Spike Detected'
                    
                    # Enhance existing signals with volume confirmation
                    for signal_key in signals:
                        signals[signal_key]['volume_confirmation'] = True
                        signals[signal_key]['description'] += f' + Volume Spike ({volume_ratio:.1f}x)'
            
            # Support/Resistance Breakout
            if 'SUPPORT' in indicators and 'RESISTANCE' in indicators:
                support_level = indicators['SUPPORT'][current_idx]
                resistance_level = indicators['RESISTANCE'][current_idx]
                
                if current_price > resistance_level * 1.01:  # 1% above resistance
                    signals['BREAKOUT_RESISTANCE'] = {
                        'action': 'BUY',
                        'strength': 'Very Strong',
                        'description': f'Breakout Confirmed above ₹{resistance_level:.2f}',
                        'entry_price': current_price,
                        'stop_loss': resistance_level * 0.99,
                        'target': current_price * 1.08
                    }
                elif current_price < support_level * 0.99:  # 1% below support
                    signals['BREAKDOWN_SUPPORT'] = {
                        'action': 'SELL',
                        'strength': 'Very Strong', 
                        'description': f'Breakdown Confirmed below ₹{support_level:.2f}',
                        'entry_price': current_price,
                        'stop_loss': support_level * 1.01,
                        'target': current_price * 0.92
                    }
        
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return {}
        
        return signals
    
    def get_signal_summary(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Get a summary of all trading signals"""
        
        if not signals:
            return {
                'overall_signal': 'HOLD',
                'signal_count': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'strongest_signal': None
            }
        
        buy_count = sum(1 for s in signals.values() if s.get('action') == 'BUY')
        sell_count = sum(1 for s in signals.values() if s.get('action') == 'SELL')
        
        # Determine overall signal
        if buy_count > sell_count:
            overall = 'BUY'
        elif sell_count > buy_count:
            overall = 'SELL'
        else:
            overall = 'HOLD'
        
        # Find strongest signal
        strength_map = {'Very Strong': 5, 'Strong': 4, 'Moderate': 3, 'Weak': 2, 'Very Weak': 1}
        strongest = None
        max_strength = 0
        
        for signal in signals.values():
            strength_val = strength_map.get(signal.get('strength', 'Weak'), 1)
            if strength_val > max_strength:
                max_strength = strength_val
                strongest = signal
        
        return {
            'overall_signal': overall,
            'signal_count': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'strongest_signal': strongest
        }