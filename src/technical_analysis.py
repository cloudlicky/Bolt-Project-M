"""Technical Analysis module with all indicators and signal generation"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, if not available use custom implementations
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using custom implementations")

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
            if TALIB_AVAILABLE:
                indicators['EMA_10'] = ta.EMA(close, timeperiod=10)
                indicators['EMA_21'] = ta.EMA(close, timeperiod=21)
                indicators['EMA_50'] = ta.EMA(close, timeperiod=50)
                indicators['SMA_20'] = ta.SMA(close, timeperiod=20)
                indicators['SMA_50'] = ta.SMA(close, timeperiod=50)
            else:
                indicators['EMA_10'] = self.calculate_ema(close, 10)
                indicators['EMA_21'] = self.calculate_ema(close, 21)
                indicators['EMA_50'] = self.calculate_ema(close, 50)
                indicators['SMA_20'] = self.calculate_sma(close, 20)
                indicators['SMA_50'] = self.calculate_sma(close, 50)
            
            # Momentum Indicators
            if TALIB_AVAILABLE:
                indicators['RSI'] = ta.RSI(close, timeperiod=14)
                indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = ta.MACD(close)
                indicators['STOCH_K'], indicators['STOCH_D'] = ta.STOCH(high, low, close)
                indicators['CCI'] = ta.CCI(high, low, close, timeperiod=14)
                indicators['ADX'] = ta.ADX(high, low, close, timeperiod=14)
                indicators['AROON_UP'], indicators['AROON_DOWN'] = ta.AROON(high, low, timeperiod=14)
            else:
                indicators['RSI'] = self.calculate_rsi(close, 14)
                indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = self.calculate_macd(close)
                indicators['STOCH_K'], indicators['STOCH_D'] = self.calculate_stochastic(high, low, close)
                indicators['CCI'] = self.calculate_cci(high, low, close, 14)
                indicators['ADX'] = self.calculate_adx(high, low, close, 14)
                indicators['AROON_UP'], indicators['AROON_DOWN'] = self.calculate_aroon(high, low, 14)
            
            # Volatility Indicators
            if TALIB_AVAILABLE:
                indicators['BB_UPPER'], indicators['BB_MIDDLE'], indicators['BB_LOWER'] = ta.BBANDS(close)
                indicators['ATR'] = ta.ATR(high, low, close, timeperiod=14)
            else:
                indicators['BB_UPPER'], indicators['BB_MIDDLE'], indicators['BB_LOWER'] = self.calculate_bollinger_bands(close)
                indicators['ATR'] = self.calculate_atr(high, low, close, 14)
            
            # Volume Indicators
            if TALIB_AVAILABLE:
                indicators['OBV'] = ta.OBV(close, volume)
                indicators['AD'] = ta.AD(high, low, close, volume)
                indicators['ADOSC'] = ta.ADOSC(high, low, close, volume)
            else:
                indicators['OBV'] = self.calculate_obv(close, volume)
                indicators['AD'] = self.calculate_ad(high, low, close, volume)
                indicators['ADOSC'] = self.calculate_adosc(high, low, close, volume)
            
            # Trend Indicators
            if TALIB_AVAILABLE:
                indicators['SAR'] = ta.SAR(high, low)
                indicators['TEMA'] = ta.TEMA(close, timeperiod=30)
            else:
                indicators['SAR'] = self.calculate_sar(high, low)
                indicators['TEMA'] = self.calculate_tema(close, 30)
            
            # Custom indicators
            indicators['VWAP'] = self.calculate_vwap(data)
            indicators['PRICE_CHANGE'] = self.calculate_price_change(close)
            indicators['VOLUME_MA'] = self.calculate_sma(volume, 20)
            indicators['VOLUME_RATIO'] = volume / indicators['VOLUME_MA']
            
            # Support and Resistance levels
            indicators['SUPPORT'], indicators['RESISTANCE'] = self.calculate_support_resistance(data)
            
            # Average Daily Range (ADR)
            indicators['ADR'] = self.calculate_adr(data)
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
        
        return indicators
    
    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        return sma
    
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.full_like(data, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Initialize with first value
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(data), np.nan)
        avg_loss = np.full(len(data), np.nan)
        
        # Initial averages
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Calculate smoothed averages
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, period)
        std = np.full_like(data, np.nan)
        
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        stoch_k = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            lowest_low = np.min(low[i - period + 1:i + 1])
            highest_high = np.max(high[i - period + 1:i + 1])
            
            if highest_high != lowest_low:
                stoch_k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                stoch_k[i] = 50
        
        stoch_d = self.calculate_sma(stoch_k, 3)
        
        return stoch_k, stoch_d
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        true_range = np.zeros(len(close))
        
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # Calculate ATR using EMA
        return self.calculate_ema(true_range, period)
    
    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def calculate_ad(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        ad = np.zeros(len(close))
        
        for i in range(len(close)):
            if high[i] != low[i]:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                clv = 0
            ad[i] = ad[i-1] + clv * volume[i] if i > 0 else clv * volume[i]
        
        return ad
    
    def calculate_adosc(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, fast: int = 3, slow: int = 10) -> np.ndarray:
        """Calculate Accumulation/Distribution Oscillator"""
        ad = self.calculate_ad(high, low, close, volume)
        fast_ema = self.calculate_ema(ad, fast)
        slow_ema = self.calculate_ema(ad, slow)
        
        return fast_ema - slow_ema
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = self.calculate_sma(typical_price, period)
        
        cci = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            mean_deviation = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp[i]))
            if mean_deviation != 0:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)
        
        return cci
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        tr = self.calculate_atr(high, low, close, 1)
        dm_plus = np.zeros(len(close))
        dm_minus = np.zeros(len(close))
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
        
        di_plus = 100 * self.calculate_ema(dm_plus, period) / self.calculate_ema(tr, period)
        di_minus = 100 * self.calculate_ema(dm_minus, period) / self.calculate_ema(tr, period)
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = self.calculate_ema(dx, period)
        
        return adx
    
    def calculate_aroon(self, high: np.ndarray, low: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Aroon Up and Aroon Down"""
        aroon_up = np.full_like(high, np.nan)
        aroon_down = np.full_like(low, np.nan)
        
        for i in range(period, len(high)):
            high_period = high[i - period:i + 1]
            low_period = low[i - period:i + 1]
            
            high_idx = np.argmax(high_period)
            low_idx = np.argmin(low_period)
            
            aroon_up[i] = ((period - high_idx) / period) * 100
            aroon_down[i] = ((period - low_idx) / period) * 100
        
        return aroon_up, aroon_down
    
    def calculate_sar(self, high: np.ndarray, low: np.ndarray, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """Calculate Parabolic SAR"""
        sar = np.full_like(high, np.nan)
        trend = np.full_like(high, np.nan)
        ep = np.full_like(high, np.nan)
        af = np.full_like(high, np.nan)
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        ep[0] = high[0]
        af[0] = acceleration
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = acceleration
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = acceleration
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return sar
    
    def calculate_tema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Triple Exponential Moving Average"""
        ema1 = self.calculate_ema(data, period)
        ema2 = self.calculate_ema(ema1, period)
        ema3 = self.calculate_ema(ema2, period)
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return tema
    
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