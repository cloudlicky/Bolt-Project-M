"""Pattern detection module for candlestick and chart patterns"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, if not available use custom implementations
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available for pattern detection, using custom implementations")

class PatternDetection:
    """Class to detect various trading patterns"""
    
    def __init__(self):
        self.candlestick_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        self.pattern_names = {
            'CDLENGULFING': 'Engulfing Pattern',
            'CDLHAMMER': 'Hammer',
            'CDLDOJI': 'Doji',
            'CDLMORNINGSTAR': 'Morning Star',
            'CDLEVENINGSTAR': 'Evening Star',
            'CDLPIERCING': 'Piercing Pattern',
            'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
            'CDLSHOOTINGSTAR': 'Shooting Star',
            'CDLINVERTEDHAMMER': 'Inverted Hammer',
            'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
            'CDLGRAVESTONEDOJI': 'Gravestone Doji',
            'CDL3WHITESOLDIERS': 'Three White Soldiers',
            'CDL3BLACKCROWS': 'Three Black Crows',
            'CDLHARAMI': 'Harami Pattern',
            'CDLHARAMICROSS': 'Harami Cross'
        }
    
    def detect_all_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect all types of patterns in the data"""
        
        patterns = []
        
        # Detect candlestick patterns
        candlestick_patterns = self.detect_candlestick_patterns(data)
        patterns.extend(candlestick_patterns)
        
        # Detect chart patterns
        chart_patterns = self.detect_chart_patterns(data)
        patterns.extend(chart_patterns)
        
        # Detect breakout patterns
        breakout_patterns = self.detect_breakout_patterns(data)
        patterns.extend(breakout_patterns)
        
        return patterns
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns using TA-Lib or custom implementations"""
        
        if len(data) < 10:
            return []
        
        patterns = []
        
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                for pattern_func in self.candlestick_patterns:
                    if hasattr(ta, pattern_func):
                        # Get the pattern detection function
                        func = getattr(ta, pattern_func)
                        
                        # Detect pattern
                        result = func(open_prices, high_prices, low_prices, close_prices)
                        
                        # Find pattern occurrences
                        pattern_indices = np.where(result != 0)[0]
                        
                        for idx in pattern_indices:
                            if idx >= len(data) - 10:  # Only recent patterns
                                pattern_strength = abs(result[idx])
                                pattern_direction = 'Bullish' if result[idx] > 0 else 'Bearish'
                                
                                patterns.append({
                                    'type': 'candlestick',
                                    'name': self.pattern_names.get(pattern_func, pattern_func),
                                    'index': idx,
                                    'date': data.index[idx] if hasattr(data.index, '__getitem__') else idx,
                                    'direction': pattern_direction,
                                    'strength': pattern_strength,
                                    'description': f"{pattern_direction} {self.pattern_names.get(pattern_func, pattern_func)} detected"
                                })
            else:
                # Use custom implementations
                patterns.extend(self._detect_custom_patterns(data))
                
        except Exception as e:
            print(f"Error detecting candlestick patterns: {str(e)}")
            # Fallback to custom patterns
            patterns.extend(self._detect_custom_patterns(data))
        
        return patterns
    
    def _detect_custom_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Custom candlestick pattern detection without TA-Lib"""
        patterns = []
        
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Check last 10 candles for patterns
        start_idx = max(0, len(data) - 10)
        
        for i in range(start_idx, len(data)):
            # Doji pattern
            if self._is_doji(open_prices[i], high_prices[i], low_prices[i], close_prices[i]):
                patterns.append({
                    'type': 'candlestick',
                    'name': 'Doji',
                    'index': i,
                    'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                    'direction': 'Neutral',
                    'strength': 1,
                    'description': 'Doji pattern detected - market indecision'
                })
            
            # Hammer pattern (need at least 2 candles)
            if i > 0 and self._is_hammer(open_prices[i], high_prices[i], low_prices[i], close_prices[i], close_prices[i-1]):
                patterns.append({
                    'type': 'candlestick',
                    'name': 'Hammer',
                    'index': i,
                    'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                    'direction': 'Bullish',
                    'strength': 2,
                    'description': 'Hammer pattern detected - potential reversal'
                })
            
            # Shooting Star pattern
            if i > 0 and self._is_shooting_star(open_prices[i], high_prices[i], low_prices[i], close_prices[i], close_prices[i-1]):
                patterns.append({
                    'type': 'candlestick',
                    'name': 'Shooting Star',
                    'index': i,
                    'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                    'direction': 'Bearish',
                    'strength': 2,
                    'description': 'Shooting Star pattern detected - potential reversal'
                })
            
            # Engulfing pattern (need at least 2 candles)
            if i > 0:
                engulfing = self._is_engulfing(
                    open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1],
                    open_prices[i], high_prices[i], low_prices[i], close_prices[i]
                )
                if engulfing:
                    direction = 'Bullish' if close_prices[i] > open_prices[i] else 'Bearish'
                    patterns.append({
                        'type': 'candlestick',
                        'name': 'Engulfing Pattern',
                        'index': i,
                        'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                        'direction': direction,
                        'strength': 3,
                        'description': f'{direction} Engulfing pattern detected'
                    })
            
            # Morning Star / Evening Star (need at least 3 candles)
            if i >= 2:
                star_pattern = self._is_star_pattern(
                    open_prices[i-2:i+1], high_prices[i-2:i+1], 
                    low_prices[i-2:i+1], close_prices[i-2:i+1]
                )
                if star_pattern:
                    direction = star_pattern
                    patterns.append({
                        'type': 'candlestick',
                        'name': f'{direction} Star',
                        'index': i,
                        'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                        'direction': direction,
                        'strength': 3,
                        'description': f'{direction} Star pattern detected - strong reversal signal'
                    })
        
        return patterns
    
    def _is_doji(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Check if candle is a Doji pattern"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return False
        
        # Doji: body is very small compared to total range
        return body_size / total_range < 0.1
    
    def _is_hammer(self, open_price: float, high: float, low: float, close: float, prev_close: float) -> bool:
        """Check if candle is a Hammer pattern"""
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        total_range = high - low
        
        if total_range == 0 or body_size == 0:
            return False
        
        # Hammer conditions:
        # 1. Small body at the top
        # 2. Long lower shadow (at least 2x body)
        # 3. Little or no upper shadow
        # 4. Appears after downtrend
        is_small_body = body_size / total_range < 0.3
        is_long_lower_shadow = lower_shadow >= 2 * body_size
        is_small_upper_shadow = upper_shadow < body_size
        is_after_downtrend = close < prev_close
        
        return is_small_body and is_long_lower_shadow and is_small_upper_shadow and is_after_downtrend
    
    def _is_shooting_star(self, open_price: float, high: float, low: float, close: float, prev_close: float) -> bool:
        """Check if candle is a Shooting Star pattern"""
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        total_range = high - low
        
        if total_range == 0 or body_size == 0:
            return False
        
        # Shooting Star conditions:
        # 1. Small body at the bottom
        # 2. Long upper shadow (at least 2x body)
        # 3. Little or no lower shadow
        # 4. Appears after uptrend
        is_small_body = body_size / total_range < 0.3
        is_long_upper_shadow = upper_shadow >= 2 * body_size
        is_small_lower_shadow = lower_shadow < body_size
        is_after_uptrend = close > prev_close
        
        return is_small_body and is_long_upper_shadow and is_small_lower_shadow and is_after_uptrend
    
    def _is_engulfing(self, prev_open: float, prev_high: float, prev_low: float, prev_close: float,
                     curr_open: float, curr_high: float, curr_low: float, curr_close: float) -> bool:
        """Check if current candle engulfs the previous candle"""
        prev_body_top = max(prev_open, prev_close)
        prev_body_bottom = min(prev_open, prev_close)
        curr_body_top = max(curr_open, curr_close)
        curr_body_bottom = min(curr_open, curr_close)
        
        # Current candle body must completely engulf previous candle body
        engulfs = (curr_body_top > prev_body_top) and (curr_body_bottom < prev_body_bottom)
        
        # Different directions (one bullish, one bearish)
        prev_bullish = prev_close > prev_open
        curr_bullish = curr_close > curr_open
        opposite_directions = prev_bullish != curr_bullish
        
        return engulfs and opposite_directions
    
    def _is_star_pattern(self, open_prices: np.ndarray, highs: np.ndarray, 
                        lows: np.ndarray, closes: np.ndarray) -> str:
        """Check for Morning Star or Evening Star pattern"""
        if len(open_prices) < 3:
            return None
        
        # First candle
        first_body = abs(closes[0] - open_prices[0])
        first_bullish = closes[0] > open_prices[0]
        
        # Second candle (star)
        second_body = abs(closes[1] - open_prices[1])
        second_high = max(open_prices[1], closes[1])
        second_low = min(open_prices[1], closes[1])
        
        # Third candle
        third_body = abs(closes[2] - open_prices[2])
        third_bullish = closes[2] > open_prices[2]
        
        # Check for Morning Star (bullish reversal)
        if (not first_bullish and  # First candle bearish
            second_body < first_body * 0.5 and  # Small star body
            second_high < min(open_prices[0], closes[0]) and  # Star gaps down
            third_bullish and  # Third candle bullish
            closes[2] > (open_prices[0] + closes[0]) / 2):  # Third closes above mid of first
            return "Morning"
        
        # Check for Evening Star (bearish reversal)
        if (first_bullish and  # First candle bullish
            second_body < first_body * 0.5 and  # Small star body
            second_low > max(open_prices[0], closes[0]) and  # Star gaps up
            not third_bullish and  # Third candle bearish
            closes[2] < (open_prices[0] + closes[0]) / 2):  # Third closes below mid of first
            return "Evening"
        
        return None
    
    def detect_chart_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns like triangles, wedges, flags, etc."""
        
        if len(data) < 50:
            return []
        
        patterns = []
        
        # Detect triangles
        triangle_patterns = self._detect_triangles(data)
        patterns.extend(triangle_patterns)
        
        # Detect wedges
        wedge_patterns = self._detect_wedges(data)
        patterns.extend(wedge_patterns)
        
        # Detect flags and pennants
        flag_patterns = self._detect_flags(data)
        patterns.extend(flag_patterns)
        
        # Detect head and shoulders
        hs_patterns = self._detect_head_shoulders(data)
        patterns.extend(hs_patterns)
        
        # Detect double tops/bottoms
        double_patterns = self._detect_double_patterns(data)
        patterns.extend(double_patterns)
        
        # Detect cup and handle
        cup_handle_patterns = self._detect_cup_handle(data)
        patterns.extend(cup_handle_patterns)
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect triangle patterns"""
        
        patterns = []
        window = 20
        
        if len(data) < window * 2:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Look for triangle patterns in the last portion of data
        for i in range(len(data) - window, len(data)):
            if i < window:
                continue
            
            # Get recent highs and lows
            recent_highs = high_prices[i-window:i]
            recent_lows = low_prices[i-window:i]
            
            # Find peaks and troughs
            high_peaks, _ = find_peaks(recent_highs, distance=5)
            low_peaks, _ = find_peaks(-recent_lows, distance=5)
            
            if len(high_peaks) >= 2 and len(low_peaks) >= 2:
                # Calculate trend lines
                high_slope = self._calculate_slope(high_peaks, recent_highs[high_peaks])
                low_slope = self._calculate_slope(low_peaks, recent_lows[low_peaks])
                
                # Determine triangle type
                if abs(high_slope) < 0.1 and low_slope > 0.1:
                    triangle_type = "Ascending Triangle"
                    direction = "Bullish"
                    description = "Ascending Triangle - Bullish Breakout Expected"
                elif high_slope < -0.1 and abs(low_slope) < 0.1:
                    triangle_type = "Descending Triangle"
                    direction = "Bearish"
                    description = "Descending Triangle - Bearish Breakdown Expected"
                elif high_slope < -0.1 and low_slope > 0.1:
                    triangle_type = "Symmetrical Triangle"
                    direction = "Neutral"
                    description = "Symmetrical Triangle - Wait for Breakout Direction"
                else:
                    continue
                
                patterns.append({
                    'type': 'Chart Pattern',
                    'name': triangle_type,
                    'direction': direction,
                    'strength': 'Moderate',
                    'date': data.index[i],
                    'price': close_prices[i],
                    'description': description,
                    'success_rate': 65,
                    'trade_suggestion': self._get_triangle_trade_suggestion(triangle_type, close_prices[i])
                })
        
        return patterns
    
    def _detect_wedges(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect wedge patterns"""
        
        patterns = []
        window = 15
        
        if len(data) < window * 2:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        for i in range(len(data) - window, len(data)):
            if i < window:
                continue
            
            recent_highs = high_prices[i-window:i]
            recent_lows = low_prices[i-window:i]
            
            # Calculate overall trends
            high_trend = linregress(range(window), recent_highs).slope
            low_trend = linregress(range(window), recent_lows).slope
            
            # Wedge conditions
            if high_trend > 0 and low_trend > 0 and high_trend < low_trend:
                wedge_type = "Rising Wedge"
                direction = "Bearish"
                description = "Rising Wedge - Bearish Reversal Setup"
            elif high_trend < 0 and low_trend < 0 and high_trend > low_trend:
                wedge_type = "Falling Wedge"
                direction = "Bullish"
                description = "Falling Wedge - Bullish Reversal Setup"
            else:
                continue
            
            patterns.append({
                'type': 'Chart Pattern',
                'name': wedge_type,
                'direction': direction,
                'strength': 'Strong',
                'date': data.index[i],
                'price': close_prices[i],
                'description': description,
                'success_rate': 70,
                'trade_suggestion': self._get_wedge_trade_suggestion(wedge_type, close_prices[i])
            })
        
        return patterns
    
    def _detect_flags(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect flag and pennant patterns"""
        
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        close_prices = data['Close'].values
        volume = data['Volume'].values
        
        # Look for flag patterns (consolidation after strong move)
        for i in range(20, len(data) - 10):
            # Check for strong prior move (flagpole)
            flagpole_start = max(0, i - 15)
            flagpole_move = (close_prices[i] - close_prices[flagpole_start]) / close_prices[flagpole_start]
            
            if abs(flagpole_move) > 0.05:  # 5% move for flagpole
                # Check for consolidation (flag)
                flag_data = close_prices[i:i+10]
                flag_volatility = np.std(flag_data) / np.mean(flag_data)
                
                if flag_volatility < 0.02:  # Low volatility consolidation
                    if flagpole_move > 0:
                        flag_type = "Bull Flag"
                        direction = "Bullish"
                        description = "Bull Flag - Continuation Pattern"
                    else:
                        flag_type = "Bear Flag"
                        direction = "Bearish"
                        description = "Bear Flag - Continuation Pattern"
                    
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': flag_type,
                        'direction': direction,
                        'strength': 'Strong',
                        'date': data.index[i+5],
                        'price': close_prices[i+5],
                        'description': description,
                        'success_rate': 75,
                        'trade_suggestion': self._get_flag_trade_suggestion(flag_type, close_prices[i+5])
                    })
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect head and shoulders patterns"""
        
        patterns = []
        
        if len(data) < 50:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Find significant peaks for head and shoulders
        peaks, properties = find_peaks(high_prices, distance=10, prominence=np.std(high_prices))
        
        if len(peaks) >= 3:
            # Look for head and shoulders pattern in recent peaks
            recent_peaks = peaks[-5:] if len(peaks) > 5 else peaks
            
            for i in range(len(recent_peaks) - 2):
                left_shoulder = recent_peaks[i]
                head = recent_peaks[i + 1]
                right_shoulder = recent_peaks[i + 2]
                
                left_height = high_prices[left_shoulder]
                head_height = high_prices[head]
                right_height = high_prices[right_shoulder]
                
                # Check if head is higher than shoulders
                if (head_height > left_height and head_height > right_height and
                    abs(left_height - right_height) / max(left_height, right_height) < 0.05):
                    
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': 'Head and Shoulders',
                        'direction': 'Bearish',
                        'strength': 'Very Strong',
                        'date': data.index[right_shoulder],
                        'price': close_prices[right_shoulder],
                        'description': 'Head and Shoulders - Strong Bearish Reversal Setup',
                        'success_rate': 85,
                        'trade_suggestion': {
                            'action': 'SELL',
                            'entry': close_prices[right_shoulder],
                            'stop_loss': head_height * 1.02,
                            'target': close_prices[right_shoulder] * 0.90
                        }
                    })
        
        # Find significant troughs for inverse head and shoulders
        troughs, properties = find_peaks(-low_prices, distance=10, prominence=np.std(low_prices))
        
        if len(troughs) >= 3:
            recent_troughs = troughs[-5:] if len(troughs) > 5 else troughs
            
            for i in range(len(recent_troughs) - 2):
                left_shoulder = recent_troughs[i]
                head = recent_troughs[i + 1]
                right_shoulder = recent_troughs[i + 2]
                
                left_depth = low_prices[left_shoulder]
                head_depth = low_prices[head]
                right_depth = low_prices[right_shoulder]
                
                # Check if head is lower than shoulders
                if (head_depth < left_depth and head_depth < right_depth and
                    abs(left_depth - right_depth) / max(left_depth, right_depth) < 0.05):
                    
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': 'Inverse Head and Shoulders',
                        'direction': 'Bullish',
                        'strength': 'Very Strong',
                        'date': data.index[right_shoulder],
                        'price': close_prices[right_shoulder],
                        'description': 'Inverse Head and Shoulders - Strong Bullish Reversal Setup',
                        'success_rate': 85,
                        'trade_suggestion': {
                            'action': 'BUY',
                            'entry': close_prices[right_shoulder],
                            'stop_loss': head_depth * 0.98,
                            'target': close_prices[right_shoulder] * 1.10
                        }
                    })
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect double top and double bottom patterns"""
        
        patterns = []
        
        if len(data) < 40:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Double Top Detection
        peaks, _ = find_peaks(high_prices, distance=15, prominence=np.std(high_prices))
        
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]
                
                height1 = high_prices[peak1]
                height2 = high_prices[peak2]
                
                # Check if peaks are similar height (within 2%)
                if abs(height1 - height2) / max(height1, height2) < 0.02:
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': 'Double Top',
                        'direction': 'Bearish',
                        'strength': 'Strong',
                        'date': data.index[peak2],
                        'price': close_prices[peak2],
                        'description': 'Double Top - Bearish Reversal Pattern',
                        'success_rate': 78,
                        'trade_suggestion': {
                            'action': 'SELL',
                            'entry': close_prices[peak2] * 0.98,
                            'stop_loss': max(height1, height2) * 1.02,
                            'target': close_prices[peak2] * 0.90
                        }
                    })
        
        # Double Bottom Detection
        troughs, _ = find_peaks(-low_prices, distance=15, prominence=np.std(low_prices))
        
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1 = troughs[i]
                trough2 = troughs[i + 1]
                
                depth1 = low_prices[trough1]
                depth2 = low_prices[trough2]
                
                # Check if troughs are similar depth (within 2%)
                if abs(depth1 - depth2) / max(depth1, depth2) < 0.02:
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': 'Double Bottom',
                        'direction': 'Bullish',
                        'strength': 'Strong',
                        'date': data.index[trough2],
                        'price': close_prices[trough2],
                        'description': 'Double Bottom - Bullish Reversal Pattern',
                        'success_rate': 78,
                        'trade_suggestion': {
                            'action': 'BUY',
                            'entry': close_prices[trough2] * 1.02,
                            'stop_loss': min(depth1, depth2) * 0.98,
                            'target': close_prices[trough2] * 1.10
                        }
                    })
        
        return patterns
    
    def _detect_cup_handle(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect cup and handle patterns"""
        
        patterns = []
        
        if len(data) < 60:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Look for cup and handle in recent data
        for i in range(40, len(data) - 20):
            # Cup formation (U-shaped)
            cup_start = i - 30
            cup_bottom = i - 15
            cup_end = i
            
            if cup_start < 0:
                continue
            
            start_price = high_prices[cup_start]
            bottom_price = low_prices[cup_bottom]
            end_price = high_prices[cup_end]
            
            # Check for cup formation
            cup_depth = (start_price - bottom_price) / start_price
            
            if (0.12 <= cup_depth <= 0.33 and  # 12-33% depth
                abs(start_price - end_price) / start_price < 0.05):  # Similar rim heights
                
                # Look for handle formation
                handle_start = i
                handle_end = min(i + 15, len(data) - 1)
                
                handle_high = np.max(high_prices[handle_start:handle_end])
                handle_low = np.min(low_prices[handle_start:handle_end])
                
                handle_depth = (handle_high - handle_low) / handle_high
                
                if handle_depth < 0.15:  # Handle depth < 15%
                    patterns.append({
                        'type': 'Chart Pattern',
                        'name': 'Cup and Handle',
                        'direction': 'Bullish',
                        'strength': 'Strong',
                        'date': data.index[handle_end],
                        'price': close_prices[handle_end],
                        'description': 'Cup and Handle - Strong Bullish Continuation',
                        'success_rate': 80,
                        'trade_suggestion': {
                            'action': 'BUY',
                            'entry': handle_high * 1.01,
                            'stop_loss': handle_low * 0.98,
                            'target': handle_high * (1 + cup_depth)
                        }
                    })
        
        return patterns
    
    def detect_breakout_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect breakout and breakdown patterns"""
        
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        volume = data['Volume'].values
        
        # Calculate support and resistance levels
        window = 20
        
        for i in range(window, len(data)):
            recent_highs = high_prices[i-window:i]
            recent_lows = low_prices[i-window:i]
            recent_volume = volume[i-window:i]
            
            resistance = np.max(recent_highs)
            support = np.min(recent_lows)
            avg_volume = np.mean(recent_volume)
            
            current_price = close_prices[i]
            current_volume = volume[i]
            
            # Breakout above resistance
            if (current_price > resistance * 1.01 and  # 1% above resistance
                current_volume > avg_volume * 1.5):    # High volume
                
                patterns.append({
                    'type': 'Breakout',
                    'name': 'Resistance Breakout',
                    'direction': 'Bullish',
                    'strength': 'Very Strong',
                    'date': data.index[i],
                    'price': current_price,
                    'description': f'Strong Breakout Confirmed above ₹{resistance:.2f}',
                    'success_rate': 75,
                    'trade_suggestion': {
                        'action': 'BUY',
                        'entry': current_price,
                        'stop_loss': resistance * 0.99,
                        'target': current_price * 1.08
                    }
                })
            
            # Breakdown below support
            elif (current_price < support * 0.99 and  # 1% below support
                  current_volume > avg_volume * 1.5):   # High volume
                
                patterns.append({
                    'type': 'Breakdown',
                    'name': 'Support Breakdown',
                    'direction': 'Bearish',
                    'strength': 'Very Strong',
                    'date': data.index[i],
                    'price': current_price,
                    'description': f'Strong Breakdown Confirmed below ₹{support:.2f}',
                    'success_rate': 75,
                    'trade_suggestion': {
                        'action': 'SELL',
                        'entry': current_price,
                        'stop_loss': support * 1.01,
                        'target': current_price * 0.92
                    }
                })
        
        return patterns
    
    def _calculate_slope(self, x_values, y_values):
        """Calculate slope of a trend line"""
        if len(x_values) < 2:
            return 0
        return linregress(x_values, y_values).slope
    
    def _get_strength_label(self, strength_value):
        """Convert numeric strength to label"""
        if strength_value >= 100:
            return "Very Strong"
        elif strength_value >= 80:
            return "Strong"
        elif strength_value >= 60:
            return "Moderate" 
        elif strength_value >= 40:
            return "Weak"
        else:
            return "Very Weak"
    
    def _get_pattern_description(self, pattern_func, direction):
        """Get description for candlestick pattern"""
        descriptions = {
            'CDLENGULFING': f'{direction} Engulfing - Strong Reversal Setup',
            'CDLHAMMER': 'Hammer - Bullish Reversal at Support',
            'CDLDOJI': 'Doji - Indecision, Potential Reversal',
            'CDLMORNINGSTAR': 'Morning Star - Strong Bullish Reversal',
            'CDLEVENINGSTAR': 'Evening Star - Strong Bearish Reversal',
            'CDLPIERCING': 'Piercing Pattern - Bullish Reversal',
            'CDLDARKCLOUDCOVER': 'Dark Cloud Cover - Bearish Reversal',
            'CDLSHOOTINGSTAR': 'Shooting Star - Bearish Reversal at Resistance',
            'CDLINVERTEDHAMMER': 'Inverted Hammer - Potential Bullish Reversal',
            'CDLDRAGONFLYDOJI': 'Dragonfly Doji - Bullish Reversal Signal',
            'CDLGRAVESTONEDOJI': 'Gravestone Doji - Bearish Reversal Signal',
            'CDL3WHITESOLDIERS': 'Three White Soldiers - Strong Bullish Momentum',
            'CDL3BLACKCROWS': 'Three Black Crows - Strong Bearish Momentum'
        }
        return descriptions.get(pattern_func, f'{direction} candlestick pattern detected')
    
    def _get_pattern_success_rate(self, pattern_func):
        """Get historical success rate for pattern"""
        success_rates = {
            'CDLENGULFING': 80,
            'CDLHAMMER': 75,
            'CDLDOJI': 60,
            'CDLMORNINGSTAR': 85,
            'CDLEVENINGSTAR': 85,
            'CDLPIERCING': 70,
            'CDLDARKCLOUDCOVER': 70,
            'CDLSHOOTINGSTAR': 72,
            'CDLINVERTEDHAMMER': 68,
            'CDLDRAGONFLYDOJI': 73,
            'CDLGRAVESTONEDOJI': 73,
            'CDL3WHITESOLDIERS': 78,
            'CDL3BLACKCROWS': 78
        }
        return success_rates.get(pattern_func, 65)
    
    def _get_trade_suggestion(self, pattern_func, direction, price):
        """Get trade suggestion for pattern"""
        if direction == 'Bullish':
            return {
                'action': 'BUY',
                'entry': price,
                'stop_loss': price * 0.97,
                'target': price * 1.06
            }
        else:
            return {
                'action': 'SELL',
                'entry': price,
                'stop_loss': price * 1.03,
                'target': price * 0.94
            }
    
    def _get_triangle_trade_suggestion(self, triangle_type, price):
        """Get trade suggestion for triangle pattern"""
        if triangle_type == "Ascending Triangle":
            return {
                'action': 'BUY',
                'entry': price * 1.02,  # Buy on breakout
                'stop_loss': price * 0.97,
                'target': price * 1.08
            }
        elif triangle_type == "Descending Triangle":
            return {
                'action': 'SELL',
                'entry': price * 0.98,  # Sell on breakdown
                'stop_loss': price * 1.03,
                'target': price * 0.92
            }
        else:
            return {
                'action': 'WAIT',
                'entry': 'Wait for Breakout Direction',
                'stop_loss': 'Set after breakout',
                'target': 'Set after breakout'
            }
    
    def _get_wedge_trade_suggestion(self, wedge_type, price):
        """Get trade suggestion for wedge pattern"""
        if wedge_type == "Rising Wedge":
            return {
                'action': 'SELL',
                'entry': price,
                'stop_loss': price * 1.03,
                'target': price * 0.92
            }
        else:  # Falling Wedge
            return {
                'action': 'BUY',
                'entry': price,
                'stop_loss': price * 0.97,
                'target': price * 1.08
            }
    
    def _get_flag_trade_suggestion(self, flag_type, price):
        """Get trade suggestion for flag pattern"""
        if flag_type == "Bull Flag":
            return {
                'action': 'BUY',
                'entry': price * 1.01,
                'stop_loss': price * 0.97,
                'target': price * 1.10
            }
        else:  # Bear Flag
            return {
                'action': 'SELL',
                'entry': price * 0.99,
                'stop_loss': price * 1.03,
                'target': price * 0.90
            }