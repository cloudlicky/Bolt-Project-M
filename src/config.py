"""Configuration file for the Swing Trading Analysis App"""

# Indian Stock symbols with company names (100 stocks across market caps)
INDIAN_STOCKS = {
    # Nifty 50 - Large Cap (30 stocks)
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services Limited", 
    "HDFCBANK": "HDFC Bank Limited",
    "INFY": "Infosys Limited",
    "HDFC": "Housing Development Finance Corporation Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "KOTAKBANK": "Kotak Mahindra Bank Limited",
    "HINDUNILVR": "Hindustan Unilever Limited",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Limited",
    "ITC": "ITC Limited",
    "AXISBANK": "Axis Bank Limited",
    "LT": "Larsen & Toubro Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "MARUTI": "Maruti Suzuki India Limited",
    "SUNPHARMA": "Sun Pharmaceutical Industries Limited",
    "TITAN": "Titan Company Limited",
    "ULTRACEMCO": "UltraTech Cement Limited",
    "ONGC": "Oil and Natural Gas Corporation Limited",
    "NESTLEIND": "Nestle India Limited",
    "POWERGRID": "Power Grid Corporation of India Limited",
    "NTPC": "NTPC Limited",
    "TECHM": "Tech Mahindra Limited",
    "HCLTECH": "HCL Technologies Limited",
    "WIPRO": "Wipro Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "TATAMOTORS": "Tata Motors Limited",
    "DRREDDY": "Dr. Reddy's Laboratories Limited",
    "CIPLA": "Cipla Limited",
    
    # Bank Nifty Additional (10 stocks)
    "INDUSINDBK": "IndusInd Bank Limited",
    "BANDHANBNK": "Bandhan Bank Limited",
    "FEDERALBNK": "Federal Bank Limited",
    "IDFCFIRSTB": "IDFC First Bank Limited",
    "PNB": "Punjab National Bank",
    "BANKBARODA": "Bank of Baroda",
    "CANBK": "Canara Bank",
    "UNIONBANK": "Union Bank of India",
    "RBLBANK": "RBL Bank Limited",
    "AUBANK": "AU Small Finance Bank Limited",
    
    # Mid Cap Stocks (35 stocks)
    "ADANIPORTS": "Adani Ports and Special Economic Zone Limited",
    "ADANIGREEN": "Adani Green Energy Limited",
    "JINDALSTEL": "Jindal Steel & Power Limited",
    "TATAPOWER": "Tata Power Company Limited",
    "SAIL": "Steel Authority of India Limited",
    "COALINDIA": "Coal India Limited",
    "IOC": "Indian Oil Corporation Limited",
    "BPCL": "Bharat Petroleum Corporation Limited",
    "HPCL": "Hindustan Petroleum Corporation Limited",
    "GODREJCP": "Godrej Consumer Products Limited",
    "HDFCLIFE": "HDFC Life Insurance Company Limited",
    "SBILIFE": "SBI Life Insurance Company Limited",
    "ICICIPRULI": "ICICI Prudential Life Insurance Company Limited",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Limited",
    "DIVISLAB": "Divi's Laboratories Limited",
    "BIOCON": "Biocon Limited",
    "LUPIN": "Lupin Limited",
    "CADILAHC": "Cadila Healthcare Limited",
    "TORNTPHARM": "Torrent Pharmaceuticals Limited",
    "GLENMARK": "Glenmark Pharmaceuticals Limited",
    "MOTHERSUMI": "Motherson Sumi Systems Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "HEROMOTOCO": "Hero MotoCorp Limited",
    "EICHERMOT": "Eicher Motors Limited",
    "TVSMOTOR": "TVS Motor Company Limited",
    "ASHOKLEY": "Ashok Leyland Limited",
    "M&M": "Mahindra & Mahindra Limited",
    "ESCORTS": "Escorts Limited",
    "BHARATFORG": "Bharat Forge Limited",
    "CUMMINSIND": "Cummins India Limited",
    "BOSCHLTD": "Bosch Limited",
    "EXIDEIND": "Exide Industries Limited",
    "AMBUJACEM": "Ambuja Cements Limited",
    "ACC": "ACC Limited",
    "SHREECEM": "Shree Cement Limited",
    
    # Small Cap Stocks (25 stocks)
    "ZEEL": "Zee Entertainment Enterprises Limited",
    "YESBANK": "Yes Bank Limited",
    "RPOWER": "Reliance Power Limited",
    "SUZLON": "Suzlon Energy Limited",
    "JPASSOCIAT": "Jaiprakash Associates Limited",
    "RCOM": "Reliance Communications Limited",
    "DHFL": "Dewan Housing Finance Corporation Limited",
    "JETAIRWAYS": "Jet Airways (India) Limited",
    "PCJEWELLER": "PC Jeweller Limited",
    "GMRINFRA": "GMR Infrastructure Limited",
    "IDEA": "Vodafone Idea Limited",
    "SPICEJET": "SpiceJet Limited",
    "INDIGO": "InterGlobe Aviation Limited",
    "JUBLFOOD": "Jubilant FoodWorks Limited",
    "MCDOWELL-N": "United Spirits Limited",
    "UBL": "United Breweries Limited",
    "RADICO": "Radico Khaitan Limited",
    "RELAXO": "Relaxo Footwears Limited",
    "BATAINDIA": "Bata India Limited",
    "PAGEIND": "Page Industries Limited",
    "AUROPHARMA": "Aurobindo Pharma Limited",
    "SUNPHARMA": "Sun Pharmaceutical Industries Limited",
    "ALKEM": "Alkem Laboratories Limited",
    "LALPATHLAB": "Dr. Lal PathLabs Limited",
    "METROPOLIS": "Metropolis Healthcare Limited"
}

# Timeframes with yfinance intervals
TIMEFRAMES = {
    "15m": "15m",
    "30m": "30m", 
    "45m": "45m",
    "1H": "1h",
    "4H": "4h", 
    "1D": "1d",
    "1W": "1wk",
    "1M": "1mo"
}

# Market cap classification
MARKET_CAPS = {
    "Large Cap": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC", "ICICIBANK", "KOTAKBANK", 
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "AXISBANK", "LT", "ASIANPAINT", 
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "ONGC", "NESTLEIND", 
        "POWERGRID", "NTPC", "TECHM", "HCLTECH", "WIPRO", "BAJFINANCE", 
        "BAJAJFINSV", "TATAMOTORS", "DRREDDY", "CIPLA"
    ],
    "Mid Cap": [
        "ADANIPORTS", "ADANIGREEN", "JINDALSTEL", "TATAPOWER", "SAIL", "COALINDIA", 
        "IOC", "BPCL", "HPCL", "GODREJCP", "HDFCLIFE", "SBILIFE", "ICICIPRULI", 
        "APOLLOHOSP", "DIVISLAB", "BIOCON", "LUPIN", "CADILAHC", "TORNTPHARM", 
        "GLENMARK", "MOTHERSUMI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", 
        "TVSMOTOR", "ASHOKLEY", "M&M", "ESCORTS", "BHARATFORG", "CUMMINSIND", 
        "BOSCHLTD", "EXIDEIND", "AMBUJACEM", "ACC", "SHREECEM"
    ],
    "Small Cap": [
        "ZEEL", "YESBANK", "RPOWER", "SUZLON", "JPASSOCIAT", "RCOM", "DHFL", 
        "JETAIRWAYS", "PCJEWELLER", "GMRINFRA", "IDEA", "SPICEJET", "INDIGO", 
        "JUBLFOOD", "MCDOWELL-N", "UBL", "RADICO", "RELAXO", "BATAINDIA", 
        "PAGEIND", "AUROPHARMA", "ALKEM", "LALPATHLAB", "METROPOLIS"
    ]
}

# Index compositions for easy selection
INDICES = {
    "Nifty 50": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC", "ICICIBANK", "KOTAKBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "AXISBANK", "LT", "ASIANPAINT",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "ONGC", "NESTLEIND"
    ],
    "Bank Nifty": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "INDUSINDBK",
        "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "BANKBARODA"
    ],
    "Nifty IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTTS", "MINDTREE", "MPHASIS"
    ],
    "Nifty Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON", "LUPIN", 
        "CADILAHC", "TORNTPHARM", "GLENMARK", "AUROPHARMA"
    ],
    "Nifty Auto": [
        "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", 
        "TVSMOTOR", "ASHOKLEY", "M&M", "ESCORTS"
    ],
    "Nifty Energy": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "HPCL", "COALINDIA", "TATAPOWER", 
        "ADANIGREEN", "RPOWER", "SUZLON"
    ]
}

# Technical indicator parameters
INDICATOR_PARAMS = {
    "EMA": {
        "short": 10,
        "medium": 21,
        "long": 50
    },
    "SMA": {
        "short": 20,
        "long": 50
    },
    "RSI": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    "MACD": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    "BB": {
        "period": 20,
        "std": 2
    },
    "STOCH": {
        "k_period": 14,
        "d_period": 3,
        "overbought": 80,
        "oversold": 20
    },
    "ADR": {
        "period": 14
    }
}

# Trading signal strength levels
SIGNAL_STRENGTH = {
    "Very Strong": 5,
    "Strong": 4,
    "Moderate": 3,
    "Weak": 2,
    "Very Weak": 1
}

# Risk management parameters
RISK_PARAMS = {
    "stop_loss_pct": 2.0,  # Default 2% stop loss
    "take_profit_pct": 6.0,  # Default 6% take profit (3:1 RR)
    "position_size_pct": 2.0,  # Default 2% position size
    "max_positions": 5  # Maximum concurrent positions
}

# Pattern success rates (historical data)
PATTERN_SUCCESS_RATES = {
    "Engulfing Pattern": 80,
    "Hammer": 75,
    "Doji": 60,
    "Morning Star": 85,
    "Evening Star": 85,
    "Piercing Pattern": 70,
    "Dark Cloud Cover": 70,
    "Shooting Star": 72,
    "Inverted Hammer": 68,
    "Dragonfly Doji": 73,
    "Gravestone Doji": 73,
    "Three White Soldiers": 78,
    "Three Black Crows": 78,
    "Ascending Triangle": 65,
    "Descending Triangle": 65,
    "Symmetrical Triangle": 60,
    "Rising Wedge": 70,
    "Falling Wedge": 70,
    "Bull Flag": 75,
    "Bear Flag": 75,
    "Head and Shoulders": 85,
    "Inverse Head and Shoulders": 85,
    "Double Top": 78,
    "Double Bottom": 78,
    "Cup and Handle": 80,
    "Resistance Breakout": 75,
    "Support Breakdown": 75
}