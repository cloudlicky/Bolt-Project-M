"""Configuration file for the Swing Trading Analysis App"""

# Indian Stock symbols with company names
INDIAN_STOCKS = {
    # Nifty 50 - Large Cap
    "RELIANCE.NS": "Reliance Industries Limited",
    "TCS.NS": "Tata Consultancy Services Limited",
    "HDFCBANK.NS": "HDFC Bank Limited",
    "INFY.NS": "Infosys Limited",
    "HDFC.NS": "Housing Development Finance Corporation Limited",
    "ICICIBANK.NS": "ICICI Bank Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Limited",
    "HINDUNILVR.NS": "Hindustan Unilever Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Limited",
    "ITC.NS": "ITC Limited",
    "AXISBANK.NS": "Axis Bank Limited",
    "LT.NS": "Larsen & Toubro Limited",
    "ASIANPAINT.NS": "Asian Paints Limited",
    "MARUTI.NS": "Maruti Suzuki India Limited",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
    "TITAN.NS": "Titan Company Limited",
    "ULTRACEMCO.NS": "UltraTech Cement Limited",
    "ONGC.NS": "Oil and Natural Gas Corporation Limited",
    "NESTLEIND.NS": "Nestle India Limited",
    "POWERGRID.NS": "Power Grid Corporation of India Limited",
    "NTPC.NS": "NTPC Limited",
    "TECHM.NS": "Tech Mahindra Limited",
    "HCLTECH.NS": "HCL Technologies Limited",
    "WIPRO.NS": "Wipro Limited",
    
    # Bank Nifty
    "INDUSINDBK.NS": "IndusInd Bank Limited",
    "BANDHANBNK.NS": "Bandhan Bank Limited",
    "FEDERALBNK.NS": "Federal Bank Limited",
    "IDFCFIRSTB.NS": "IDFC First Bank Limited",
    "PNB.NS": "Punjab National Bank",
    
    # Mid Cap Stocks
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Limited",
    "ADANIGREEN.NS": "Adani Green Energy Limited",
    "JINDALSTEL.NS": "Jindal Steel & Power Limited",
    "TATAPOWER.NS": "Tata Power Company Limited",
    "SAIL.NS": "Steel Authority of India Limited",
    "COALINDIA.NS": "Coal India Limited",
    "IOC.NS": "Indian Oil Corporation Limited",
    "BPCL.NS": "Bharat Petroleum Corporation Limited",
    "HPCL.NS": "Hindustan Petroleum Corporation Limited",
    "GODREJCP.NS": "Godrej Consumer Products Limited",
    
    # Small Cap Stocks
    "ZEEL.NS": "Zee Entertainment Enterprises Limited",
    "YESBANK.NS": "Yes Bank Limited",
    "RPOWER.NS": "Reliance Power Limited",
    "SUZLON.NS": "Suzlon Energy Limited",
    "JPASSOCIAT.NS": "Jaiprakash Associates Limited",
    "RCOM.NS": "Reliance Communications Limited",
    "DHFL.NS": "Dewan Housing Finance Corporation Limited",
    "JETAIRWAYS.NS": "Jet Airways (India) Limited",
    "PCJEWELLER.NS": "PC Jeweller Limited",
    "YESBANK.NS": "Yes Bank Limited",
    
    # Additional Popular Stocks
    "TATAMOTORS.NS": "Tata Motors Limited",
    "BAJFINANCE.NS": "Bajaj Finance Limited",
    "BAJAJFINSV.NS": "Bajaj Finserv Limited",
    "HDFCLIFE.NS": "HDFC Life Insurance Company Limited",
    "SBILIFE.NS": "SBI Life Insurance Company Limited",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance Company Limited",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Limited",
    "CIPLA.NS": "Cipla Limited",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Limited",
    "DIVISLAB.NS": "Divi's Laboratories Limited"
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
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HDFC.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS",
        "ITC.NS", "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS"
    ],
    "Mid Cap": [
        "ADANIPORTS.NS", "ADANIGREEN.NS", "JINDALSTEL.NS", "TATAPOWER.NS",
        "SAIL.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS", "HPCL.NS", "GODREJCP.NS",
        "TATAMOTORS.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"
    ],
    "Small Cap": [
        "ZEEL.NS", "YESBANK.NS", "RPOWER.NS", "SUZLON.NS", "JPASSOCIAT.NS",
        "RCOM.NS", "DHFL.NS", "JETAIRWAYS.NS", "PCJEWELLER.NS"
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
    }
}

# Candlestick patterns to detect
CANDLESTICK_PATTERNS = [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE",
    "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS", "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI",
    "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS",
    "CDLINNECK", "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
    "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS"
]

# Chart pattern templates
CHART_PATTERNS = {
    "Triangle": {
        "ascending": "Ascending Triangle",
        "descending": "Descending Triangle", 
        "symmetrical": "Symmetrical Triangle"
    },
    "Wedge": {
        "rising": "Rising Wedge",
        "falling": "Falling Wedge"
    },
    "Flag": {
        "bull": "Bull Flag",
        "bear": "Bear Flag"
    },
    "Head_Shoulders": {
        "normal": "Head and Shoulders",
        "inverse": "Inverse Head and Shoulders"
    },
    "Double": {
        "top": "Double Top",
        "bottom": "Double Bottom"
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