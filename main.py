import time
import hmac
import hashlib
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('DELTA_API_KEY')
api_secret = os.getenv('DELTA_API_SECRET')
base_url = 'https://api.india.delta.exchange'
symbol = 'BTCUSD'
resolution = '1h'

def get_signature(method, timestamp, path, query_string, payload):
    # The signature must be: METHOD + TIMESTAMP + PATH + ?QUERY + PAYLOAD
    if query_string:
        full_path = f"{path}?{query_string}"
    else:
        full_path = path
        
    signature_data = method + timestamp + full_path + payload
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        signature_data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def fetch_data():
    path = '/v2/history/candles'
    method = 'GET'
    
    # 1. Define params
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=7)).timestamp())
    
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'start': start_time,
        'end': end_time
    }
    
    # 2. Sort params alphabetically (Crucial for Signature)
    # This creates a string like: "end=...&resolution=...&start=...&symbol=..."
    sorted_params = sorted(params.items())
    query_string = urlencode(sorted_params)
    
    # 3. Generate Timestamp and Signature
    timestamp = str(int(time.time()))
    signature = get_signature(method, timestamp, path, query_string, '')
    
    headers = {
        'api-key': api_key,
        'timestamp': timestamp,
        'signature': signature,
        'User-Agent': 'python-client',
        'Content-Type': 'application/json'
    }
    
    # 4. Make Request using the EXACT same query string
    # We append query_string manually to ensure 'requests' doesn't re-order it
    url = f"{base_url}{path}?{query_string}"
    
    print(f"Requesting: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# --- MAIN LOGIC ---
data = fetch_data()

if data and data.get('success'):
    candles = data['result']
    if not candles:
        print("No data returned for this range.")
        exit()

    df = pd.DataFrame(candles)
    
    # Clean up data
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Fetched {len(df)} candles.")
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Plot: Price
    ax1.set_title(f'{symbol} Price vs OI ({resolution})')
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # Top Plot: OI (on right axis)
    # Check for 'oi', 'open_interest', or 'close_oi'
    oi_col = None
    for col in ['oi', 'open_interest', 'close_oi']:
        if col in df.columns:
            oi_col = col
            break
            
    if oi_col:
        ax1_oi = ax1.twinx()
        ax1_oi.plot(df.index, df[oi_col], label='Open Interest', color='orange', linestyle='--')
        ax1_oi.set_ylabel('Open Interest', color='orange')
        ax1_oi.tick_params(axis='y', labelcolor='orange')
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_oi.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        print("Warning: No Open Interest column found in data.")

    # Bottom Plot: Volume
    ax2.bar(df.index, df['volume'], color='green', alpha=0.5, width=0.03)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date (UTC)')
    
    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

else:
    print("Failed to fetch data.")