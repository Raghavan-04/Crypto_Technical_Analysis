import os
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime
from urllib.parse import urlencode
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('DELTA_API_KEY')
API_SECRET = os.getenv('DELTA_API_SECRET')
BASE_URL = 'https://api.india.delta.exchange'
UNDERLYING_ASSET = 'BTC'
JSON_FILENAME = 'market_data.json'
UPDATE_INTERVAL = 1

# --- AUTHENTICATION HELPER ---
def generate_headers(method, path, params=None, payload=''):
    timestamp = str(int(time.time()))
    query_string = ""
    if params:
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
    
    full_path = f"{path}?{query_string}" if query_string else path
    signature_data = method + timestamp + full_path + payload
    
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        signature_data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return {
        'api-key': API_KEY,
        'timestamp': timestamp,
        'signature': signature,
        'User-Agent': 'python-client',
        'Content-Type': 'application/json'
    }

# --- RATE LIMITER ---
class DeltaRateLimiter:
    def __init__(self):
        self.tokens = 2000
        self.last_update = time.time()
        self.fill_rate = 2000 / 60

    def wait(self, weight):
        while True:
            now = time.time()
            self.tokens = min(2000, self.tokens + (now - self.last_update) * self.fill_rate)
            self.last_update = now
            if self.tokens >= weight:
                self.tokens -= weight
                return
            time.sleep(0.1)

limiter = DeltaRateLimiter()

# --- REQUEST WRAPPER ---
def make_authenticated_request(endpoint, params=None):
    limiter.wait(3)
    headers = generate_headers('GET', endpoint, params)
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('result', [])
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return []

# --- CALCULATION HELPERS ---
def calculate_max_pain(df):
    if df.empty: return 0
    strikes = sorted(df['strike'].unique())
    calls = df[df['type'] == 'Call']
    puts = df[df['type'] == 'Put']
    
    pain_map = {}
    for price in strikes:
        call_loss = calls.apply(lambda r: max(0, price - r['strike']) * r['oi'], axis=1).sum()
        put_loss = puts.apply(lambda r: max(0, r['strike'] - price) * r['oi'], axis=1).sum()
        pain_map[price] = call_loss + put_loss

    return min(pain_map, key=pain_map.get) if pain_map else 0

def process_and_save():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Fetching data...", end='\r')

    # 1. Fetch ALL Tickers
    tickers = make_authenticated_request('/v2/tickers')
    if not tickers:
        print("❌ No data received from API.")
        return

    # 2. SMART SEARCH: Find Spot Price and Volume
    spot_price = 0.0
    spot_volume = 0.0
    
    # Strategy A: Find the Index Price (This is the TRUE spot price for options)
    for item in tickers:
        if item['symbol'] == '.DEXBTUSD': # Standard BTC Index on Delta
            spot_price = float(item.get('close') or item.get('mark_price') or 0)
            break
            
    # Strategy B: Find the Perpetual Future (BTCUSDT) for Volume (and price backup)
    for item in tickers:
        if item['symbol'] == 'BTCUSDT': # Linear Perpetual
            spot_volume = float(item.get('volume') or 0)
            # Use this price if Index wasn't found
            if spot_price == 0:
                spot_price = float(item.get('mark_price') or item.get('close') or 0)
            break
        elif item['symbol'] == 'BTCUSD' and spot_volume == 0: # Inverse Perpetual (Backup)
            spot_volume = float(item.get('volume_usd') or item.get('volume') or 0)
            if spot_price == 0:
                spot_price = float(item.get('mark_price') or item.get('close') or 0)

    # Strategy C: Ultimate Fallback (Look inside Option Tickers)
    if spot_price == 0:
        for item in tickers:
            if item['symbol'].startswith(f"C-{UNDERLYING_ASSET}-"):
                # Many option tickers contain the underlying spot price
                spot_price = float(item.get('spot_price') or item.get('underlying_price') or 0)
                if spot_price > 0: break

    # 3. Process Options
    parsed_options = []
    for item in tickers:
        # Filter for BTC Options
        if item['symbol'].startswith(f"C-{UNDERLYING_ASSET}-") or item['symbol'].startswith(f"P-{UNDERLYING_ASSET}-"):
            parts = item['symbol'].split('-')
            if len(parts) >= 4:
                greeks = item.get('greeks') or {}
                parsed_options.append({
                    'symbol': item['symbol'],
                    'type': 'Call' if parts[0] == 'C' else 'Put',
                    'strike': float(parts[2]),
                    'expiry': parts[3],
                    'mark_price': float(item.get('mark_price') or 0),
                    'oi': float(item.get('oi') or 0),
                    'volume': float(item.get('volume') or 0),
                    'iv': float(greeks.get('implied_volatility') or 0),
                    'delta': float(greeks.get('delta') or 0),
                    'theta': float(greeks.get('theta') or 0),
                    'gamma': float(greeks.get('gamma') or 0),
                    'vega': float(greeks.get('vega') or 0)
                })

    if not parsed_options:
        print("❌ No options found.")
        return

    # 4. DataFrame Processing
    df = pd.DataFrame(parsed_options)
    most_active_expiry = df['expiry'].mode()[0]
    df_active = df[df['expiry'] == most_active_expiry].copy()

    # 5. Calculate Max Pain
    max_pain = calculate_max_pain(df_active)
    
    # 6. Filter 20 Strikes around MAX PAIN
    # If spot price is found, center around Spot. If not, center around Max Pain.
    center_point = spot_price if spot_price > 0 else max_pain
    
    df_active['diff'] = abs(df_active['strike'] - center_point)
    target_strikes = df_active.sort_values('diff')['strike'].unique()[:20]
    final_df = df_active[df_active['strike'].isin(target_strikes)].sort_values('strike')
    final_data_list = final_df.drop(columns=['diff']).to_dict(orient='records')

    # 7. Save JSON
    json_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "underlying": UNDERLYING_ASSET,
            "spot_price": spot_price,
            "spot_volume": spot_volume,
            "selected_expiry": most_active_expiry,
            "max_pain": max_pain
        },
        "options_chain": final_data_list
    }

    try:
        with open(JSON_FILENAME, 'w') as f:
            json.dump(json_output, f, indent=4)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved {JSON_FILENAME} | Spot: {spot_price:.2f} | MP: {max_pain} | Vol: {spot_volume:,.0f}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("❌ Error: API Keys not found in .env file.")
        exit()
        
    print(f"🚀 Data Collector Running... (Advanced Spot Search Enabled)")
    
    while True:
        try:
            process_and_save()
            time.sleep(UPDATE_INTERVAL)
        except KeyboardInterrupt:
            print("\n👋 Stopped.")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            time.sleep(5)