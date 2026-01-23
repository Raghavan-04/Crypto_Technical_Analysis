import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime
import os
import sys

# --- NEW IMPORTS & CONFIGURATION ---
# Try to import winsound for Windows, otherwise use os.system for Mac/Linux
try:
    import winsound
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False

# Configuration
JSON_FILE = 'market_data.json'
UPDATE_INTERVAL = 2000  # milliseconds (2 seconds)
MAX_HISTORY = 50        # Number of historical points to keep
EXPORT_ENABLED = True   # Enable/disable data export
ALERTS_ENABLED = True   # Enable/disable sound alerts
ALERT_LOG_MAX = 20      # Maximum number of alerts to store

# Global variables to store historical data and state
history = {
    'timestamps': [],
    'spot_prices': [],
    'pcr_oi': [],
    'pcr_volume': [],
    'net_delta': [],
    'total_gamma': [],
    'max_pain': [],
    'call_oi': [],
    'put_oi': []
}
alert_log = []
previous_analysis = None # Stores the analysis from the previous successful frame

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- HELPER FUNCTIONS (omitted for brevity, no changes here) ---

def play_alert(frequency=1000, duration=200):
    """Plays a system sound alert."""
    if not ALERTS_ENABLED:
        return
        
    if IS_WINDOWS:
        import winsound
        winsound.Beep(frequency, duration)
    else:
        # Fallback for Linux/Mac - a simple print for non-critical alerts
        print("(ALERT)", end='', flush=True)

def calculate_price_momentum(current_price, history_prices, periods=5):
    """Calculates price momentum over the last N periods."""
    if len(history_prices) < periods:
        return 0.0

    start_price = history_prices[-periods]
    if start_price == 0:
        return 0.0
        
    momentum = ((current_price - start_price) / start_price) * 100
    return momentum

def calculate_risk_level(analysis):
    """Calculates a market risk score (0-7 points)."""
    risk_score = 0
    risk_summary = []
    
    # 1. Extreme PCR (Max 2 points)
    if analysis['pcr_oi'] < 0.5 or analysis['pcr_oi'] > 1.5:
        risk_score += 2
        risk_summary.append("Extreme PCR")
    
    # 2. Far from Max Pain (Max 2 points)
    if abs(analysis['pain_pull_pct']) > 5:
        risk_score += 2
        risk_summary.append("Far from Max Pain")
    
    # 3. High Gamma near Spot (Max 3 points)
    max_gamma_strike = analysis['strike_analysis'].loc[analysis['strike_analysis']['total_gamma'].idxmax()]
    gamma_distance = abs(max_gamma_strike['strike'] - analysis['spot_price']) / analysis['spot_price'] * 100
    
    if gamma_distance < 1.0: # Within 1% of spot
        risk_score += 3
        risk_summary.append(f"High Gamma Zone near ${max_gamma_strike['strike']:,.0f}")

    # Risk Level categorization
    if risk_score <= 2:
        level = "LOW"
        color = "green"
    elif risk_score <= 5:
        level = "MEDIUM"
        color = "yellow"
    else:
        level = "HIGH"
        color = "red"
        
    return risk_score, level, color, risk_summary

def check_alerts(analysis, previous_analysis):
    """Monitors conditions and triggers alerts."""
    if not ALERTS_ENABLED or previous_analysis is None:
        return

    global alert_log
    current_time = datetime.now().strftime('%H:%M:%S')
    
    def log_alert(message, urgency="CRITICAL"):
        alert_tuple = (message, urgency)
        
        # Simple anti-spam: check if the same alert type was logged recently
        if alert_tuple not in [(m, u) for t, m, u in alert_log[-3:]]:
            alert_log.append((current_time, message, urgency))
            if len(alert_log) > ALERT_LOG_MAX:
                alert_log.pop(0)
            play_alert() # Play sound on new, non-spam alert

    # 1. PCR Shift Alert
    pcr_change = abs(analysis['pcr_oi'] - previous_analysis['pcr_oi'])
    if pcr_change > 0.15:
        log_alert(f"Significant PCR shift: {previous_analysis['pcr_oi']:.2f} -> {analysis['pcr_oi']:.2f}", "MAJOR")

    # 2. Max Pain Proximity
    if abs(analysis['pain_pull_pct']) < 1.5 and abs(previous_analysis['pain_pull_pct']) >= 1.5:
        log_alert(f"Price entering Max Pain Proximity (1.5% zone) at ${analysis['spot_price']:,.0f}", "MAJOR")

    # 3. High Gamma Spike Alert
    max_gamma_strike_curr = analysis['strike_analysis'].loc[analysis['strike_analysis']['total_gamma'].idxmax()]
    gamma_distance_curr = abs(max_gamma_strike_curr['strike'] - analysis['spot_price']) / analysis['spot_price'] * 100
    
    if gamma_distance_curr < 1.0 and analysis['total_gamma'] > previous_analysis['total_gamma'] * 1.5:
        log_alert(f"HIGH GAMMA SPIKE: Volatility increase expected near ${max_gamma_strike_curr['strike']:,.0f}", "CRITICAL")
    
    # 4. Net Delta Shift (large directional flow change)
    delta_change = analysis['net_delta'] - previous_analysis['net_delta']
    if abs(delta_change) > 5000:
        flow = "Bullish" if delta_change > 0 else "Bearish"
        log_alert(f"Massive Net Delta Shift ({flow}): {delta_change:+.0f} change", "MAJOR")

def calculate_oi_changes(current_strike_analysis, previous_strike_analysis):
    """Compares current and previous OI to find buildup/reduction."""
    if previous_strike_analysis is None:
        return pd.DataFrame()

    # Merge current and previous OI by strike
    prev_oi = previous_strike_analysis[['strike', 'total_oi']].set_index('strike')
    curr_oi = current_strike_analysis[['strike', 'total_oi']].set_index('strike')
    
    merged = curr_oi.join(prev_oi, how='outer', lsuffix='_curr', rsuffix='_prev').fillna(0)
    merged['oi_change'] = merged['total_oi_curr'] - merged['total_oi_prev']
    
    # Filter for significant changes (e.g., abs change > 500 contracts)
    oi_flow = merged[abs(merged['oi_change']) > 500].reset_index()
    oi_flow['flow'] = oi_flow['oi_change'].apply(lambda x: 'Buildup' if x > 0 else 'Reduction')
    
    # Return top 5 Buildups and top 5 Reductions
    top_buildup = oi_flow[oi_flow['flow'] == 'Buildup'].nlargest(5, 'oi_change')
    top_reduction = oi_flow[oi_flow['flow'] == 'Reduction'].nsmallest(5, 'oi_change')
    
    return pd.concat([top_buildup, top_reduction], ignore_index=True).drop_duplicates()

def export_data(analysis):
    """Exports current analysis to CSV files."""
    if not EXPORT_ENABLED:
        print("Export is disabled in configuration.")
        return

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Strike Analysis
    analysis['strike_analysis'].to_csv(f'strike_analysis_{timestamp_str}.csv', index=False)
    
    # 2. Market Summary
    summary_data = {
        'Timestamp': [datetime.now()],
        'Underlying': [analysis['metadata']['underlying']],
        'SpotPrice': [analysis['spot_price']],
        'MaxPain': [analysis['max_pain']],
        'PCR_OI': [analysis['pcr_oi']],
        'NetDelta': [analysis['net_delta']],
        'TotalGamma': [analysis['total_gamma']],
        'Momentum_5P': [analysis.get('momentum_5p', 0)],
        'Risk_Level': [analysis.get('risk_level', 'N/A')],
        'Alerts_Count': [len(alert_log)]
    }
    pd.DataFrame(summary_data).to_csv(f'market_summary_{timestamp_str}.csv', index=False)
    
    print(f"\n[EXPORT] Data exported successfully to CSV files with timestamp {timestamp_str}\n")

# --- CORE FUNCTIONS (omitted for brevity, no changes here) ---

def load_and_analyze():
    """Load JSON, perform analysis, and calculate new metrics"""
    global previous_analysis
    
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        options = data['options_chain']
        
        # Convert to DataFrame and calculate basic metrics
        df = pd.DataFrame(options)
        calls = df[df['type'] == 'Call'].copy()
        puts = df[df['type'] == 'Put'].copy()
        
        spot_price = metadata['spot_price']
        max_pain = metadata['max_pain']
        timestamp = metadata['timestamp']
        
        total_put_oi = puts['oi'].sum()
        total_call_oi = calls['oi'].sum()
        total_put_volume = puts['volume'].sum()
        total_call_volume = calls['volume'].sum()
        
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        calls['weighted_delta'] = calls['delta'] * calls['oi']
        puts['weighted_delta'] = puts['delta'] * puts['oi']
        net_delta = calls['weighted_delta'].sum() + puts['weighted_delta'].sum()
        
        df['weighted_gamma'] = df['gamma'] * df['oi']
        total_gamma = df['weighted_gamma'].sum()
        
        df['weighted_vega'] = df['vega'] * df['oi']
        total_vega = df['weighted_vega'].sum()
        
        df['weighted_theta'] = df['theta'] * df['oi']
        total_theta = df['weighted_theta'].sum()
        
        # Strike analysis
        strike_analysis = pd.DataFrame()
        strikes = sorted(df['strike'].unique())
        
        for strike in strikes:
            strike_data = df[df['strike'] == strike]
            call_data = strike_data[strike_data['type'] == 'Call']
            put_data = strike_data[strike_data['type'] == 'Put']
            
            strike_analysis = pd.concat([strike_analysis, pd.DataFrame([{
                'strike': strike,
                'call_oi': call_data['oi'].sum(),
                'put_oi': put_data['oi'].sum(),
                'call_volume': call_data['volume'].sum(),
                'put_volume': put_data['volume'].sum(),
                'total_oi': call_data['oi'].sum() + put_data['oi'].sum(),
                'net_oi': call_data['oi'].sum() - put_data['oi'].sum(),
                'total_gamma': (strike_data['gamma'] * strike_data['oi']).sum(),
                'net_delta': (strike_data['delta'] * strike_data['oi']).sum(),
                'pcr_strike': put_data['oi'].sum() / call_data['oi'].sum() if call_data['oi'].sum() > 0 else 0
            }])], ignore_index=True)
        
        # Identify key levels
        strike_analysis['distance_pct'] = abs(strike_analysis['strike'] - spot_price) / spot_price * 100
        atm_strikes = strike_analysis[strike_analysis['distance_pct'] < 5].copy()
        
        support_levels = strike_analysis[strike_analysis['strike'] < spot_price].nlargest(3, 'total_oi')
        resistance_levels = strike_analysis[strike_analysis['strike'] > spot_price].nlargest(3, 'total_oi')
        
        df['turnover'] = df.apply(lambda x: x['volume'] / x['oi'] if x['oi'] > 0 else 0, axis=1)
        df['premium'] = df['mark_price'] * df['volume']
        unusual = df[df['turnover'] > 2.0].copy().nlargest(5, 'premium')
        
        dealer_gamma = total_gamma
        
        pain_distance = max_pain - spot_price
        pain_pull_pct = (pain_distance / spot_price) * 100
        
        # --- NEW METRIC CALCULATIONS ---
        momentum_5p = calculate_price_momentum(spot_price, history['spot_prices'])
        oi_flow_changes = calculate_oi_changes(strike_analysis, previous_analysis['strike_analysis'] if previous_analysis else None)
        
        # Update history
        history['timestamps'].append(datetime.now())
        history['spot_prices'].append(spot_price)
        history['pcr_oi'].append(pcr_oi)
        history['pcr_volume'].append(pcr_volume)
        history['net_delta'].append(net_delta)
        history['total_gamma'].append(total_gamma)
        history['max_pain'].append(max_pain)
        history['call_oi'].append(total_call_oi)
        history['put_oi'].append(total_put_oi)
        
        # Keep only last MAX_HISTORY data points
        for key in history:
            if len(history[key]) > MAX_HISTORY:
                history[key] = history[key][-MAX_HISTORY:]
        
        current_analysis = {
            'metadata': metadata,
            'spot_price': spot_price,
            'max_pain': max_pain,
            'pain_pull_pct': pain_pull_pct,
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_volume,
            'net_delta': net_delta,
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'total_theta': total_theta,
            'total_put_oi': total_put_oi,
            'total_call_oi': total_call_oi,
            'total_put_volume': total_put_volume,
            'total_call_volume': total_call_volume,
            'strike_analysis': strike_analysis,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'unusual': unusual,
            'atm_strikes': atm_strikes,
            'dealer_gamma': dealer_gamma,
            'timestamp': timestamp,
            # NEW METRICS
            'momentum_5p': momentum_5p,
            'oi_flow_changes': oi_flow_changes
        }
        
        # Check alerts based on current and previous state
        check_alerts(current_analysis, previous_analysis)
        
        return current_analysis
        
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found!")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

def print_terminal_output(analysis):
    """Print simplified actionable analysis to terminal"""
    clear_screen()
    
    # ANSI escape codes for colors
    C_RED = "\033[91m"; C_GREEN = "\033[92m"; C_YELLOW = "\033[93m"; C_CYAN = "\033[96m"; C_END = "\033[0m"
    C_BOLD = "\033[1m"
    
    print("="*80)
    print(f"{C_CYAN}{C_BOLD}[FOCUS] TRADING DASHBOARD - {analysis['metadata']['underlying']}{C_END}")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Spot: ${analysis['spot_price']:,.2f}")
    
    # --- ACTIVE ALERTS ---
    print("\n[ALERT] ACTIVE ALERTS")
    print("-" * 80)
    if alert_log:
        for time, msg, urgency in alert_log[-3:]: # Show last 3
            color = C_RED if urgency == "CRITICAL" else C_YELLOW
            print(f"  {color}[{time} | {urgency}]{C_END} {msg}")
    else:
        print(f"  {C_GREEN}(OK) No recent alerts.{C_END}")
    print("-" * 80)
    
    # --- MARKET CONDITIONS & GREEKS OVERVIEW ---
    risk_score, risk_level, risk_color, risk_summary = calculate_risk_level(analysis)
    
    # Momentum Display
    momentum_arrow = "^" if analysis['momentum_5p'] > 0 else "v"
    momentum_color = C_GREEN if analysis['momentum_5p'] > 0 else C_RED
    
    print(f"\n{C_BOLD}MARKET CONDITIONS OVERVIEW{C_END}")
    print("-" * 40)
    print(f"  Risk Level: {C_BOLD}[{risk_level} ({risk_score}/7)]{C_END} - {risk_summary[0] if risk_summary else 'Stable'}")
    print(f"  Price Momentum (5P): {momentum_color}{momentum_arrow} {analysis['momentum_5p']:+.2f}%\033[0m")
    
    print(f"\n{C_BOLD}GREEKS OVERVIEW{C_END}")
    print("-" * 40)
    print(f"  Net Delta: {analysis['net_delta']:,.0f} (Directional Bias)")
    print(f"  Total Gamma: {analysis['total_gamma']:,.0f} (Volatility Sensitivity)")
    print(f"  Total Vega: {analysis['total_vega']:,.0f} (IV Exposure)")
    print(f"  Total Theta: {analysis['total_theta']:,.0f} (Time Decay)")
    print("-" * 40)
    
    # Market Direction Signal
    print(f"\n{C_BOLD}MARKET BIAS SIGNALS{C_END}")
    print("-" * 40)
    
    bullish = 0
    bearish = 0
    signals = []
    
    if analysis['pcr_oi'] < 0.7:
        bullish += 1
        signals.append("(BULL) PCR < 0.7 (Call Heavy)")
    elif analysis['pcr_oi'] > 1.3:
        bearish += 1
        signals.append("(BEAR) PCR > 1.3 (Put Heavy)")
    else:
        signals.append("(NEUT) Balanced Put/Call ratio")
    
    if analysis['net_delta'] > 0:
        bullish += 1
        signals.append("(BULL) Net Positive Delta")
    else:
        bearish += 1
        signals.append("(BEAR) Net Negative Delta")
    
    if analysis['pain_pull_pct'] > 2:
        signals.append(f"^ Price {analysis['pain_pull_pct']:.1f}% below Max Pain (Upward Pull)")
    elif analysis['pain_pull_pct'] < -2:
        signals.append(f"v Price {abs(analysis['pain_pull_pct']):.1f}% above Max Pain (Downward Pull)")
    
    for signal in signals:
        print(f"  {signal}")
    
    print("\n" + "="*40)
    if bullish > bearish:
        print(f"{C_GREEN}{C_BOLD}(BULL) OVERALL: BULLISH BIAS{C_END}")
    elif bearish > bullish:
        print(f"{C_RED}{C_BOLD}(BEAR) OVERALL: BEARISH BIAS{C_END}")
    else:
        print(f"{C_YELLOW}{C_BOLD}(NEUT) OVERALL: NEUTRAL{C_END}")
    print("="*40)

    # --- OI FLOW CHANGES ---
    print(f"\n{C_BOLD}OI FLOW CHANGES (Smart Money Tracking){C_END}")
    print("-" * 80)
    oi_flow = analysis['oi_flow_changes']
    if len(oi_flow) > 0:
        for idx, row in oi_flow.head(5).iterrows():
            flow_color = C_GREEN if row['flow'] == 'Buildup' else C_RED
            print(f"  {flow_color}{row['flow']:9}{C_END} | Strike: ${row['strike']:,.0f} | Change: {row['oi_change']:+,.0f} OI")
    else:
        print(f"  {C_CYAN}(NEUT) No significant OI flow changes detected.{C_END}")
    print("-" * 80)
    
    # Key Levels
    print(f"\n{C_BOLD}KEY PRICE LEVELS{C_END}")
    print("-" * 40)
    print(f"Max Pain: ${analysis['max_pain']:,.0f} (Price magnet)")
    print("\nResistance (Above Spot):")
    for idx, row in analysis['resistance_levels'].iterrows():
        dist = ((row['strike'] - analysis['spot_price']) / analysis['spot_price']) * 100
        print(f"  ${row['strike']:,.0f} (+{dist:.1f}%) - OI: {row['total_oi']:,.0f}")
    
    print("\nSupport (Below Spot):")
    for idx, row in analysis['support_levels'].iterrows():
        dist = ((analysis['spot_price'] - row['strike']) / analysis['spot_price']) * 100
        print(f"  ${row['strike']:,.0f} (-{dist:.1f}%) - OI: {row['total_oi']:,.0f}")
    
    # Trading Zones
    print(f"\n{C_BOLD}GAMMA EXPOSURE{C_END}")
    print("-" * 40)
    max_gamma_strike = analysis['strike_analysis'].loc[analysis['strike_analysis']['total_gamma'].idxmax()]
    print(f"Max Gamma Strike: ${max_gamma_strike['strike']:,.0f}")
    if abs(max_gamma_strike['strike'] - analysis['spot_price']) / analysis['spot_price'] < 0.02:
        print(f"{C_RED}(WARN) High gamma near spot = choppy price action expected{C_END}")
    else:
        print(f"{C_GREEN}(OK) Low gamma zone = smoother directional moves possible{C_END}")
    
    # --- INTERACTIVE CONTROLS ---
    print("\n\n" + "="*80)
    print(f"{C_CYAN}{C_BOLD}CONTROLS:  [E]xport Data to CSV  |  [R]eset History  |  [Q]uit{C_END}")
    print("="*80)


# --- KEYBOARD CONTROL HANDLER ---
def on_key_press(event):
    """Handles keyboard shortcuts for export, quit, and reset."""
    global previous_analysis, history, alert_log
    
    if event.key.lower() == 'e':
        if previous_analysis:
            export_data(previous_analysis)
        else:
            print("Cannot export: No analysis data available yet.")
    elif event.key.lower() == 'q':
        print("\n\n[QUIT] Monitoring stopped by user via keyboard.")
        plt.close(fig)
        sys.exit(0)
    elif event.key.lower() == 'r':
        # Reset history
        history.update({key: [] for key in history})
        previous_analysis = None
        alert_log.clear()
        print("\n\n[RESET] History and state reset.")

# Initialize matplotlib figure with dark theme
plt.style.use('dark_background')

# --- OPTIMIZED GRIDSPEC FOR SMALLER SCREENS ---
# Reduced figsize and adjusted hspace/wspace
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4) 

# Create subplots
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])

# Connect keyboard events
fig.canvas.mpl_connect('key_press_event', on_key_press)

def update_plot(frame):
    """Update all plots"""
    global previous_analysis
    
    analysis = load_and_analyze()
    
    if analysis is None:
        return
    
    # Print to terminal
    print_terminal_output(analysis)
    
    # --- IMPORTANT: Store current analysis for the next frame's comparison ---
    previous_analysis = analysis 
    
    # Clear all axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.clear()
    
    strike_analysis = analysis['strike_analysis']
    spot_price = analysis['spot_price']
    max_pain = analysis['max_pain']
    
    # Calculate Risk Level (used in multiple plots)
    risk_score, risk_level, risk_color, _ = calculate_risk_level(analysis)

    # ========================================================================
    # 1. PRICE CHART WITH SUPPORT/RESISTANCE (Font Size Reduced)
    # ========================================================================
    if len(history['timestamps']) > 1:
        ax1.plot(history['timestamps'], history['spot_prices'], 
                color='cyan', linewidth=1.5, marker='.', markersize=3, label='Spot Price') # Reduced linewidth/markersize
        
        # Max Pain line
        ax1.plot(history['timestamps'], history['max_pain'], 
                color='magenta', linewidth=1.5, linestyle='--', alpha=0.8, label='Max Pain')
        
        # Support levels (green dashed) - Simplified annotation
        for idx, row in analysis['support_levels'].head(2).iterrows():
            ax1.axhline(y=row['strike'], color='green', linestyle=':', linewidth=1.2, alpha=1)
            ax1.text(history['timestamps'][-1], row['strike'], f" S: ${row['strike']:,.0f}", 
                    fontsize=9, color='green', va='bottom', ha='right') # Fontsize 7
        
        # Resistance levels (red dashed) - Simplified annotation
        for idx, row in analysis['resistance_levels'].head(2).iterrows():
            ax1.axhline(y=row['strike'], color='red', linestyle=':', linewidth=1.2, alpha=1)
            ax1.text(history['timestamps'][-1], row['strike'], f" R: ${row['strike']:,.0f}", 
                    fontsize=9, color='red', va='bottom', ha='right') # Fontsize 7
        
        ax1.fill_between(history['timestamps'], history['spot_prices'], history['max_pain'], 
                        alpha=0.2, color='yellow')
        
        # Momentum Box Annotation (Simplified)
        momentum_color_plot = 'green' if analysis['momentum_5p'] > 0 else 'red'
        momentum_text = f"5P Momentum: {analysis['momentum_5p']:+.2f}%"
        
        ax1.text(0.01, 0.98, momentum_text, transform=ax1.transAxes, fontsize=8, va='top', ha='left', # Fontsize 8
                bbox=dict(boxstyle='round', facecolor='black', alpha=1, edgecolor=momentum_color_plot, linewidth=1))
        
        ax1.set_xlabel('Time', fontsize=7, fontweight='bold') # Fontsize 7
        ax1.set_ylabel('Price ($)', fontsize=7, fontweight='bold') # Fontsize 7
        ax1.set_title('PRICE ACTION with Key Levels', fontsize=8, fontweight='bold', pad=5) # Fontsize 8
        ax1.legend(fontsize=7, loc='lower left') # Fontsize 7
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45, labelsize=7) # Fontsize 7
        ax1.tick_params(axis='y', labelsize=7) # Fontsize 7
    
    # ========================================================================
    # 2. MARKET SENTIMENT GAUGE (Font Size Reduced)
    # ========================================================================
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_aspect('equal')
    sentiment_score = 0
    if analysis['pcr_oi'] < 0.7: sentiment_score += 0.5
    elif analysis['pcr_oi'] > 1.3: sentiment_score -= 0.5
    if analysis['net_delta'] > 0: sentiment_score += 0.5
    else: sentiment_score -= 0.5
    
    # Draw gauge (unchanged)
    theta = np.linspace(0, np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), color='white', linewidth=3, alpha=0.3)
    bearish_zone = np.linspace(0, np.pi/3, 50); neutral_zone = np.linspace(np.pi/3, 2*np.pi/3, 50); bullish_zone = np.linspace(2*np.pi/3, np.pi, 50)
    ax2.fill_between(np.cos(bearish_zone), np.sin(bearish_zone), alpha=0.3, color='red')
    ax2.fill_between(np.cos(neutral_zone), np.sin(neutral_zone), alpha=0.3, color='yellow')
    ax2.fill_between(np.cos(bullish_zone), np.sin(bullish_zone), alpha=0.3, color='green')
    needle_angle = np.pi * (1 - (sentiment_score + 1) / 2)
    ax2.plot([0, 0.8 * np.cos(needle_angle)], [0, 0.8 * np.sin(needle_angle)], color='white', linewidth=3) # Reduced linewidth
    ax2.plot(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle), 'o', color='white', markersize=8) # Reduced markersize
    
    # Labels (No Emojis)
    ax2.text(-1.2, 0.1, 'BEARISH', fontsize=7, color='red', fontweight='bold', ha='center') # Fontsize 7
    ax2.text(0, 1.2, 'NEUTRAL', fontsize=7, color='yellow', fontweight='bold', ha='center') # Fontsize 7
    ax2.text(1.2, 0.1, 'BULLISH', fontsize=7, color='green', fontweight='bold', ha='center') # Fontsize 7
    
    # Center text
    ax2.text(0, -0.5, f'PCR: {analysis["pcr_oi"]:.2f}', fontsize=8, ha='center', # Fontsize 8
            bbox=dict(boxstyle='round', facecolor='black', alpha=1))
    
    # Risk Level Annotation
    ax2.text(0, -1.2, f'RISK: {risk_level} ({risk_score}/7)', fontsize=8, ha='center', # Fontsize 8
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=risk_color, linewidth=1)) # Reduced linewidth

    ax2.set_title('[FOCUS] MARKET SENTIMENT', fontsize=8, fontweight='bold', pad=5) # Fontsize 8
    ax2.axis('off')
    
    # ========================================================================
    # 3. OI IMBALANCE (Font Size Reduced)
    # ========================================================================
    colors_imbalance = ['green' if x > 0 else 'red' for x in strike_analysis['net_oi']]
    ax3.barh(strike_analysis['strike'], strike_analysis['net_oi'], 
                    color=colors_imbalance, alpha=0.7)
    ax3.axvline(x=0, color='white', linestyle='-', linewidth=1.5)
    ax3.axhline(y=spot_price, color='yellow', linestyle='--', linewidth=1.5, label=f'Spot: ${spot_price:,.0f}')
    ax3.axhline(y=max_pain, color='magenta', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Max Pain: ${max_pain:,.0f}')
    
    ax3.set_xlabel('Net OI (Calls - Puts)', fontsize=7, fontweight='bold') # Fontsize 7
    ax3.set_ylabel('Strike Price', fontsize=7, fontweight='bold') # Fontsize 7
    ax3.set_title('[BALANCE] OI IMBALANCE (Call vs Put)', fontsize=8, fontweight='bold') # Fontsize 8
    ax3.legend(fontsize=7, loc='best') # Fontsize 7
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.tick_params(labelsize=7) # Fontsize 7
    
    # Annotation simplified
    max_imbalance = strike_analysis.loc[strike_analysis['net_oi'].abs().idxmax()]
    imbalance_type = "CALL" if max_imbalance['net_oi'] > 0 else "PUT"
    ax3.text(0.02, 0.98, f"Strongest {imbalance_type} OI: ${max_imbalance['strike']:,.0f}", 
            transform=ax3.transAxes, fontsize=7, va='top', # Fontsize 7
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # ========================================================================
    # 4. GAMMA WALL (Font Size Reduced)
    # ========================================================================
    colors_gamma = ['purple' if x == strike_analysis['total_gamma'].max() else '#9933ff' 
                   for x in strike_analysis['total_gamma']]
    ax4.bar(strike_analysis['strike'], strike_analysis['total_gamma'], 
            width=500, color=colors_gamma, alpha=0.8)
    ax4.axvline(x=spot_price, color='yellow', linestyle='--', linewidth=1.5, label=f'Spot: ${spot_price:,.0f}')
    
    max_gamma_strike = strike_analysis.loc[strike_analysis['total_gamma'].idxmax()]
    ax4.axvline(x=max_gamma_strike['strike'], color='red', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'Max Gamma: ${max_gamma_strike["strike"]:,.0f}')
    
    ax4.set_xlabel('Strike Price', fontsize=7, fontweight='bold') # Fontsize 7
    ax4.set_ylabel('Gamma Exposure', fontsize=7, fontweight='bold') # Fontsize 7
    ax4.set_title('[PIN] GAMMA WALL (Price Pinning Zone)', fontsize=8, fontweight='bold') # Fontsize 8
    ax4.legend(fontsize=7) # Fontsize 7
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7) # Fontsize 7
    
    # Interpretation simplified
    gamma_distance = abs(max_gamma_strike['strike'] - spot_price) / spot_price * 100
    if gamma_distance < 2:
        warning = "(WARN) HIGH GAMMA ZONE\nChoppy action likely"
        color = 'orange'
    else:
        warning = "(OK) LOW GAMMA ZONE\nSmoother moves possible"
        color = 'green'
    
    ax4.text(0.02, 0.98, warning, transform=ax4.transAxes, fontsize=7, va='top', # Fontsize 7
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor=color, linewidth=1)) # Reduced linewidth
    
    # ========================================================================
    # 5. DELTA PROFILE (Font Size Reduced)
    # ========================================================================
    colors_delta = ['#00ff00' if x > 0 else '#ff3333' for x in strike_analysis['net_delta']]
    ax5.bar(strike_analysis['strike'], strike_analysis['net_delta'], 
            width=500, color=colors_delta, alpha=0.75)
    ax5.axhline(y=0, color='white', linestyle='-', linewidth=1.0)
    ax5.axvline(x=spot_price, color='yellow', linestyle='--', linewidth=1.5)
    
    ax5.set_xlabel('Strike Price', fontsize=7, fontweight='bold') # Fontsize 7
    ax5.set_ylabel('Net Delta (Hedging Flow)', fontsize=7, fontweight='bold') # Fontsize 7
    ax5.set_title('[FLOW] DELTA PROFILE (Dealer Hedging)', fontsize=8, fontweight='bold') # Fontsize 8
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=7) # Fontsize 7
    
    # Interpretation simplified
    net_delta_text = f"Net Delta: {analysis['net_delta']:,.0f}\n"
    if analysis['net_delta'] > 100:
        net_delta_text += "(BULL) Strong Bullish Flow"
        delta_color = 'green'
    elif analysis['net_delta'] < -100:
        net_delta_text += "(BEAR) Strong Bearish Flow"
        delta_color = 'red'
    else:
        net_delta_text += "(NEUT) Neutral/Balanced"
        delta_color = 'yellow'
    
    ax5.text(0.98, 0.98, net_delta_text, transform=ax5.transAxes, fontsize=7, va='top', ha='right', # Fontsize 7
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor=delta_color, linewidth=1)) # Reduced linewidth
    
    # ========================================================================
    # 6. PCR TREND (Font Size Reduced)
    # ========================================================================
    if len(history['timestamps']) > 1:
        ax6.plot(history['timestamps'], history['pcr_oi'], 
                color='#ff6600', linewidth=1.5, marker='.', markersize=3, label='PCR (OI)')
        
        # Reference lines
        ax6.axhline(y=1.0, color='white', linestyle='-', linewidth=1.0, alpha=0.5, label='Neutral (1.0)')
        ax6.axhline(y=0.7, color='green', linestyle=':', linewidth=1.2, alpha=1, label='Bullish Zone')
        ax6.axhline(y=1.3, color='red', linestyle=':', linewidth=1.2, alpha=1, label='Bearish Zone')
        
        # Fill zones
        ax6.fill_between(history['timestamps'], 0, 0.7, alpha=0.1, color='green')
        ax6.fill_between(history['timestamps'], 1.3, 2, alpha=0.1, color='red')
        
        # Current PCR value box
        current_pcr = history['pcr_oi'][-1]
        if current_pcr < 0.7:
            pcr_status = "BULLISH"
            pcr_color = 'green'
        elif current_pcr > 1.3:
            pcr_status = "BEARISH"
            pcr_color = 'red'
        else:
            pcr_status = "NEUTRAL"
            pcr_color = 'yellow'
        
        ax6.text(0.98, 0.98, f'Current: {current_pcr:.2f}\n{pcr_status}', 
                transform=ax6.transAxes, fontsize=7, va='top', ha='right', # Fontsize 7
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=pcr_color, linewidth=1))
        
        ax6.set_xlabel('Time', fontsize=7, fontweight='bold') # Fontsize 7
        ax6.set_ylabel('Put/Call Ratio', fontsize=7, fontweight='bold') # Fontsize 7
        ax6.set_title('PCR TREND (Sentiment Over Time)', fontsize=8, fontweight='bold') # Fontsize 8
        ax6.legend(fontsize=7, loc='upper left') # Fontsize 7
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.tick_params(axis='x', rotation=45, labelsize=7) # Fontsize 7
        ax6.tick_params(axis='y', labelsize=7) # Fontsize 7
        ax6.set_ylim(0, min(2, max(history['pcr_oi']) * 1.2))
    
    # ========================================================================
    # 7. VOLUME HEAT MAP (Font Size Reduced)
    # ========================================================================
    strike_positions = np.arange(len(strike_analysis))
    
    call_bars = ax7.barh(strike_positions, strike_analysis['call_volume'], 
                        height=0.8, color='#00ff00', alpha=0.7, label='Call Volume')
    put_bars = ax7.barh(strike_positions, -strike_analysis['put_volume'], 
                       height=0.8, color='#ff3333', alpha=0.7, label='Put Volume')
    
    spot_idx = np.argmin(abs(strike_analysis['strike'] - spot_price))
    ax7.axhline(y=spot_idx, color='yellow', linestyle='--', linewidth=1.5, label=f'Spot: ${spot_price:,.0f}')
    
    ax7.axvline(x=0, color='white', linestyle='-', linewidth=1.0)
    ax7.set_yticks(strike_positions)
    ax7.set_yticklabels([f'${s:,.0f}' for s in strike_analysis['strike']], fontsize=7) # Fontsize 7
    ax7.set_xlabel('Volume (Calls Right, Puts Left)', fontsize=7, fontweight='bold') # Fontsize 7
    ax7.set_ylabel('Strike Price', fontsize=7, fontweight='bold') # Fontsize 7
    ax7.set_title('[HOT] VOLUME HEAT MAP (Trading Activity)', fontsize=8, fontweight='bold') # Fontsize 8
    ax7.legend(fontsize=7, loc='best') # Fontsize 7
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Highlight most active strike simplified
    most_active = strike_analysis.loc[(strike_analysis['call_volume'] + strike_analysis['put_volume']).idxmax()]
    ax7.text(0.02, 0.98, f"Hottest Strike: ${most_active['strike']:,.0f}", 
            transform=ax7.transAxes, fontsize=7, va='top', # Fontsize 7
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

    # ========================================================================
    # 8. TRADE SIGNAL SUMMARY (Font Size Reduced)
    # ========================================================================
    ax8.axis('off')
    
    # Calculate comprehensive signals
    bullish_signals = []
    bearish_signals = []
    
    # PCR Signal
    if analysis['pcr_oi'] < 0.7:
        bullish_signals.append("PCR < 0.7 (Bullish)")
    elif analysis['pcr_oi'] > 1.3:
        bearish_signals.append("PCR > 1.3 (Bearish)")
    
    # Delta Signal
    if analysis['net_delta'] > 100:
        bullish_signals.append("Strong Positive Delta")
    elif analysis['net_delta'] < -100:
        bearish_signals.append("Strong Negative Delta")
    
    # Build summary
    summary_text = "=" * 38 + "\n"
    summary_text += "TRADING SIGNALS SUMMARY\n"
    summary_text += "=" * 38 + "\n\n"
    
    summary_text += "BULLISH SIGNALS:\n"
    if bullish_signals:
        for signal in bullish_signals:
            summary_text += f"  + {signal}\n"
    else:
        summary_text += "  None\n"
    
    summary_text += "\nBEARISH SIGNALS:\n"
    if bearish_signals:
        for signal in bearish_signals:
            summary_text += f"  - {signal}\n"
    else:
        summary_text += "  None\n"
    
    summary_text += "\nKEY LEVELS:\n"
    summary_text += f"  Max Pain: ${analysis['max_pain']:,.0f}\n"
    if len(analysis['resistance_levels']) > 0:
        res = analysis['resistance_levels'].iloc[0]
        summary_text += f"  Resistance: ${res['strike']:,.0f}\n"
    if len(analysis['support_levels']) > 0:
        sup = analysis['support_levels'].iloc[0]
        summary_text += f"  Support: ${sup['strike']:,.0f}\n"
    
    summary_text += "\n" + "=" * 38 + "\n"
    
    # Overall recommendation
    total_bullish = len(bullish_signals)
    total_bearish = len(bearish_signals)
    
    if total_bullish > total_bearish + 1:
        summary_text += "RECOMMENDATION: BULLISH\n"
        summary_text += "Bias: Long\n"
        box_color = 'green'
    elif total_bearish > total_bullish + 1:
        summary_text += "RECOMMENDATION: BEARISH\n"
        summary_text += "Bias: Short\n"
        box_color = 'red'
    else:
        summary_text += "RECOMMENDATION: WAIT\n"
        summary_text += "Bias: Neutral/Mixed\n"
        box_color = 'yellow'
    
    summary_text += "\n" + "=" * 38 + "\n"
    summary_text += f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
    
    ax8.text(0.5, 0.5, summary_text, transform=ax8.transAxes, 
            fontsize=8, verticalalignment='center', horizontalalignment='center', # Fontsize 8
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.9, 
                     edgecolor=box_color, linewidth=2))

    # Main title
    fig.suptitle(f'[FOCUS] INSTITUTIONAL OPTIONS FLOW ANALYZER | {analysis["metadata"]["underlying"]} | Updates Every {UPDATE_INTERVAL/1000:.0f}s', 
                fontsize=10, fontweight='bold', color='cyan') # Fontsize 10

# Create animation
print(f"\n[START] Starting Real-Time Market Monitor...")
print(f"Reading from: {JSON_FILE}")
print(f"Update interval: {UPDATE_INTERVAL/1000} seconds")
print(f"\nTerminal shows simplified signals")
print(f"Charts show detailed technical analysis")
print(f"\nCONTROLS:  [E]xport Data  |  [R]eset History  |  [Q]uit")
print("\n" + "="*80)

anim = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL, cache_frame_data=False)

try:
    # Use rect to reserve space for the suptitle at the top
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()
except KeyboardInterrupt:
    print("\n\n[STOP] Monitoring stopped by user.")
    sys.exit(0)