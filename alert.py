import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrow, Wedge, Polygon
from collections import deque
import platform
from datetime import datetime
from scipy.stats import norm
import warnings
import gc

warnings.filterwarnings('ignore')

# --- MAC M1/OS OPTIMIZATION ---
if platform.system() == 'Darwin':
    try:
        matplotlib.use('MacOSX')
    except:
        pass

# --- CONFIGURATION ---
JSON_FILE = "market_data.json"
REFRESH_INTERVAL = 500
STRIKE_RANGE = 8

# --- OPTIMIZATION GLOBALS ---
last_file_mtime = 0
cached_json_data = None
update_counter = 0

# --- GLOBAL STATE ---
mass_history = deque(maxlen=50)
oi_cumulative = {}
momentum_history = deque(maxlen=30)
price_movement_history = deque(maxlen=20)
last_zone_alert = None # Kept for alert throttling

# New Global for manual OI Change calculation (since new JSON misses 'oich')
previous_oi_snapshot = {} 

# --- SETUP PLOTS (SPLIT INTO TWO FIGURES) ---
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 9})

# === FIGURE 1: MARKET STRUCTURE ANALYSIS ===
fig1 = plt.figure(figsize=(14, 9), dpi=100)
fig1.canvas.manager.set_window_title('Market Structure & Probability')

# Grid for Fig 1
gs1 = fig1.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
ax_mass = fig1.add_subplot(gs1[0, :])
ax_pain = fig1.add_subplot(gs1[1, 0])
ax_prob = fig1.add_subplot(gs1[1, 1])

# === FIGURE 2: ACTION DASHBOARD ===
fig2 = plt.figure(figsize=(7, 9), dpi=100)
fig2.canvas.manager.set_window_title('Momentum & Signals')

# Grid for Fig 2
gs2 = fig2.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.2)
ax_mom = fig2.add_subplot(gs2[0])
ax_sig = fig2.add_subplot(gs2[1])


# --- DATA PROCESSING FUNCTIONS ---

def load_data():
    global last_file_mtime, cached_json_data, update_counter
    if not os.path.exists(JSON_FILE):
        return None
    try:
        current_mtime = os.path.getmtime(JSON_FILE)
        if current_mtime == last_file_mtime and cached_json_data is not None:
            return cached_json_data
        with open(JSON_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            cached_json_data = json.loads(content)
            last_file_mtime = current_mtime
            update_counter += 1
            if update_counter % 20 == 0:
                gc.collect()
            return cached_json_data
    except Exception as e:
        return None

def update_cumulative_oi(strikes_data):
    global oi_cumulative
    for strike, data in strikes_data.items():
        if strike not in oi_cumulative:
            oi_cumulative[strike] = {
                'CE_OI': data.get('CE_OI', 0), # Use .get for safety
                'PE_OI': data.get('PE_OI', 0), # Use .get for safety
                'CE_OI_BUILDUP': 0,
                'PE_OI_BUILDUP': 0,
                'CE_VOL_BUILDUP': 0,
                'PE_VOL_BUILDUP': 0,
                'last_update': datetime.now()
            }
        else:
            prev_ce = oi_cumulative[strike]['CE_OI']
            prev_pe = oi_cumulative[strike]['PE_OI']
            ce_change = data.get('CE_OI', 0) - prev_ce
            pe_change = data.get('PE_OI', 0) - prev_pe

            if ce_change > 0:
                oi_cumulative[strike]['CE_OI_BUILDUP'] += ce_change
            if pe_change > 0:
                oi_cumulative[strike]['PE_OI_BUILDUP'] += pe_change

            oi_cumulative[strike]['CE_VOL_BUILDUP'] = max(
                0, oi_cumulative[strike]['CE_VOL_BUILDUP'] * 0.8 + data.get('CE_VOL', 0)
            )
            oi_cumulative[strike]['PE_VOL_BUILDUP'] = max(
                0, oi_cumulative[strike]['PE_VOL_BUILDUP'] * 0.8 + data.get('PE_VOL', 0)
            )

            oi_cumulative[strike]['CE_OI'] = data.get('CE_OI', 0)
            oi_cumulative[strike]['PE_OI'] = data.get('PE_OI', 0)
            oi_cumulative[strike]['last_update'] = datetime.now()

    current_time = datetime.now()
    to_remove = []
    for strike in oi_cumulative:
        if (current_time - oi_cumulative[strike]['last_update']).seconds > 3600:
            to_remove.append(strike)
    for strike in to_remove:
        del oi_cumulative[strike]

    return oi_cumulative

def calculate_volume_weighted_oi(strikes_data):
    weighted_data = {}
    all_volumes = [d.get('CE_VOL', 0) + d.get('PE_VOL', 0) for d in strikes_data.values()]
    avg_volume = np.mean(all_volumes) if all_volumes else 1

    for strike, data in strikes_data.items():
        ce_vol = data.get('CE_VOL', 0)
        pe_vol = data.get('PE_VOL', 0)
        
        # Ensure avg_volume is not zero for safe division
        safe_avg_volume = avg_volume if avg_volume > 0 else 1 
        
        ce_surge = min((ce_vol / safe_avg_volume), 10.0)
        pe_surge = min((pe_vol / safe_avg_volume), 10.0)
        ce_w = ce_surge * (1 + np.log1p(ce_surge) / 10)
        pe_w = pe_surge * (1 + np.log1p(pe_surge) / 10)

        weighted_data[strike] = {
            'CE_OI_WEIGHTED': data.get('CE_OI', 0) * ce_w,
            'PE_OI_WEIGHTED': data.get('PE_OI', 0) * pe_w,
            'CE_ACTIVITY': ce_w,
            'PE_ACTIVITY': pe_w,
            'CE_VOL_SURGE': ce_surge,
            'PE_VOL_SURGE': pe_surge,
            'VOLUME_RATIO': ce_vol / pe_vol if pe_vol > 0 else float('inf')
        }
    return weighted_data

def calculate_max_pain(strikes_data, weighted_data, spot_price):
    max_pain = None;
    min_loss = float('inf');
    pain_data = {}
    all_strikes = sorted(strikes_data.keys())
    
    if not all_strikes: return spot_price, pain_data, 0.0 # Return sane defaults

    for test_strike in all_strikes:
        total_loss = 0
        for strike in all_strikes:
            # Use .get with 0 default for safety, though keys should exist from weighted_data logic
            ce = weighted_data.get(strike, {}).get('CE_OI_WEIGHTED', 0) 
            pe = weighted_data.get(strike, {}).get('PE_OI_WEIGHTED', 0)
            if test_strike > strike: total_loss += ce * (test_strike - strike)
            if test_strike < strike: total_loss += pe * (strike - test_strike)
        pain_data[test_strike] = total_loss
        if total_loss < min_loss: min_loss = total_loss; max_pain = test_strike

    dist = abs(spot_price - max_pain) if max_pain is not None else 0
    avg_gap = np.mean(np.diff(all_strikes)) if len(all_strikes) > 1 else 50
    pull = 1 - min(dist / (avg_gap * 3), 1.0)
    
    return max_pain if max_pain is not None else spot_price, pain_data, pull

def calculate_mass_concentration(strikes_data, weighted_data, cumulative_oi):
    strikes = [];
    masses = [];
    cum_masses = []
    
    for strike in sorted(strikes_data.keys()):
        # Use .get with default 0 for safety
        cur_mass = weighted_data.get(strike, {}).get('CE_OI_WEIGHTED', 0) + weighted_data.get(strike, {}).get('PE_OI_WEIGHTED', 0)
        hist_mass = cumulative_oi.get(strike, {}).get('CE_OI_BUILDUP', 0) + cumulative_oi.get(strike, {}).get('PE_OI_BUILDUP', 0)
        tot = (cur_mass * 0.7) + (hist_mass * 0.3)
        if tot > 0: 
            strikes.append(strike); 
            masses.append(cur_mass); 
            cum_masses.append(tot)

    if not strikes: return None, None, None, None, None
    
    strikes = np.array(strikes);
    cum_masses = np.array(cum_masses)
    
    total_mass = np.sum(cum_masses)
    if total_mass == 0: return None, None, None, None, None # Safety check for zero mass

    com = np.sum(strikes * cum_masses) / total_mass
    var = np.sum(cum_masses * (strikes - com) ** 2) / total_mass
    rad = np.sqrt(var)
    
    return com, com - rad, com + rad, masses, cum_masses

def calculate_momentum_distribution(strikes_data, weighted_data, cumulative_oi, spot_price, max_pain):
    all_strikes = sorted(strikes_data.keys())
    strike_spacing = np.mean(np.diff(all_strikes)) if len(all_strikes) > 1 else 50

    strikes = []
    weights = []
    directional_weights = []

    for strike in all_strikes:
        if strike in cumulative_oi:
            ce_build = cumulative_oi[strike].get('CE_OI_BUILDUP', 0)
            pe_build = cumulative_oi[strike].get('PE_OI_BUILDUP', 0)
            bullish_weight = pe_build * (1 if strike < spot_price else 0.5)
            bearish_weight = ce_build * (1 if strike > spot_price else 0.5)
            total_weight = ce_build + pe_build
            if total_weight > 0:
                strikes.append(strike)
                weights.append(total_weight)
                directional_weights.append(bullish_weight - bearish_weight)

    if not strikes:
        for strike in all_strikes:
            ce_oi = strikes_data[strike].get('CE_OI', 0)
            pe_oi = strikes_data[strike].get('PE_OI', 0)
            total = ce_oi + pe_oi
            if total > 0:
                strikes.append(strike)
                weights.append(total)
                directional_weights.append(pe_oi - ce_oi)

    if not strikes:
        return None

    strikes = np.array(strikes)
    weights = np.array(weights)
    directional_weights = np.array(directional_weights)
    
    total_weight = np.sum(weights)
    if total_weight == 0: return None # Safety for zero weight

    center_of_mass = np.sum(strikes * weights) / total_weight
    weighted_mean = center_of_mass
    variance = np.sum(weights * (strikes - weighted_mean) ** 2) / total_weight
    std_dev = np.sqrt(variance)

    total_ce_vol = sum(d.get('CE_VOL', 0) for d in strikes_data.values())
    total_pe_vol = sum(d.get('PE_VOL', 0) for d in strikes_data.values())
    total_vol = total_ce_vol + total_pe_vol

    volume_volatility = min(total_vol / 10000, 3.0)
    adjusted_std_dev = std_dev * (0.8 + volume_volatility * 0.4)
    adjusted_std_dev = max(adjusted_std_dev, strike_spacing * 0.5)
    
    # Safety check for zero std_dev
    if adjusted_std_dev == 0: adjusted_std_dev = strike_spacing 

    total_directional_weight = np.sum(np.abs(directional_weights))
    normalized_bias = np.sum(directional_weights) / total_directional_weight if total_directional_weight > 0 else 0

    bias_shift = normalized_bias * adjusted_std_dev * 0.5
    distribution_mean = weighted_mean + bias_shift
    strike_levels = all_strikes

    prob_up_1 = 1 - norm.cdf(strike_levels[0] + strike_spacing, distribution_mean, adjusted_std_dev)
    prob_up_2 = 1 - norm.cdf(strike_levels[0] + strike_spacing * 2, distribution_mean, adjusted_std_dev)
    prob_up_3 = 1 - norm.cdf(strike_levels[0] + strike_spacing * 3, distribution_mean, adjusted_std_dev)

    prob_down_1 = norm.cdf(strike_levels[0] - strike_spacing, distribution_mean, adjusted_std_dev)
    prob_down_2 = norm.cdf(strike_levels[0] - strike_spacing * 2, distribution_mean, adjusted_std_dev)
    prob_down_3 = norm.cdf(strike_levels[0] - strike_spacing * 3, distribution_mean, adjusted_std_dev)

    conf_68 = (distribution_mean - adjusted_std_dev, distribution_mean + adjusted_std_dev)
    conf_95 = (distribution_mean - 2 * adjusted_std_dev, distribution_mean + 2 * adjusted_std_dev)
    momentum_score = normalized_bias * 100

    probabilities = {}
    for strike in strike_levels:
        if abs(strike - spot_price) <= strike_spacing * 3:
            prob = norm.pdf(strike, distribution_mean, adjusted_std_dev)
            probabilities[strike] = prob

    if probabilities:
        most_probable_strike = max(probabilities, key=probabilities.get)
        max_probability = probabilities[most_probable_strike]
    else:
        most_probable_strike = spot_price
        max_probability = norm.pdf(spot_price, distribution_mean, adjusted_std_dev)

    expected_move_up = prob_up_1 * strike_spacing + prob_up_2 * strike_spacing * 2 + prob_up_3 * strike_spacing * 3
    expected_move_down = prob_down_1 * strike_spacing + prob_down_2 * strike_spacing * 2 + prob_down_3 * strike_spacing * 3
    net_expected_move = expected_move_up - expected_move_down

    return {
        'mean': distribution_mean,
        'std_dev': adjusted_std_dev,
        'center_of_mass': center_of_mass,
        'momentum_score': momentum_score,
        'normalized_bias': normalized_bias,
        'probabilities': probabilities,
        'most_probable_strike': most_probable_strike,
        'max_probability': max_probability,
        'conf_68': conf_68,
        'conf_95': conf_95,
        'prob_up_1': prob_up_1,
        'prob_up_2': prob_up_2,
        'prob_up_3': prob_up_3,
        'prob_down_1': prob_down_1,
        'prob_down_2': prob_down_2,
        'prob_down_3': prob_down_3,
        'expected_move': net_expected_move,
        'strike_spacing': strike_spacing,
        'distribution_points': {
            'strikes': strikes,
            'weights': weights,
            'directional_weights': directional_weights
        }
    }

def calculate_momentum_signals(distribution_data, spot_price, max_pain):
    # Default return for safety
    default_return = {
        'signal': 'NEUTRAL', 'color': '#FFFF00', 'confidence': 0,
        'direction': 'SIDEWAYS', 'targets': [], 'stop_loss': None, 'risk_reward': 0.0,
        'momentum_score': 0, 'expected_move': 0
    }
    if distribution_data is None:
        return default_return

    dist = distribution_data
    momentum_score = dist['momentum_score']
    normalized_bias = dist['normalized_bias']

    if momentum_score > 50:
        signal = 'STRONG BUY'; color = '#00FF00'; direction = 'STRONGLY BULLISH'; confidence = min(90 + abs(momentum_score) / 2, 95)
    elif momentum_score > 20:
        signal = 'BUY'; color = '#88FF00'; direction = 'BULLISH'; confidence = min(70 + abs(momentum_score), 85)
    elif momentum_score < -50:
        signal = 'STRONG SELL'; color = '#FF0000'; direction = 'STRONGLY BEARISH'; confidence = min(90 + abs(momentum_score) / 2, 95)
    elif momentum_score < -20:
        signal = 'SELL'; color = '#FF8800'; direction = 'BEARISH'; confidence = min(70 + abs(momentum_score), 85)
    else:
        signal = 'HOLD'; color = '#FFFF00'; direction = 'NEUTRAL'; confidence = 50

    targets = []
    
    # Check if key data is present before calculating targets
    if 'most_probable_strike' not in dist or 'strike_spacing' not in dist:
        return default_return # Return safe defaults if necessary keys are missing

    # Target 1 (Most Probable Strike)
    if abs(normalized_bias) > 0.1:
        target1 = dist['most_probable_strike'] 
        prob = dist['prob_up_1'] * 100 if normalized_bias > 0 else dist['prob_down_1'] * 100
        if prob > 20:
            targets.append({'strike': target1, 'probability': prob, 'type': 'IMMEDIATE'})
            
    # Target 2 (2x Strike Spacing)
    if abs(normalized_bias) > 0.3:
        if normalized_bias > 0:
            target2 = spot_price + dist['strike_spacing'] * 2
            prob2 = dist['prob_up_2'] * 100
        else:
            target2 = spot_price - dist['strike_spacing'] * 2
            prob2 = dist['prob_down_2'] * 100
        if prob2 > 10:
            targets.append({'strike': target2, 'probability': prob2, 'type': 'SECONDARY'})

    # Stop Loss
    strike_spacing = dist['strike_spacing']
    if normalized_bias > 0:
        stop_loss = spot_price - strike_spacing * 1.5
    elif normalized_bias < 0:
        stop_loss = spot_price + strike_spacing * 1.5
    else:
        stop_loss = None

    risk_reward = 0.0
    if targets and stop_loss is not None:
        if normalized_bias > 0:
            reward = targets[0]['strike'] - spot_price
            risk = spot_price - stop_loss
        else:
            reward = spot_price - targets[0]['strike']
            risk = stop_loss - spot_price
            
        # Safety check: reward and risk must be positive for a valid ratio
        if risk > 0 and reward > 0:
            risk_reward = reward / risk

    return {
        'signal': signal, 'color': color, 'confidence': confidence, 'direction': direction,
        'targets': targets[:2], 'stop_loss': stop_loss, 'risk_reward': risk_reward,
        'momentum_score': momentum_score, 'expected_move': dist['expected_move']
    }

def draw_max_pain_analysis(ax, pain_data, max_pain, spot_price):
    ax.clear()
    ax.set_facecolor('#0a0a15')

    if not pain_data:
        ax.text(0.5, 0.5, "Waiting for Data...", ha='center', color='white')
        return

    sorted_strikes = sorted(pain_data.keys())
    pain_values = [pain_data[k] for k in sorted_strikes]
    max_p_val = max(pain_values) if pain_values else 1
    norm_pain = [p / max_p_val for p in pain_values]

    ax.plot(sorted_strikes, norm_pain, color='yellow', linewidth=1.5, alpha=0.8, label='Pain Curve')
    ax.fill_between(sorted_strikes, 0, norm_pain, color='yellow', alpha=0.1)

    ax.axvline(x=max_pain, color='red', linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(x=spot_price, color='cyan', linestyle='--', linewidth=2, alpha=0.9)

    # ... [Rest of draw_max_pain_analysis remains the same] ...
    try:
        idx = sorted_strikes.index(max_pain)
        if 0 < idx < len(sorted_strikes) - 1:
            val_mp = pain_data[max_pain]
            val_left = pain_data[sorted_strikes[idx - 1]]
            val_right = pain_data[sorted_strikes[idx + 1]]
            avg_rise = ((val_left - val_mp) + (val_right - val_mp)) / 2
            steepness = (avg_rise / val_mp) * 100 if val_mp > 0 else 0

            if steepness > 5.0: shape_str = "Deep 'V' (Strong Defense)"; shape_color = '#00FF00'
            elif steepness > 2.0: shape_str = "Standard Curve"; shape_color = '#FFFF00'
            else: shape_str = "Flat 'U' (Weak/Uncertain)"; shape_color = '#FF8888'
        else:
            shape_str = "Edge Data"; shape_color = 'white'
    except:
        shape_str = "Calculating..."; shape_color = 'white'

    diff = spot_price - max_pain
    if diff > 10:
        direction = "BEARISH Pull ↓"; dir_color = '#FF4444'; gap_text = f"Gap: {abs(diff):.0f} pts"
    elif diff < -10:
        direction = "BULLISH Pull ↑"; dir_color = '#44FF44'; gap_text = f"Gap: {abs(diff):.0f} pts"
    else:
        direction = "PINNED (Neutral)"; dir_color = '#FFFF00'; gap_text = "At Target"

    y_min, y_max = ax.get_ylim()
    ax.annotate('Smart Money\nTarget', xy=(max_pain, 0.8), xytext=(max_pain + 20, 0.85),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                color='red', fontsize=7, fontweight='bold', ha='left')
    ax.annotate('Current\nPrice', xy=(spot_price, 0.6), xytext=(spot_price - 20, 0.65),
                arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=5),
                color='cyan', fontsize=7, fontweight='bold', ha='right')

    info_text = f"MODE: {direction}\nSTRUCT: {shape_str}\n{gap_text}"
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=dir_color)
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', color='white', bbox=props)

    ax.set_title(f"MAX PAIN: {max_pain:.0f}", fontsize=10, fontweight='bold', color='white')
    ax.grid(alpha=0.15); ax.set_yticks([])


def create_probability_distribution_plot(ax, distribution_data, spot_price, max_pain, signals, supports=None, resistances=None):
    if not distribution_data:
        ax.clear(); ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', fontsize=12, color='white')
        ax.set_facecolor('#0a0a15'); ax.axis('off'); return

    dist = distribution_data
    ax.clear(); ax.set_facecolor('#0a0a15')

    x_min = dist['mean'] - 3.5 * dist['std_dev']
    x_max = dist['mean'] + 3.5 * dist['std_dev']
    x = np.linspace(x_min, x_max, 500)
    pdf = norm.pdf(x, dist['mean'], dist['std_dev'])
    
    max_pdf = np.max(pdf)
    if max_pdf == 0: # Safety check
        ax.clear(); ax.text(0.5, 0.5, "Cannot form distribution (Zero Std Dev)", ha='center', va='center', fontsize=12, color='white')
        ax.set_facecolor('#0a0a15'); ax.axis('off'); return
        
    pdf_normalized = pdf / max_pdf * 0.8

    ax.fill_between(x, 0, pdf_normalized, alpha=0.3, color='cyan', label='Probability Density')
    ax.plot(x, pdf_normalized, 'cyan', linewidth=2, alpha=0.7)

    conf_68 = dist['conf_68']; conf_95 = dist['conf_95']
    mask_68 = (x >= conf_68[0]) & (x <= conf_68[1])
    ax.fill_between(x[mask_68], 0, pdf_normalized[mask_68], alpha=0.2, color='yellow', label='68% Confidence')
    mask_95 = (x >= conf_95[0]) & (x <= conf_95[1])
    ax.fill_between(x[mask_95], 0, pdf_normalized[mask_95], alpha=0.1, color='orange', label='95% Confidence')

    if supports:
        for s in supports[:2]: ax.axvline(x=s['strike'], color='green', linestyle='--', linewidth=1, alpha=0.5)
    if resistances:
        for r in resistances[:2]: ax.axvline(x=r['strike'], color='red', linestyle='--', linewidth=1, alpha=0.5)

    markers_info = []
    spot_pdf = norm.pdf(spot_price, dist['mean'], dist['std_dev']) / max_pdf * 0.8
    ax.axvline(x=spot_price, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(spot_price, spot_pdf, 'wo', markersize=8, markeredgecolor='cyan', markeredgewidth=2)
    # --- MODIFICATION: SPOT PRICE DECIMAL PLACE IS .4F ---
    markers_info.append({'x': spot_price, 'y': spot_pdf, 'label': f'SPOT\n{spot_price:.4f}', 'color': 'white', 'priority': 1})

    max_pain_pdf = norm.pdf(max_pain, dist['mean'], dist['std_dev']) / max_pdf * 0.8
    ax.axvline(x=max_pain, color='yellow', linestyle=':', linewidth=1.5, alpha=0.7)
    markers_info.append({'x': max_pain, 'y': max_pain_pdf, 'label': f'MP\n{max_pain:.0f}', 'color': 'yellow', 'priority': 2})

    most_prob_strike = dist['most_probable_strike']
    most_prob_pdf = dist['max_probability'] / max_pdf * 0.8
    ax.plot(most_prob_strike, most_prob_pdf, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
    markers_info.append({'x': most_prob_strike, 'y': most_prob_pdf, 'label': f'Next\n{most_prob_strike:.0f}', 'color': 'lime', 'priority': 0})

    if signals and 'targets' in signals:
        for i, target in enumerate(signals['targets']):
            target_strike = target['strike']
            target_pdf = norm.pdf(target_strike, dist['mean'], dist['std_dev']) / max_pdf * 0.8
            color = 'lime' if 'IMMEDIATE' in target['type'] else 'green'
            marker = '^' if target_strike > spot_price else 'v'
            ax.plot(target_strike, target_pdf, marker, color=color, markersize=9, markeredgecolor='white', markeredgewidth=1.5)
            # Keeping strike and probability as .0f
            label = f'T{i + 1}: {target_strike:.0f}\n({target["probability"]:.0f}%)'
            markers_info.append({'x': target_strike, 'y': target_pdf, 'label': label, 'color': color, 'priority': 3 + i})

    markers_info.sort(key=lambda m: m['x'])
    x_range = x_max - x_min
    groups = []
    current_group = [markers_info[0]] if markers_info else []

    for i in range(1, len(markers_info)):
        if abs(markers_info[i]['x'] - markers_info[i - 1]['x']) < x_range * 0.15:
            current_group.append(markers_info[i])
        else:
            if current_group: groups.append(current_group)
            current_group = [markers_info[i]]
    if current_group: groups.append(current_group)

    for group in groups:
        group.sort(key=lambda m: m['priority'])
        for idx, marker in enumerate(group):
            if len(group) == 1: y_offset, ha_align, x_adjust = 0.10, 'center', 0
            elif len(group) == 2:
                y_offset = 0.12; ha_align = 'right' if idx == 0 else 'left'
                x_adjust = -x_range * 0.025 if idx == 0 else x_range * 0.025
            else:
                y_offset = [0.10, 0.25, 0.40, 0.10, 0.25][idx % 5]
                ha_align = 'center' if idx % 3 == 0 else ('left' if idx % 3 == 1 else 'right')
                x_adjust = 0 if idx % 3 == 0 else (x_range * 0.03 if idx % 3 == 1 else -x_range * 0.03)

            label_y = marker['y'] + y_offset
            if y_offset > 0.12 or abs(x_adjust) > 0:
                ax.plot([marker['x'], marker['x'] + x_adjust], [marker['y'], label_y], color=marker['color'], linewidth=0.7, alpha=0.5, linestyle=':')

            bbox_props = dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.85, edgecolor=marker['color'], linewidth=1)
            ax.text(marker['x'] + x_adjust, label_y, marker['label'], ha=ha_align, va='bottom', fontsize=6.5, color=marker['color'], fontweight='bold', bbox=bbox_props)

    ax.set_xlabel('Strike Price', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title(f"Probability Distribution", fontsize=11, fontweight='bold', color='cyan')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_yticklabels([]); ax.set_ylim(0, 1.1)

def create_momentum_dashboard(ax, distribution_data, signals, spot_price):
    ax.clear(); ax.set_facecolor('#0a0a15')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    if not distribution_data:
        ax.text(50, 50, "No Momentum Data", ha='center', va='center', fontsize=12, color='white'); return

    dist = distribution_data
    ax.text(50, 97, "MOMENTUM DASHBOARD", ha='center', fontsize=14, fontweight='bold', color='white')

    gauge_x, gauge_y, gauge_radius = 50, 75, 15
    ax.add_patch(Wedge((gauge_x, gauge_y), gauge_radius, 180, 0, facecolor='#333333', edgecolor='white', linewidth=2))

    momentum_score = dist['momentum_score']
    needle_angle = 180 + (momentum_score / 100) * 180
    ax.add_patch(Wedge((gauge_x, gauge_y), gauge_radius, 180, 216, facecolor='#FF4444', alpha=0.6, edgecolor='white', linewidth=1))
    ax.add_patch(Wedge((gauge_x, gauge_y), gauge_radius, 216, 324, facecolor='#FFFF44', alpha=0.6, edgecolor='white', linewidth=1))
    ax.add_patch(Wedge((gauge_x, gauge_y), gauge_radius, 324, 360, facecolor='#44FF44', alpha=0.6, edgecolor='white', linewidth=1))

    needle_length = gauge_radius * 0.9
    needle_x = gauge_x + needle_length * np.cos(np.radians(needle_angle))
    needle_y = gauge_y + needle_length * np.sin(np.radians(needle_angle))
    ax.plot([gauge_x, needle_x], [gauge_y, needle_y], 'white', linewidth=3)
    ax.plot(gauge_x, gauge_y, 'wo', markersize=8)

    ax.text(gauge_x, gauge_y - gauge_radius - 5, f"Momentum: {momentum_score:+.0f}", ha='center', fontsize=12,
            fontweight='bold', color=signals['color'] if signals else 'white')

    matrix_x, matrix_y = 50, 45
    prob_cells = [
        {'name': '↑3', 'prob': dist['prob_up_3'] * 100, 'color': '#00AA00'},
        {'name': '↑1', 'prob': dist['prob_up_1'] * 100, 'color': '#AAFF00'},
        {'name': 'Flat', 'prob': (1 - dist['prob_up_1'] - dist['prob_down_1']) * 100, 'color': '#FFFF00'},
        {'name': '↓1', 'prob': dist['prob_down_1'] * 100, 'color': '#FFAA00'},
        {'name': '↓3', 'prob': dist['prob_down_3'] * 100, 'color': '#FF0000'}
    ]
    cell_width, cell_height = 14, 8
    start_x = matrix_x - (len(prob_cells) * cell_width) / 2 + cell_width / 2

    for i, cell in enumerate(prob_cells):
        cell_x = start_x + i * cell_width
        ax.add_patch(Rectangle((cell_x - cell_width / 2, matrix_y - cell_height / 2), cell_width, cell_height,
                               facecolor=cell['color'], alpha=0.3, edgecolor=cell['color'], linewidth=2))
        ax.text(cell_x, matrix_y, f"{cell['prob']:.0f}%", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(cell_x, matrix_y - cell_height / 2 - 3, cell['name'], ha='center', va='top', fontsize=8, color=cell['color'])

    ax.text(0, 36, "BEARISH (-)", color='tomato', ha='center', fontsize=7, fontweight='bold')
    ax.text(-6, 32, "0 to -20: Weak", color='white', fontsize=6, alpha=0.7)
    ax.text(-6, 28, "-20 to -50: Mod.", color='white', fontsize=6, alpha=0.7)
    ax.text(-6, 24, "<-50: Strong", color='white', fontsize=6, alpha=0.7)

    ax.text(92, 36, "BULLISH (+)", color='lime', ha='center', fontsize=7, fontweight='bold')
    ax.text(92, 32, "0 to +20: Weak", color='white', fontsize=6, alpha=0.7)
    ax.text(92, 28, "+20 to +50: Mod.", color='white', fontsize=6, alpha=0.7)
    ax.text(92, 24, ">+50: Strong", color='white', fontsize=6, alpha=0.7)

    targets_x, targets_y = 50, 15
    if signals and signals.get('targets'):
        for i, target in enumerate(signals['targets']):
            y_pos = targets_y - i * 8
            ax.text(targets_x - 15, y_pos, f"TARGET {i + 1}:", ha='right', fontsize=10, color='lime', fontweight='bold')
            ax.text(targets_x - 12, y_pos, f"{target['strike']:.0f}", ha='left', fontsize=10, color='white')

        if signals.get('stop_loss') is not None:
            ax.text(targets_x + 10, targets_y, "SL:", ha='right', fontsize=10, color='red', fontweight='bold')
            ax.text(targets_x + 13, targets_y, f"{signals['stop_loss']:.0f}", ha='left', fontsize=10, color='white')

def identify_support_resistance(strikes_data, weighted_data, cumulative_oi, spot_price):
    all_strikes = sorted(strikes_data.keys())
    levels = []; tot_ce = 0; tot_pe = 0
    
    if not all_strikes: return [], [], {}, {} # Return safe empty data

    for strike in all_strikes:
        # Use .get with 0 default for safety
        ce = weighted_data.get(strike, {}).get('CE_OI_WEIGHTED', 0) + (
            cumulative_oi.get(strike, {}).get('CE_OI_BUILDUP', 0) * 0.5)
        pe = weighted_data.get(strike, {}).get('PE_OI_WEIGHTED', 0) + (
            cumulative_oi.get(strike, {}).get('PE_OI_BUILDUP', 0) * 0.5)
        act_ce = weighted_data.get(strike, {}).get('CE_ACTIVITY', 0); act_pe = weighted_data.get(strike, {}).get('PE_ACTIVITY', 0)
        ce_f = ce * act_ce; pe_f = pe * act_pe
        tot_ce += ce_f; tot_pe += pe_f
        levels.append({'strike': strike, 'ce_strength': ce_f, 'pe_strength': pe_f})

    for l in levels:
        l['ce_percent'] = (l['ce_strength'] / tot_ce * 100) if tot_ce > 0 else 0
        l['pe_percent'] = (l['pe_strength'] / tot_pe * 100) if tot_pe > 0 else 0

    levels.sort(key=lambda x: x['ce_strength'], reverse=True)
    res = [l for l in levels if l['strike'] > spot_price][:3]
    levels.sort(key=lambda x: x['pe_strength'], reverse=True)
    sup = [l for l in levels if l['strike'] < spot_price][:3]

    return sup, res, {l['strike']: {'percent': l['pe_percent']} for l in sup}, {
        l['strike']: {'percent': l['ce_percent']} for l in res}

def generate_trading_signal(strikes_data, spot_price, max_pain, supports, resistances, com, low_z, up_z, pull, momentum_signals):
    # Use .get with 0 default for safety when calculating sums
    ce_oi_chg = sum(d.get('CE_OI_CHG', 0) for d in strikes_data.values())
    pe_oi_chg = sum(d.get('PE_OI_CHG', 0) for d in strikes_data.values())
    ce_vol = sum(d.get('CE_VOL', 0) for d in strikes_data.values())
    pe_vol = sum(d.get('PE_VOL', 0) for d in strikes_data.values())
    tot_oi = sum(d.get('CE_OI', 0) + d.get('PE_OI', 0) for d in strikes_data.values())
    
    # Safe PCR calculation
    total_ce_oi = sum(d.get('CE_OI', 0) for d in strikes_data.values())
    pcr = sum(d.get('PE_OI', 0) for d in strikes_data.values()) / total_ce_oi if total_ce_oi > 0 else 1
    
    vol_oi_ratio = (ce_vol + pe_vol) / tot_oi if tot_oi > 0 else 0
    net_oi_chg = ce_oi_chg - pe_oi_chg

    score = 50; conf = 0; reasons = []

    if momentum_signals:
        momentum_score = momentum_signals.get('momentum_score', 0)
        score += momentum_score * 0.3
        if momentum_signals.get('direction'): reasons.append(f"Momentum: {momentum_signals['direction']}")

    if ce_oi_chg > 0 and pe_oi_chg > 0:
        if ce_oi_chg > pe_oi_chg * 1.5: score -= 15; reasons.append("Call writing dominant"); conf += 20
        elif pe_oi_chg > ce_oi_chg * 1.5: score += 15; reasons.append("Put writing dominant"); conf += 20
    elif net_oi_chg > 0: score -= 10; reasons.append("Call buildup > Put"); conf += 10
    else: score += 10; reasons.append("Put buildup > Call"); conf += 10

    if vol_oi_ratio > 0.15: conf += 25; reasons.append("High Volume Confirmation")
    elif vol_oi_ratio < 0.05: conf -= 20; reasons.append("⚠️ Low Volume (Weak)")

    if pcr > 1.5: score += 15; reasons.append(f"High PCR ({pcr:.2f})"); conf += 15
    elif pcr < 0.7: score -= 15; reasons.append(f"Low PCR ({pcr:.2f})"); conf += 15

    dist_pain = spot_price - max_pain
    if spot_price == 0: # Avoid division by zero if spot is somehow zero
        dist_pain_ratio = 0
    else:
        dist_pain_ratio = abs(dist_pain / spot_price)
        
    if dist_pain_ratio < 0.01: score += 0; reasons.append("At Max Pain")
    elif dist_pain < 0: score += 15; reasons.append("Below MP (Pull Up)")
    else: score -= 15; reasons.append("Above MP (Pull Down)")

    conf = min(max(conf, 0), 100)
    if score >= 75: sig = "STRONG BUY"; col = "#00FF00"; act = "GO LONG"
    elif score >= 60: sig = "BUY"; col = "#88FF00"; act = "Consider Long"
    elif score >= 45: sig = "HOLD"; col = "#FFFF00"; act = "Wait / Neutral"
    elif score >= 30: sig = "SELL"; col = "#FF8800"; act = "Consider Short"
    else: sig = "STRONG SELL"; col = "#FF0000"; act = "GO SHORT"
    if conf < 40: sig = "HOLD"; act = "Low Confidence - Wait"; col = "#888888"

    return {'signal': sig, 'color': col, 'action': act, 'score': score, 'conf': conf, 'reasons': reasons, 'pcr': pcr,
            'vol_oi': vol_oi_ratio, 'net_oi': net_oi_chg}

def print_trading_signal_to_console(spot, max_pain, sig, m_sig, sups, ress, com, lz, uz):
    """Prints a consolidated trading signal and market summary to the console."""
    
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # 1. ZONE/ALERT CHECK
    all_zones = []
    zone_width = 15
    # Check if COM is calculated
    if com is not None and lz is not None and uz is not None:
        all_zones.append({'type': 'MASS_CENTER', 'center': com, 'lower': com-zone_width, 'upper': com+zone_width, 'label': f'Center of Mass ({com:.0f})'})
    for i, sup in enumerate(sups[:2]):
        all_zones.append({'type': 'SUPPORT', 'level': i+1, 'center': sup['strike'], 'lower': sup['strike']-zone_width, 'upper': sup['strike']+zone_width, 'label': f'Support S{i+1} ({sup["strike"]:.0f})'})
    for i, res in enumerate(ress[:2]):
        all_zones.append({'type': 'RESISTANCE', 'level': i+1, 'center': res['strike'], 'lower': res['strike']-zone_width, 'upper': res['strike']+zone_width, 'label': f'Resistance R{i+1} ({res["strike"]:.0f})'})
    near_zones = [z for z in all_zones if z['lower'] <= spot <= z['upper']]

    
    # 2. CONSOLE OUTPUT
    
    # Header
    print("\n" + "=" * 80)
    print(f"| {'BTC OPTIONS TRADING SIGNAL':^78} |")
    print("=" * 80)
    print(f"| {'TIME: ' + current_time:<38} | {'SPOT: $' + f'{spot:.4f}':<38} |")
    print("-" * 80)

    # Core Signal & Confidence
    signal_line = f"| {'SIGNAL: ' + sig['signal']:<38} | {'ACTION: ' + sig['action']:<38} |"
    conf_line = f"| {'CONFIDENCE: ' + f'{sig["conf"]:.0f}%':<38} | {'SCORE: ' + f'{sig["score"]:.0f}':<38} |"
    print(signal_line)
    print(conf_line)
    print("-" * 80)
    
    # Prepare Safe Targets/SL/R:R
    has_targets = len(m_sig.get('targets', [])) > 0
    target1_strike = f'{m_sig["targets"][0]["strike"]:.0f}' if has_targets else 'N/A'
    target2_strike = f'{m_sig["targets"][1]["strike"]:.0f}' if len(m_sig.get('targets', [])) > 1 else 'N/A'
    sl_strike = f'{m_sig["stop_loss"]:.0f}' if m_sig.get('stop_loss') is not None else 'N/A'
    rr_ratio = f'{m_sig["risk_reward"]:.2f}' if m_sig.get('risk_reward', 0) > 0 else 'N/A'
    com_strike = f'{com:.0f}' if com is not None else 'N/A'
    
    print(f"| {'MAX PAIN: ' + f'{max_pain:.0f}':<38} | {'PCR: ' + f'{sig["pcr"]:.2f}':<38} |")
    print(f"| {'R/R: ' + rr_ratio:<38} | {'EXPECTED MOVE: ' + f'{m_sig["expected_move"]:.0f} pts':<38} |")
    print("-" * 80)
    
    # Actionable Levels
    print(f"| {'TARGETS / STOP LOSS':^78} |")
    print("-" * 80)
    print(f"| {'TP1: ' + target1_strike:<19}", end="")
    print(f"| {'TP2: ' + target2_strike:<19}", end="")
    print(f"| {'STOP LOSS (SL): ' + sl_strike:<19}", end="")
    print(f"| {'COM: ' + com_strike:<18} |")
    print("-" * 80)

    # Zone Alerts
    if near_zones:
        print(f"| {'WARNING: PRICE IN CRITICAL ZONE':^78} |")
        zone_info = " | ".join([z['label'] for z in near_zones])
        print(f"| {zone_info:<78} |")
        print("=" * 80)
    else:
        print(f"| {'MARKET SUMMARY: ' + ' | '.join(sig['reasons'][:2]):^78} |")
        print("=" * 80)


# --- MAIN LOOP ---

def update(frame):
    global last_zone_alert, last_file_mtime, previous_oi_snapshot

    # --- 1. Load Data ---
    if not os.path.exists(JSON_FILE):
        return

    try:
        current_mtime = os.path.getmtime(JSON_FILE)
        if current_mtime == last_file_mtime:
            return
    except:
        return

    raw = load_data()
    if not raw: 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for valid JSON data...")
        return

    # --- 2. New Parsing Logic ---
    opts = raw.get('options_chain', [])
    meta = raw.get('metadata', {})
    
    if not opts: 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] JSON loaded but options_chain is empty. Skipping analysis.")
        return

    # Get Spot Price (Handle case where metadata is 0)
    spot = meta.get('spot_price', 0)
    
    # Fallback Spot Calculation if metadata spot is 0
    if spot == 0 and opts:
        try:
            min_diff = float('inf')
            atm_strike = 0
            call_price = 0
            put_price = 0
            
            calls = {o.get('strike', 0): o for o in opts if o.get('type') == 'Call'}
            puts = {o.get('strike', 0): o for o in opts if o.get('type') == 'Put'}
            
            common_strikes = set(calls.keys()) & set(puts.keys())
            
            for k in common_strikes:
                c_p = calls[k].get('mark_price', 0)
                p_p = puts[k].get('mark_price', 0)
                diff = abs(c_p - p_p)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = k
                    call_price = c_p
                    put_price = p_p
            
            spot = atm_strike + (call_price - put_price) if atm_strike != 0 else opts[len(opts)//2].get('strike', 0)
        except Exception as e:
             spot = opts[len(opts)//2].get('strike', 0)
             # print(f"Spot Fallback Error: {e}") # Debugging line
    
    if not spot: 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Could not determine spot price. Skipping analysis.")
        return

    s_data = {}
    
    # Process Options Chain (New Structure)
    for o in opts:
        k = o.get('strike', 0)
        if k <= 0: continue
        
        if k not in s_data: 
            s_data[k] = {'CE_OI': 0, 'PE_OI': 0, 'CE_VOL': 0, 'PE_VOL': 0, 'CE_OI_CHG': 0, 'PE_OI_CHG': 0}
            
        t_raw = o.get('type', '')
        t = "CE" if t_raw == "Call" else ("PE" if t_raw == "Put" else "")
        
        oi = o.get('oi', 0)
        vol = o.get('volume', 0)
        
        # --- LOGIC TO CALCULATE MISSING OI CHANGE ---
        opt_key = f"{t}_{k}"
        prev_oi = previous_oi_snapshot.get(opt_key, oi)
        oich = oi - prev_oi
        previous_oi_snapshot[opt_key] = oi
        # --------------------------------------------

        if t == "CE":
            s_data[k]['CE_OI'] = oi
            s_data[k]['CE_VOL'] = vol
            s_data[k]['CE_OI_CHG'] = oich
        elif t == "PE":
            s_data[k]['PE_OI'] = oi
            s_data[k]['PE_VOL'] = vol
            s_data[k]['PE_OI_CHG'] = oich

    # --- 3. Run Analysis (Same as before) ---
    cum_oi = update_cumulative_oi(s_data)
    weighted = calculate_volume_weighted_oi(s_data)
    
    max_pain = meta.get('max_pain', 0)
    calc_mp, pain_d, pull = calculate_max_pain(s_data, weighted, spot)
    if max_pain == 0: max_pain = calc_mp
    
    com, lz, uz, _, cum_mass = calculate_mass_concentration(s_data, weighted, cum_oi)
    sups, ress, sd, rd = identify_support_resistance(s_data, weighted, cum_oi, spot)
    distribution_data = calculate_momentum_distribution(s_data, weighted, cum_oi, spot, max_pain)
    momentum_signals = calculate_momentum_signals(distribution_data, spot, max_pain)
    sig = generate_trading_signal(s_data, spot, max_pain, sups, ress, com, lz, uz, pull, momentum_signals)

    # --- 4. CONSOLE OUTPUT (HIGH-CONFIDENCE ALERT LOGIC) ---
    
    MIN_CONFIDENCE_THRESHOLD = 70

    if sig['conf'] >= MIN_CONFIDENCE_THRESHOLD:
        # Check for non-zero R/R before giving trade signal
        if momentum_signals.get('risk_reward', 0.0) >= 1.0:
            print("\n" + "#" * 80)
            print(f"| {'HIGH-CONFIDENCE TRADE ALERT!':^78} |")
            print("#" * 80)
            # Call the full print function only on high confidence
            print_trading_signal_to_console(spot, max_pain, sig, momentum_signals, sups, ress, com, lz, uz)
            # Reset alert for cleaner terminal (optional, but good practice)
            last_zone_alert = None 
        else:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current_time}] High Confidence ({sig['conf']:.0f}%) but Low R/R ({momentum_signals.get('risk_reward', 0.0):.2f}). Waiting for better setup...")
    else:
        # Print a minimal status update when no high-confidence trade is found
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] Low Confidence ({sig['conf']:.0f}%) - Signal: {sig['signal']} ({sig['action']}). Waiting for setup...")


    # --- 5. Update Plots (Visualization) ---
    if plt.fignum_exists(fig1.number):
        all_k = sorted(s_data.keys())
        if not all_k: 
            return # Skip plotting if no strikes exist after filtering
            
        close_k = min(all_k, key=lambda x: abs(x - spot))
        mid = all_k.index(close_k)
        start = max(0, mid - STRIKE_RANGE)
        end = min(len(all_k), mid + STRIKE_RANGE + 1)
        sel_k = all_k[start:end]

        ks = np.array(sel_k)
        step = ks[1] - ks[0] if len(ks) > 1 else 50
        ce = np.array([weighted.get(k, {}).get('CE_OI_WEIGHTED', 0) for k in sel_k])
        pe = np.array([weighted.get(k, {}).get('PE_OI_WEIGHTED', 0) for k in sel_k])
        
        # Ensure mapping is safe
        k_map = {s: i for i, s in enumerate(all_k)}
        cum_mass_list = cum_mass.tolist() if isinstance(cum_mass, np.ndarray) else (cum_mass if cum_mass else [])
        masses = [cum_mass_list[k_map.get(s, -1)] if k_map.get(s, -1) != -1 and len(cum_mass_list) > k_map.get(s, -1) else 0 for s in sel_k]


        ax_mass.clear()
        x = np.arange(len(ks))
        ax_mass.fill_between(x, 0, ce, color='#00FF00', alpha=0.4, label='Call OI')
        ax_mass.fill_between(x, ce, ce + pe, color='#FF3333', alpha=0.4, label='Put OI')
        ax_mass.plot(x, masses, color='white', linewidth=3, marker='o', label='Mass')

        sp_pos = np.interp(spot, ks, x)
        ax_mass.axvline(x=sp_pos, color='cyan', linestyle='--', linewidth=2)
        # --- MODIFICATION: SPOT PRICE DECIMAL PLACE IS .4F ---
        ax_mass.text(sp_pos + 0.1, 0.5, f"{spot:.4f}", transform=ax_mass.get_xaxis_transform(), color='cyan', rotation=90, va='center', ha='left', fontsize=8, fontweight='bold')
        ax_mass.text(sp_pos, ax_mass.get_ylim()[1], f"SPOT: {spot:.4f}", color='black', ha='center', va='bottom', fontsize=9, fontweight='bold', bbox=dict(facecolor='cyan', edgecolor='white', boxstyle='round,pad=0.3'))

        if max_pain in sel_k:
            mp_pos = np.interp(max_pain, ks, x)
            ax_mass.axvline(x=mp_pos, color='yellow', linestyle=':', linewidth=2, label='Max Pain')
            ax_mass.text(mp_pos + 0.1, 0.5, f"{max_pain:.0f}", transform=ax_mass.get_xaxis_transform(), color='yellow', rotation=90, va='center', ha='left', fontsize=8)

        if com and lz and uz:
            if lz <= ks[-1] and uz >= ks[0]:
                l_pos = np.interp(lz, ks, x); u_pos = np.interp(uz, ks, x); c_pos = np.interp(com, ks, x)
                ax_mass.axvspan(l_pos, u_pos, color='orange', alpha=0.15)
                ax_mass.axvline(x=c_pos, color='orange', linewidth=2)

        # Plotting S/R markers with checks
        for i, s in enumerate(sups):
            if s['strike'] in sel_k and all_k and len(x) > 1: # Check for data existence
                try:
                    idx = sel_k.index(s['strike']); px = x[idx]; val = masses[idx]
                    ax_mass.axvline(x=px, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
                    ax_mass.axvspan(px - 0.3, px + 0.3, color='green', alpha=0.1)
                    left_price = s['strike'] - (step * 0.3); right_price = s['strike'] + (step * 0.3)
                    ax_mass.text(px - 0.3, -0.05, f"{left_price:.0f}", transform=ax_mass.get_xaxis_transform(), rotation=90, ha='center', va='top', fontsize=7, fontweight='bold', color='lime', clip_on=False)
                    ax_mass.text(px + 0.3, -0.05, f"{right_price:.0f}", transform=ax_mass.get_xaxis_transform(), rotation=90, ha='center', va='top', fontsize=7, fontweight='bold', color='lime', clip_on=False)
                    ax_mass.plot(px, val, 'g^', markersize=10)
                    ax_mass.text(px, val * 1.15, f"S{i+1}", color='white', ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.7, edgecolor='lime'))
                except:
                    pass # Safely skip if indexing fails

        for i, r in enumerate(ress):
            if r['strike'] in sel_k and all_k and len(x) > 1: # Check for data existence
                try:
                    idx = sel_k.index(r['strike']); px = x[idx]; val = masses[idx]
                    ax_mass.axvline(x=px, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
                    ax_mass.axvspan(px - 0.3, px + 0.3, color='red', alpha=0.1)
                    left_price = r['strike'] - (step * 0.3); right_price = r['strike'] + (step * 0.3)
                    ax_mass.text(px - 0.3, -0.05, f"{left_price:.0f}", transform=ax_mass.get_xaxis_transform(), rotation=90, ha='center', va='top', fontsize=7, fontweight='bold', color='tomato', clip_on=False)
                    ax_mass.text(px + 0.3, -0.05, f"{right_price:.0f}", transform=ax_mass.get_xaxis_transform(), rotation=90, ha='center', va='top', fontsize=7, fontweight='bold', color='tomato', clip_on=False)
                    ax_mass.plot(px, val, 'rv', markersize=10)
                    ax_mass.text(px, val * 1.15, f"R{i+1}", color='white', ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7, edgecolor='tomato'))
                except:
                    pass # Safely skip if indexing fails

        ax_mass.set_title(f"MASS CONCENTRATION MAP", fontsize=12, fontweight='bold', color='white')
        ax_mass.set_xticks(x); ax_mass.set_xticklabels([f"{k:.0f}" for k in ks], rotation=45, fontsize=8)
        ax_mass.legend(loc='upper left', fontsize=8, ncol=4); ax_mass.grid(alpha=0.15)

        draw_max_pain_analysis(ax_pain, pain_d, max_pain, spot)
        create_probability_distribution_plot(ax_prob, distribution_data, spot, max_pain, momentum_signals, supports=sups, resistances=ress)

    if plt.fignum_exists(fig2.number):
        create_momentum_dashboard(ax_mom, distribution_data, momentum_signals, spot)

        ax_sig.clear(); ax_sig.set_facecolor('#0a0a15'); ax_sig.axis('off')
        ax_sig.set_xlim(0, 10); ax_sig.set_ylim(0, 10)
        ax_sig.text(5, 9, "TRADING SIGNAL", ha='center', fontsize=12, fontweight='bold', color='white')
        ax_sig.add_patch(FancyBboxPatch((1.5, 6.0), 7, 2, boxstyle="round,pad=0.1", fc=sig['color'], alpha=0.3, ec=sig['color'], linewidth=2))
        ax_sig.text(5, 7.2, sig['signal'], ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax_sig.text(5, 6.4, sig['action'], ha='center', fontsize=9, color='white', fontweight='bold')

        ax_sig.text(2, 4.5, "Score", ha='center', fontsize=8, color='white')
        sw = (sig['score'] / 100) * 2
        ax_sig.add_patch(Rectangle((1, 3.8), 2, 0.4, fc='#333333', ec='white'))
        ax_sig.add_patch(Rectangle((1, 3.8), sw, 0.4, fc=sig['color']))
        ax_sig.text(2, 4.0, f"{sig['score']:.0f}", ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        ax_sig.text(5, 4.5, "Confidence", ha='center', fontsize=8, color='white')
        cw = (sig['conf'] / 100) * 2
        ax_sig.add_patch(Rectangle((4, 3.8), 2, 0.4, fc='#333333', ec='white'))
        ax_sig.add_patch(Rectangle((4, 3.8), cw, 0.4, fc='#00FF00'))
        ax_sig.text(5, 4.0, f"{sig['conf']:.0f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        ax_sig.text(8, 4.5, "PCR", ha='center', fontsize=8, color='white')
        ax_sig.text(8, 4.0, f"{sig['pcr']:.2f}", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        y_f = 2.5
        for r in sig['reasons'][:3]:
            col = '#FF8888' if '⚠️' in r else 'white'
            ax_sig.text(5, y_f, f"• {r}", ha='center', fontsize=8, color=col)
            y_f -= 0.8
        
        fig2.canvas.draw_idle()

try:
    mng = plt.get_current_fig_manager()
    try: mng.resize(*mng.window.maxsize())
    except: pass
except: pass

ani = FuncAnimation(fig1, update, interval=REFRESH_INTERVAL, cache_frame_data=False)
plt.show()