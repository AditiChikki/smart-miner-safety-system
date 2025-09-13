import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google.colab import drive
import random

# Number of samples
n_samples = 100000

gas_limits = {
    'CO': {'safe': (0, 50), 'ltel': (50, 100), 'stel': (100, 500)},
    'CO2': {'safe': (0, 5000), 'ltel': (5000, 7500), 'stel': (7500, 30000)},
    'H2S': {'safe': (0, 10), 'ltel': (10, 50), 'stel': (50, 100)},
    'NH3': {'safe': (0, 25), 'ltel': (25, 30), 'stel': (30, 35)}
}

# Define ranges for ECG intervals
low_pr_range = (80, 120)
low_qt_range = (300, 350)
low_st_range = (60, 80)

normal_pr_range = (120, 200)
normal_qt_range = (350, 450)
normal_st_range = (80, 120)

elevated_pr_range = (200, 220)
elevated_qt_range = (450, 470)
elevated_st_range = (120, 140)

high_pr_range = (220, 250)
high_qt_range = (470, 490)
high_st_range = (140, 160)

def generate_gas_levels(gas_type, n_samples):
    limits = gas_limits[gas_type]
    safe_samples = int(n_samples * 0.993)
    ltel_samples = int(n_samples * 0.004)
    stel_samples = int(n_samples * 0.003)

    safe_levels = np.random.uniform(*limits['safe'], safe_samples)
    ltel_levels = np.random.uniform(*limits['ltel'], ltel_samples)
    stel_levels = np.random.uniform(*limits['stel'], stel_samples)

    gas_levels = np.concatenate([safe_levels, ltel_levels, stel_levels])
    np.random.shuffle(gas_levels)
    return gas_levels

gas_data = {gas: generate_gas_levels(gas, n_samples) for gas in gas_limits.keys()}

df = pd.DataFrame(gas_data)

def flag_gas_exposure(row):
    if any(gas_limits[gas]['stel'][0] <= row[gas] <= gas_limits[gas]['stel'][1] for gas in gas_limits):
        return 2
    elif any(gas_limits[gas]['ltel'][0] <= row[gas] <= gas_limits[gas]['ltel'][1] for gas in gas_limits):
        return 1
    return 0

def generate_ecg_intervals(row):
    max_severity = row['GasExposureFlag']

    if max_severity == 0:
        # Randomly assign low values in 5% of cases
        if random.random() < 0.05:
            pr = np.random.uniform(*low_pr_range)
            qt = np.random.uniform(*low_qt_range)
            st = np.random.uniform(*low_st_range)
        else:
            pr = np.random.uniform(*normal_pr_range)
            qt = np.random.uniform(*normal_qt_range)
            st = np.random.uniform(*normal_st_range)
    elif max_severity == 1:
        pr = np.random.uniform(*elevated_pr_range)
        qt = np.random.uniform(*elevated_qt_range)
        st = np.random.uniform(*elevated_st_range)
    else:
        pr = np.random.uniform(*high_pr_range)
        qt = np.random.uniform(*high_qt_range)
        st = np.random.uniform(*high_st_range)

    return pd.Series([pr, qt, st])

def flag_cardiac_abnormalities(row):
    if row['GasExposureFlag'] == 2:
        return 2 if random.random() < 0.85 else 1  # 85% chance for severe abnormality
    elif row['GasExposureFlag'] == 1:
        return 1 if random.random() < 0.6 else 0  # 60% chance for moderate abnormality
    elif row['PR'] < low_pr_range[1] or row['QT'] < low_qt_range[1] or row['ST'] < low_st_range[1]:
        return 2 if random.random() < 0.9 else 1  # Low values have 70% moderate, 30% severe abnormality
    return 0

def determine_stress_level(row):
    if row['GasExposureFlag'] == 2 and row['CardiacAbnormalityFlag'] == 2:
        return 'high'
    elif row['GasExposureFlag'] == 2 or row['CardiacAbnormalityFlag'] == 2:
        return 'high' if random.random() < 0.75 else 'moderate'
    elif row['CardiacAbnormalityFlag'] == 1:
        return 'moderate'
    return 'low'

df['GasExposureFlag'] = df.apply(flag_gas_exposure, axis=1)
df[['PR', 'QT', 'ST']] = df.apply(generate_ecg_intervals, axis=1)
df['HRV'] = df['PR'].rolling(window=10).std().fillna(0)
df['CardiacAbnormalityFlag'] = df.apply(flag_cardiac_abnormalities, axis=1)
df['StressLevel'] = df.apply(determine_stress_level, axis=1)

drive.mount('/content/drive')
save_path = '/content/drive/My Drive/RRRworker_protection_data_with_flags.csv'
df.to_csv(save_path, index=False)

print(f"File saved to: {save_path}")
print(df[['PR', 'QT', 'ST']].describe())
print(df['CardiacAbnormalityFlag'].value_counts(normalize=True))
