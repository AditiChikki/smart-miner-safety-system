import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from google.colab import drive
import joblib

# Load dataset
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/RRRworker_protection_data_with_flags.csv'
df = pd.read_csv(data_path)

# Fields for anomaly detection
fields = ['CO', 'CO2', 'H2S', 'NH3', 'PR', 'QT', 'ST', 'HRV']

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[fields])

# Step 1: Train Autoencoder on Sensor Data
input_dim = len(fields)
autoencoder = Sequential([
    Input(shape=(input_dim,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(df_scaled, df_scaled, epochs=100, batch_size=256, shuffle=True, verbose=0)

# Step 2: Compute Reconstruction Errors
reconstructed = autoencoder.predict(df_scaled)
reconstruction_error = np.mean(np.abs(df_scaled - reconstructed), axis=1)
df['Reconstruction_Error'] = reconstruction_error

# Step 3: Train Isolation Forest on Sensor Data + Reconstruction Errors
iso_features = np.column_stack((df_scaled, reconstruction_error.reshape(-1, 1)))
iso_forest = IsolationForest(contamination=0.007, random_state=42)
df['IsoForest_Anomaly'] = iso_forest.fit_predict(iso_features)
df['IsoForest_Anomaly'] = df['IsoForest_Anomaly'].apply(lambda x: 1 if x == -1 else 0)

joblib.dump(iso_forest, '/content/drive/My Drive/anomaly_isolation_forest.pkl')
print("Isolation Forest model saved.")


# Step 4: Train a New Autoencoder with All Features (Hybrid Learning)
hybrid_features = np.column_stack((df_scaled, df['IsoForest_Anomaly'].values.reshape(-1, 1)))

hybrid_autoencoder = Sequential([
    Input(shape=(input_dim + 1,)),  # Additional input: IsoForest_Anomaly
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim + 1, activation='sigmoid')
])

hybrid_autoencoder.compile(optimizer='adam', loss='mse')
hybrid_autoencoder.fit(hybrid_features, hybrid_features, epochs=100, batch_size=256, shuffle=True, verbose=0)
# Save Hybrid Autoencoder
hybrid_autoencoder.save('/content/drive/My Drive/hybrid_autoencoder.h5')
print("Hybrid Autoencoder model saved.")

# Step 5: Compute Hybrid Autoencoder Reconstruction Errors
hybrid_reconstructed = hybrid_autoencoder.predict(hybrid_features)
hybrid_reconstruction_error = np.mean(np.abs(hybrid_features - hybrid_reconstructed), axis=1)
df['Hybrid_Reconstruction_Error'] = hybrid_reconstruction_error

# Step 6: Set a Threshold for Hybrid Autoencoder Anomalies
#threshold = np.mean(hybrid_reconstruction_error[df['IsoForest_Anomaly'] == 0]) + 0.6 * np.std(hybrid_reconstruction_error[df['IsoForest_Anomaly'] == 0])
threshold = 0.009
print(threshold)
df['Hybrid_Anomaly'] = (df['Hybrid_Reconstruction_Error'] > threshold).astype(int)

# Step 7: Apply Manual Thresholding
manual_thresholds = {
    'CO': (0, 50),
    'CO2': (0, 5000),
    'H2S': (0, 10),
    'NH3': (0, 25),
    'PR': (120, 200),
    'QT': (350, 450),
    'ST': (80, 120)
}

# Step 8: Final Anomaly Decision (Hybrid Autoencoder with Manual Thresholding)
def final_anomaly_decision(row):
    if row['Hybrid_Anomaly'] == 0:
        return 0  # If Hybrid Autoencoder says it's normal, keep it normal

    # If Hybrid Autoencoder says it's an anomaly, check manual thresholds
    for field, (low, high) in manual_thresholds.items():
        if not (low <= row[field] <= high):
            return 1  # Anomaly remains if it falls outside threshold

    return 0  # If all values are within threshold, override to normal

df['Final_Anomaly'] = df.apply(final_anomaly_decision, axis=1)



# Step 9: Compare with Ground Truth
true_labels = ((df['StressLevel'] == 'high') | (df['StressLevel'] == 'moderate')).astype(int)
print("Classification Report:")
print(classification_report(true_labels, df['Final_Anomaly']))

# Step 10: Save Results
save_path = '/content/drive/My Drive/Final_Anomaly_Detection_Hybrid.csv'
df.to_csv(save_path, index=False)
print(f"Results saved to: {save_path}")
