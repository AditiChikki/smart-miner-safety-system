# Smart Miner Safety System: Real-Time Mine Worker Safety System

An IoT-based smart safety system for underground mine workers, using an ESP32 microcontroller to monitor toxic gas levels and heart rate in real-time. The system runs on-device anomaly detection using Isolation Forest and Autoencoder models to generate immediate hazard alerts without relying entirely on cloud connectivity.


## Features

-  **ESP32-based real-time sensing**
-  **Gas Monitoring**: CO₂, SO₂, NH₃, CO (using MQ sensors)
-  **Heart Rate Monitoring** using optical sensors
-  **Anomaly Detection**: Isolation Forest + Autoencoder models
-  **Local Hazard Alerts** via buzzer or LED
-  **Optional Cloud Integration** for remote monitoring and data logging

##  Machine Learning Overview

The system uses a **hybrid anomaly detection** model:

- **Isolation Forest**: Detects rare outliers (e.g., sudden gas spikes)
- **Autoencoder**: Learns patterns from normal readings and flags abnormal cases
- These models are trained on sample mining data and optimized for ESP32 inference

### Hardware Requirements

- ESP32-Wroom-32 board
- MQ-series gas sensors (CO, CO₂, SO₂, NH₃)
- Pulse sensor or AD8232 heart rate module
- Buzzer or LED for alerts
- Jumper wires, breadboard, and power source

###  Software Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aditichikki/smart-miner-safety-system.git
   cd smart-miner-safety-system
