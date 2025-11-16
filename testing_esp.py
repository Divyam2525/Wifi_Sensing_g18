import serial
import re
import numpy as np
import joblib
import time

# ============================================
# USER SETTINGS
# ============================================
COM_PORT = "COM7"      # ESP32 serial port
BAUD = 115200

# ============================================
# Load trained model + scaler + label encoder
# ============================================
print("📦 Loading model and tools...")

model = joblib.load("esp_model/rf_model.pkl")
scaler = joblib.load("esp_model/scaler.pkl")
label_encoder = joblib.load("esp_model/labels.pkl")

print("✅ Model loaded.")
print("Classes:", label_encoder.classes_)

# ============================================
# Connect to ESP32
# ============================================
print(f"\n🔌 Connecting to ESP32 on {COM_PORT}...")
ser = serial.Serial(COM_PORT, BAUD, timeout=1)
ser.setDTR(False)
ser.setRTS(False)
time.sleep(1)

print("📡 Listening for CSI packets...\n")

# ============================================
# Parse raw CSI
# ============================================
def parse_csi(line):
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        return None

    parts = match.group(1).split(";")
    amps, phases = [], []

    for p in parts:
        if "," not in p:
            continue
        try:
            a, ph = p.split(",")
            amps.append(float(a))
            phases.append(float(ph))
        except:
            return None

    # Expect exactly 64 amplitude & 64 phase values
    if len(amps) != 64 or len(phases) != 64:
        return None

    return np.array(amps + phases)   # shape = (128,)

# ============================================
# Real-time prediction loop
# ============================================
while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if "CSI:" not in line:
            continue

        features = parse_csi(line)
        if features is None:
            continue

        # Reshape for scaler + model
        X = features.reshape(1, -1)

        # Scale (same as training)
        X_scaled = scaler.transform(X)

        # Predict class
        pred = model.predict(X_scaled)[0]
        label = label_encoder.inverse_transform([pred])[0]

        # Predict probabilities too (useful)
        proba = model.predict_proba(X_scaled)[0]
        p_empty = proba[label_encoder.transform(["empty"])[0]]
        p_metal = proba[label_encoder.transform(["metal"])[0]]

        # Output
        print("===================================")
        print("Prediction :", label.upper())
        print(f"P(empty)  : {p_empty:.3f}")
        print(f"P(metal)  : {p_metal:.3f}")
        print("===================================\n")

    except KeyboardInterrupt:
        print("\n⛔ Stopped manually.")
        break

    except Exception as e:
        print("⚠️ Error:", e)
        continue
