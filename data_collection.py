import serial
import re
import numpy as np
import csv

# ================================
# USER SETTINGS
# ================================
COM_PORT = "COM7"      # Change if needed
BAUD = 115200
TOTAL_SAMPLES = 500    # Total number of samples you want to collect
OUTPUT_FILE = "indexed_csi.csv"

# ================================
# Connect to ESP32
# ================================
ser = serial.Serial(COM_PORT, BAUD, timeout=1)
ser.setDTR(False)
ser.setRTS(False)

print(f"\n📡 Listening for CSI packets on {COM_PORT}")
print(f"🎯 Target samples: {TOTAL_SAMPLES}\n")

# ================================
# Parse CSI from ESP32 line
# ================================
def parse_csi(line):
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        return None, None

    parts = match.group(1).split(";")
    amps = []
    phases = []

    for p in parts:
        if "," not in p:
            continue
        try:
            a, ph = p.split(",")
            amps.append(float(a))
            phases.append(float(ph))
        except:
            return None, None

    # Expecting EXACT 64 amplitude-phase pairs (128 values)
    if len(amps) != 64 or len(phases) != 64:
        return None, None

    return np.array(amps), np.array(phases)

# ================================
# Create CSV header (64 amps + 64 phases)
# ================================
header = [f"amp_{i}" for i in range(64)] + \
         [f"phase_{i}" for i in range(64)] + \
         ["index"]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ================================
# MAIN DATA COLLECTION LOOP
# ================================
count = 0

while count < TOTAL_SAMPLES:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if "CSI:" not in line:
            continue

        amps, phases = parse_csi(line)
        if amps is None:
            continue

        # Build row (128 features + index)
        row = list(amps) + list(phases) + [count]

        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # PRINT status
        print(f"📦 Sample #{count} recorded")

        count += 1

    except KeyboardInterrupt:
        print("\n⛔ Stopped manually.")
        break
    except Exception as e:
        print("⚠️ Error:", e)
        continue

print(f"\n🎉 DONE! Saved all samples to {OUTPUT_FILE}")
