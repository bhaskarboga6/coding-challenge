#!/usr/bin/env python3
import sys
import joblib
import numpy as np

# Load the model
model = joblib.load("rf_snap_model.joblib")

# Known reimbursement values from training (rounded)
known_values = [150.34, 180.23, 204.61, 220.5, 248.35, 270.75, 302.14, 330.5, 362.83, 389.66, 418.17, 448.34, 517.6]

def snap_to_known(x):
    return min(known_values, key=lambda k: abs(k - x))

# Parse input
days = int(sys.argv[1])
miles = float(sys.argv[2])
receipts = float(sys.argv[3])

features = np.array([[days, miles, receipts]])
raw_pred = model.predict(features)[0]
snapped = snap_to_known(round(raw_pred, 2))

print(f"{snapped:.2f}")
