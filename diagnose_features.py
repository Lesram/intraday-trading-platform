#!/usr/bin/env python3
"""Quick diagnostic to check what features our models expect."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
from tensorflow import keras

# Check Random Forest expected features
print("🔍 Checking Random Forest model...")
rf_model = joblib.load('rf_ensemble_v2.pkl')
print(f"RF expects {rf_model.n_features_in_} features")

# Check XGBoost expected features
print("\n🔍 Checking XGBoost model...")
xgb_model = joblib.load('xgb_ensemble_v2.pkl')
try:
    # XGBoost might have n_features_in_ attribute
    if hasattr(xgb_model, 'n_features_in_'):
        print(f"XGB expects {xgb_model.n_features_in_} features")
    else:
        # Try to infer from other attributes
        print("XGB feature count not directly available, will need manual checking")
        # Check if we can get feature names or importance
        if hasattr(xgb_model, 'feature_names_in_'):
            print(f"XGB feature names: {len(xgb_model.feature_names_in_)}")
except Exception as e:
    print(f"XGB check error: {e}")

# Check LSTM expected features
print("\n🔍 Checking LSTM model...")
lstm_model = keras.models.load_model('lstm_ensemble_best.keras')
print(f"LSTM input shape: {lstm_model.input_shape}")

# Check feature scaler
print("\n🔍 Checking feature scaler...")
scaler = joblib.load('feature_scaler_v2.gz')
if hasattr(scaler, 'n_features_in_'):
    print(f"Scaler expects {scaler.n_features_in_} features")
else:
    print("Scaler feature count not available")

print("\n🎯 Now let's see what features we're actually generating...")

# Import our ML predictor
from advanced_ml_predictor import AdvancedMLPredictor

predictor = AdvancedMLPredictor()

# Test with sample data
sample_data = {
    'price': 150.0,
    'volume': 1000000,
    'rsi': 55.0,
    'macd': 0.5,
    'sma': 148.0,
    'volatility': 0.02,
    'momentum': 0.01,
    'prices': [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165],
    'volumes': [900000, 950000, 1000000, 1050000, 1100000, 1000000, 950000, 1000000, 1050000, 1100000, 1000000]
}

features = predictor.prepare_features(sample_data)
print(f"Our feature generator creates {features.shape[1]} features")
print(f"Feature values: {features[0]}")
