#!/usr/bin/env python3
"""Test script to validate ML predictor initialization and functionality."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging

from advanced_ml_predictor import advanced_predictor, get_advanced_predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_predictor_initialization():
    """Test ML predictor initialization with proper getter."""

    logger.info("🧪 Starting ML Predictor Initialization Test v2")

    # Check initial state
    logger.info(f"🔍 Initial global advanced_predictor state: {advanced_predictor}")

    # Use getter function
    predictor = get_advanced_predictor()

    # Check after getter
    logger.info(f"🔍 After getter advanced_predictor state: {advanced_predictor}")
    logger.info(f"🔍 Returned predictor object: {predictor}")

    if predictor is None:
        logger.error("❌ Advanced predictor is None after getter")
        return False
    else:
        logger.info(f"✅ Advanced predictor loaded successfully: {type(predictor)}")

        # Test prediction with sample data
        sample_data = {
            'price': 150.0,
            'volume': 1000000,
            'rsi': 55.0,
            'macd': 0.5,
            'sma': 148.0,
            'volatility': 0.02,
            'momentum': 0.01
        }

        try:
            prediction = predictor.predict_with_ensemble(sample_data)
            logger.info(f"🎯 Sample prediction result: {json.dumps(prediction, indent=2)}")

            if 'ensemble_prediction' in prediction:
                logger.info("✅ Prediction contains expected ensemble_prediction key")
                return True
            else:
                logger.error("❌ Prediction missing ensemble_prediction key")
                return False

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return False

if __name__ == "__main__":
    success = test_predictor_initialization()
    if success:
        logger.info("🎉 All ML predictor tests passed!")
    else:
        logger.error("💥 ML predictor tests failed!")
