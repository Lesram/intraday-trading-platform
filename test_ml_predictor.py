#!/usr/bin/env python3
"""
Quick test script to diagnose ML predictor issues
"""
import os
import sys

sys.path.append(os.getcwd())

import logging

from advanced_ml_predictor import advanced_predictor, initialize_advanced_predictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_predictor():
    logger.info("🧪 Testing ML Predictor Initialization...")

    try:
        # Check initial state
        logger.info(f"🔍 Initial advanced_predictor state: {advanced_predictor}")

        # Initialize the predictor
        logger.info("🔧 Calling initialize_advanced_predictor()...")
        initialize_advanced_predictor()

        logger.info(f"🔍 After initialization advanced_predictor state: {advanced_predictor}")

        if advanced_predictor is None:
            logger.error("❌ Advanced predictor is None after initialization")
            return False

        logger.info(f"✅ Predictor initialized: {type(advanced_predictor)}")

        # Check models
        if hasattr(advanced_predictor, 'models'):
            logger.info(f"📊 Models available: {list(advanced_predictor.models.keys())}")
            models_loaded = {k: v is not None for k, v in advanced_predictor.models.items()}
            logger.info(f"📊 Models loaded status: {models_loaded}")
        else:
            logger.error("❌ Predictor has no models attribute")
            return False

        # Test prediction with sample data
        sample_market_data = {
            'symbol': 'TEST',
            'prices': [100, 101, 102, 101, 103, 102, 104, 105, 104, 106, 107, 105, 108, 107, 109, 110, 109, 111, 110, 112],
            'volumes': [1000, 1100, 900, 1200, 950, 1300, 850, 1400, 1050, 900, 1100, 1250, 950, 1350, 800, 1150, 1000, 1200, 950, 1400],
            'rsi': 55,
            'vix_proxy': 22,
            'market_trend': 0.02
        }

        logger.info("🔮 Testing prediction with sample data...")
        result = advanced_predictor.predict_with_ensemble(sample_market_data)

        logger.info(f"🎯 Prediction Result: {result}")

        if result and 'ensemble_prediction' in result:
            logger.info(f"✅ ML Predictor working! Final prediction: {result['ensemble_prediction']:.3f}")
            logger.info(f"   Individual predictions: {result.get('individual_predictions', {})}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            logger.error(f"❌ Invalid prediction result: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ ML Predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_predictor()
    print(f"\n🎯 ML Predictor Test: {'PASSED' if success else 'FAILED'}")
