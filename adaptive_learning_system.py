#!/usr/bin/env python3
"""
🧠 PRIORITY 3: ADAPTIVE LEARNING LOOP
Online retraining and drift detection for sustainable AI performance
"""

import os
import sys
import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

# ML imports
try:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score
    import joblib
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")

# Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_env_file():
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

class ModelType(Enum):
    LSTM = "lstm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    COVARIATE_SHIFT = "covariate_shift"

@dataclass
class ModelPerformanceMetrics:
    """Track model performance over time"""
    timestamp: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_confidence: float
    prediction_count: int
    correct_predictions: int
    profit_attribution: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DriftDetectionResult:
    """Results from drift detection analysis"""
    drift_detected: bool
    drift_type: DriftType
    drift_magnitude: float
    drift_confidence: float
    affected_features: List[str]
    recommendation: str
    timestamp: str
    
    def to_dict(self):
        return {
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.value,
            'drift_magnitude': self.drift_magnitude,
            'drift_confidence': self.drift_confidence,
            'affected_features': self.affected_features,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp
        }

@dataclass
class RetrainingConfig:
    """Configuration for adaptive retraining"""
    retrain_frequency_hours: int = 168  # Weekly
    min_performance_threshold: float = 0.60
    drift_threshold: float = 0.15
    min_new_samples: int = 100
    validation_split: float = 0.2
    max_retrain_attempts: int = 3
    enable_online_learning: bool = True
    enable_ensemble_reweighting: bool = True
    backup_models_count: int = 5

class ModelPerformanceTracker:
    """Track and analyze model performance over time"""
    
    def __init__(self, storage_path: str = "models/performance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.performance_history = {}
        self.load_performance_history()
    
    def load_performance_history(self):
        """Load historical performance data"""
        try:
            history_file = self.storage_path / "performance_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data
                logging.info(f"📊 Loaded performance history: {len(self.performance_history)} entries")
            else:
                self.performance_history = {}
                logging.info("📊 Starting fresh performance tracking")
        except Exception as e:
            logging.error(f"❌ Failed to load performance history: {e}")
            self.performance_history = {}
    
    def save_performance_history(self):
        """Save performance history to disk"""
        try:
            history_file = self.storage_path / "performance_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            logging.info(f"💾 Saved performance history: {len(self.performance_history)} entries")
        except Exception as e:
            logging.error(f"❌ Failed to save performance history: {e}")
    
    def record_performance(self, metrics: ModelPerformanceMetrics):
        """Record new performance metrics"""
        timestamp = metrics.timestamp
        model_type = metrics.model_type
        
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        
        self.performance_history[model_type].append(metrics.to_dict())
        
        # Keep only last 1000 entries per model
        if len(self.performance_history[model_type]) > 1000:
            self.performance_history[model_type] = self.performance_history[model_type][-1000:]
        
        self.save_performance_history()
        logging.info(f"📈 Recorded performance for {model_type}: Accuracy {metrics.accuracy:.3f}")
    
    def get_recent_performance(self, model_type: str, hours_back: int = 24) -> List[Dict]:
        """Get recent performance metrics"""
        if model_type not in self.performance_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_metrics = []
        
        for metrics in self.performance_history[model_type]:
            metric_time = datetime.fromisoformat(metrics['timestamp'])
            if metric_time >= cutoff_time:
                recent_metrics.append(metrics)
        
        return recent_metrics
    
    def calculate_performance_trend(self, model_type: str, window_hours: int = 168) -> Dict:
        """Calculate performance trend over time window"""
        recent_metrics = self.get_recent_performance(model_type, window_hours)
        
        if len(recent_metrics) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0}
        
        # Calculate trend in accuracy
        accuracies = [m['accuracy'] for m in recent_metrics]
        if len(accuracies) >= 2:
            recent_avg = np.mean(accuracies[-10:])  # Last 10 measurements
            older_avg = np.mean(accuracies[:10])     # First 10 measurements
            
            trend_change = recent_avg - older_avg
            
            if trend_change > 0.02:
                trend = 'improving'
            elif trend_change < -0.02:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            trend_change = 0.0
        
        return {
            'trend': trend,
            'change': trend_change,
            'recent_accuracy': np.mean(accuracies[-5:]) if len(accuracies) >= 5 else 0.0,
            'sample_count': len(recent_metrics)
        }
    
    def detect_performance_anomalies(self, model_type: str) -> List[Dict]:
        """Detect performance anomalies"""
        recent_metrics = self.get_recent_performance(model_type, 168)  # Last week
        
        if len(recent_metrics) < 10:
            return []
        
        anomalies = []
        accuracies = [m['accuracy'] for m in recent_metrics]
        
        # Statistical anomaly detection
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        for i, metrics in enumerate(recent_metrics):
            accuracy = metrics['accuracy']
            z_score = abs((accuracy - mean_accuracy) / std_accuracy) if std_accuracy > 0 else 0
            
            if z_score > 2.5:  # 2.5 standard deviations
                anomalies.append({
                    'timestamp': metrics['timestamp'],
                    'accuracy': accuracy,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium'
                })
        
        return anomalies

class DriftDetector:
    """Detect data and concept drift in the model"""
    
    def __init__(self, storage_path: str = "models/drift"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.baseline_stats = {}
        self.load_baseline_stats()
    
    def load_baseline_stats(self):
        """Load baseline statistics for drift detection"""
        try:
            baseline_file = self.storage_path / "baseline_stats.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    self.baseline_stats = json.load(f)
                logging.info(f"📊 Loaded baseline statistics")
            else:
                self.baseline_stats = {}
                logging.info("📊 No baseline statistics found")
        except Exception as e:
            logging.error(f"❌ Failed to load baseline stats: {e}")
            self.baseline_stats = {}
    
    def save_baseline_stats(self):
        """Save baseline statistics"""
        try:
            baseline_file = self.storage_path / "baseline_stats.json"
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
            logging.info(f"💾 Saved baseline statistics")
        except Exception as e:
            logging.error(f"❌ Failed to save baseline stats: {e}")
    
    def update_baseline(self, feature_data: np.ndarray, feature_names: List[str]):
        """Update baseline statistics with new data"""
        try:
            baseline = {
                'timestamp': datetime.now().isoformat(),
                'feature_stats': {}
            }
            
            for i, feature_name in enumerate(feature_names):
                if i < feature_data.shape[1]:
                    feature_values = feature_data[:, i]
                    baseline['feature_stats'][feature_name] = {
                        'mean': float(np.mean(feature_values)),
                        'std': float(np.std(feature_values)),
                        'min': float(np.min(feature_values)),
                        'max': float(np.max(feature_values)),
                        'q25': float(np.percentile(feature_values, 25)),
                        'q50': float(np.percentile(feature_values, 50)),
                        'q75': float(np.percentile(feature_values, 75))
                    }
            
            self.baseline_stats = baseline
            self.save_baseline_stats()
            logging.info(f"📊 Updated baseline with {len(feature_names)} features")
            
        except Exception as e:
            logging.error(f"❌ Failed to update baseline: {e}")
    
    def detect_data_drift(self, new_data: np.ndarray, feature_names: List[str], 
                         threshold: float = 0.15) -> DriftDetectionResult:
        """Detect data drift using statistical tests"""
        try:
            if not self.baseline_stats or 'feature_stats' not in self.baseline_stats:
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type=DriftType.DATA_DRIFT,
                    drift_magnitude=0.0,
                    drift_confidence=0.0,
                    affected_features=[],
                    recommendation="No baseline available - establish baseline first",
                    timestamp=datetime.now().isoformat()
                )
            
            baseline_stats = self.baseline_stats['feature_stats']
            affected_features = []
            drift_scores = []
            
            for i, feature_name in enumerate(feature_names):
                if feature_name in baseline_stats and i < new_data.shape[1]:
                    baseline = baseline_stats[feature_name]
                    current_values = new_data[:, i]
                    
                    # Calculate statistical distance (simplified KL divergence)
                    current_mean = np.mean(current_values)
                    current_std = np.std(current_values)
                    
                    baseline_mean = baseline['mean']
                    baseline_std = baseline['std']
                    
                    # Normalized difference in means
                    mean_diff = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
                    
                    # Ratio of standard deviations
                    std_ratio = current_std / (baseline_std + 1e-8)
                    std_drift = abs(1 - std_ratio)
                    
                    # Combined drift score
                    drift_score = (mean_diff + std_drift) / 2
                    drift_scores.append(drift_score)
                    
                    if drift_score > threshold:
                        affected_features.append(feature_name)
            
            if len(drift_scores) == 0:
                overall_drift = 0.0
            else:
                overall_drift = np.mean(drift_scores)
            
            drift_detected = overall_drift > threshold
            confidence = min(overall_drift / threshold, 1.0) if threshold > 0 else 0.0
            
            if drift_detected:
                if len(affected_features) > len(feature_names) * 0.5:
                    recommendation = "Major drift detected - immediate retraining recommended"
                else:
                    recommendation = "Moderate drift detected - schedule retraining within 24 hours"
            else:
                recommendation = "No significant drift detected - continue monitoring"
            
            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=DriftType.DATA_DRIFT,
                drift_magnitude=overall_drift,
                drift_confidence=confidence,
                affected_features=affected_features,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"❌ Drift detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.DATA_DRIFT,
                drift_magnitude=0.0,
                drift_confidence=0.0,
                affected_features=[],
                recommendation=f"Drift detection error: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def detect_concept_drift(self, predictions: np.ndarray, actuals: np.ndarray,
                           recent_window: int = 100) -> DriftDetectionResult:
        """Detect concept drift based on prediction accuracy"""
        try:
            if len(predictions) < recent_window * 2:
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type=DriftType.CONCEPT_DRIFT,
                    drift_magnitude=0.0,
                    drift_confidence=0.0,
                    affected_features=[],
                    recommendation="Insufficient data for concept drift detection",
                    timestamp=datetime.now().isoformat()
                )
            
            # Compare recent performance vs historical
            recent_predictions = predictions[-recent_window:]
            recent_actuals = actuals[-recent_window:]
            
            historical_predictions = predictions[:-recent_window][-recent_window:]
            historical_actuals = actuals[:-recent_window][-recent_window:]
            
            # Calculate accuracy for both periods
            recent_accuracy = accuracy_score(recent_actuals, recent_predictions)
            historical_accuracy = accuracy_score(historical_actuals, historical_predictions)
            
            # Drift magnitude is the difference in accuracy
            accuracy_drop = historical_accuracy - recent_accuracy
            drift_magnitude = abs(accuracy_drop)
            
            # Detect significant performance drop
            drift_detected = accuracy_drop > 0.05  # 5% accuracy drop threshold
            confidence = min(drift_magnitude / 0.05, 1.0)
            
            if drift_detected:
                if accuracy_drop > 0.10:
                    recommendation = "Severe concept drift - immediate retraining required"
                else:
                    recommendation = "Concept drift detected - retraining recommended"
            else:
                recommendation = "No concept drift detected"
            
            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=DriftType.CONCEPT_DRIFT,
                drift_magnitude=drift_magnitude,
                drift_confidence=confidence,
                affected_features=['prediction_accuracy'],
                recommendation=recommendation,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"❌ Concept drift detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.CONCEPT_DRIFT,
                drift_magnitude=0.0,
                drift_confidence=0.0,
                affected_features=[],
                recommendation=f"Concept drift detection error: {e}",
                timestamp=datetime.now().isoformat()
            )

class AdaptiveRetrainer:
    """Handles adaptive model retraining"""
    
    def __init__(self, config: RetrainingConfig, storage_path: str = "models"):
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.retraining_history = []
        self.model_backups = {}
        
    def should_retrain(self, performance_tracker: ModelPerformanceTracker,
                      drift_detector: DriftDetector, model_type: str) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        try:
            # Check performance trend
            trend = performance_tracker.calculate_performance_trend(model_type)
            
            # Check for anomalies
            anomalies = performance_tracker.detect_performance_anomalies(model_type)
            
            # Performance-based triggers
            if trend['trend'] == 'declining' and abs(trend['change']) > 0.05:
                return True, f"Performance declining: {trend['change']:.3f}"
            
            if trend['recent_accuracy'] < self.config.min_performance_threshold:
                return True, f"Accuracy below threshold: {trend['recent_accuracy']:.3f}"
            
            if len(anomalies) > 3:  # Multiple anomalies
                return True, f"Multiple performance anomalies detected: {len(anomalies)}"
            
            # Time-based trigger
            recent_metrics = performance_tracker.get_recent_performance(
                model_type, self.config.retrain_frequency_hours
            )
            
            if len(recent_metrics) == 0:
                return True, "No recent performance data - scheduled retraining"
            
            # Check last retraining time
            if len(self.retraining_history) > 0:
                last_retrain = datetime.fromisoformat(self.retraining_history[-1]['timestamp'])
                time_since_retrain = datetime.now() - last_retrain
                
                if time_since_retrain.total_seconds() > self.config.retrain_frequency_hours * 3600:
                    return True, f"Scheduled retraining: {time_since_retrain.days} days since last retrain"
            
            return False, "No retraining needed"
            
        except Exception as e:
            logging.error(f"❌ Retrain decision error: {e}")
            return False, f"Error in retrain decision: {e}"
    
    def backup_model(self, model_type: str, model_path: str):
        """Create backup of current model"""
        try:
            backup_dir = self.storage_path / "backups" / model_type
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"model_backup_{timestamp}"
            
            if Path(model_path).exists():
                import shutil
                if Path(model_path).is_dir():
                    shutil.copytree(model_path, backup_path)
                else:
                    shutil.copy2(model_path, backup_path)
                
                # Track backup
                if model_type not in self.model_backups:
                    self.model_backups[model_type] = []
                
                self.model_backups[model_type].append({
                    'path': str(backup_path),
                    'timestamp': datetime.now().isoformat(),
                    'original_path': model_path
                })
                
                # Keep only last N backups
                if len(self.model_backups[model_type]) > self.config.backup_models_count:
                    old_backup = self.model_backups[model_type].pop(0)
                    try:
                        old_path = Path(old_backup['path'])
                        if old_path.exists():
                            if old_path.is_dir():
                                shutil.rmtree(old_path)
                            else:
                                old_path.unlink()
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to remove old backup: {e}")
                
                logging.info(f"📦 Model backup created: {backup_path}")
                return str(backup_path)
            else:
                logging.warning(f"⚠️ Model path not found for backup: {model_path}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Model backup failed: {e}")
            return None
    
    def simulate_retraining(self, model_type: str, reason: str) -> Dict:
        """Simulate model retraining (placeholder for actual retraining)"""
        try:
            logging.info(f"🔄 Starting simulated retraining for {model_type}")
            logging.info(f"   Reason: {reason}")
            
            # Simulate retraining process
            import time
            time.sleep(2)  # Simulate training time
            
            # Simulate new model performance
            import random
            new_accuracy = random.uniform(0.85, 0.95)
            new_f1_score = random.uniform(0.80, 0.92)
            
            # Record retraining
            retrain_record = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'reason': reason,
                'pre_retrain_performance': 'simulated',
                'post_retrain_performance': {
                    'accuracy': new_accuracy,
                    'f1_score': new_f1_score
                },
                'status': 'success',
                'training_time_seconds': 2
            }
            
            self.retraining_history.append(retrain_record)
            
            logging.info(f"✅ Retraining completed: New accuracy {new_accuracy:.3f}")
            return retrain_record
            
        except Exception as e:
            logging.error(f"❌ Retraining failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'reason': reason,
                'status': 'failed',
                'error': str(e)
            }
    
    def get_retraining_summary(self) -> Dict:
        """Get summary of retraining activities"""
        return {
            'total_retrainings': len(self.retraining_history),
            'recent_retrainings': [r for r in self.retraining_history 
                                 if datetime.fromisoformat(r['timestamp']) > 
                                 datetime.now() - timedelta(days=30)],
            'model_backups': {k: len(v) for k, v in self.model_backups.items()},
            'config': asdict(self.config)
        }

class AdaptiveLearningSystem:
    """Main adaptive learning coordination system"""
    
    def __init__(self, config: RetrainingConfig = None):
        self.config = config or RetrainingConfig()
        self.performance_tracker = ModelPerformanceTracker()
        self.drift_detector = DriftDetector()
        self.retrainer = AdaptiveRetrainer(self.config)
        self.is_monitoring = False
        
        # Supported models (simulated for now)
        self.supported_models = [
            ModelType.LSTM,
            ModelType.XGBOOST, 
            ModelType.RANDOM_FOREST,
            ModelType.ENSEMBLE
        ]
    
    async def start_adaptive_learning(self):
        """Start the adaptive learning monitoring loop"""
        try:
            logging.info("🧠 Starting Adaptive Learning System...")
            self.is_monitoring = True
            
            monitoring_tasks = []
            
            # Task 1: Performance monitoring
            monitoring_tasks.append(
                asyncio.create_task(self._performance_monitoring_loop())
            )
            
            # Task 2: Drift detection
            monitoring_tasks.append(
                asyncio.create_task(self._drift_detection_loop())
            )
            
            # Task 3: Retraining coordination
            monitoring_tasks.append(
                asyncio.create_task(self._retraining_coordination_loop())
            )
            
            # Task 4: Health monitoring
            monitoring_tasks.append(
                asyncio.create_task(self._system_health_loop())
            )
            
            # Run all monitoring tasks
            await asyncio.gather(*monitoring_tasks)
            
        except Exception as e:
            logging.error(f"❌ Adaptive learning system failed: {e}")
        finally:
            self.is_monitoring = False
            logging.info("🛑 Adaptive learning system stopped")
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        while self.is_monitoring:
            try:
                for model_type in self.supported_models:
                    # Simulate performance metrics
                    metrics = self._generate_simulated_metrics(model_type.value)
                    self.performance_tracker.record_performance(metrics)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logging.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _drift_detection_loop(self):
        """Continuous drift detection"""
        while self.is_monitoring:
            try:
                # Simulate drift detection
                for model_type in self.supported_models:
                    # Generate synthetic data for drift detection
                    synthetic_data = np.random.normal(0, 1, (100, 10))
                    feature_names = [f"feature_{i}" for i in range(10)]
                    
                    # Update baseline occasionally
                    if np.random.random() < 0.1:  # 10% chance
                        self.drift_detector.update_baseline(synthetic_data, feature_names)
                    
                    # Detect drift
                    drift_result = self.drift_detector.detect_data_drift(
                        synthetic_data, feature_names
                    )
                    
                    if drift_result.drift_detected:
                        logging.warning(f"⚠️ Drift detected in {model_type.value}: "
                                      f"{drift_result.drift_magnitude:.3f}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logging.error(f"❌ Drift detection error: {e}")
                await asyncio.sleep(300)
    
    async def _retraining_coordination_loop(self):
        """Coordinate retraining decisions"""
        while self.is_monitoring:
            try:
                for model_type in self.supported_models:
                    should_retrain, reason = self.retrainer.should_retrain(
                        self.performance_tracker, 
                        self.drift_detector, 
                        model_type.value
                    )
                    
                    if should_retrain:
                        logging.info(f"🔄 Triggering retraining for {model_type.value}: {reason}")
                        
                        # Backup current model
                        backup_path = self.retrainer.backup_model(
                            model_type.value, 
                            f"models/{model_type.value}_model"
                        )
                        
                        # Simulate retraining
                        retrain_result = self.retrainer.simulate_retraining(
                            model_type.value, reason
                        )
                        
                        if retrain_result['status'] == 'success':
                            logging.info(f"✅ Retraining successful for {model_type.value}")
                        else:
                            logging.error(f"❌ Retraining failed for {model_type.value}")
                
                await asyncio.sleep(7200)  # Check every 2 hours
                
            except Exception as e:
                logging.error(f"❌ Retraining coordination error: {e}")
                await asyncio.sleep(600)
    
    async def _system_health_loop(self):
        """Monitor adaptive learning system health"""
        while self.is_monitoring:
            try:
                # System health checks
                health_status = {
                    'performance_tracker_health': len(self.performance_tracker.performance_history) > 0,
                    'drift_detector_health': len(self.drift_detector.baseline_stats) > 0,
                    'retrainer_health': len(self.retrainer.retraining_history) >= 0,
                    'uptime_hours': 1  # Simplified for demo
                }
                
                logging.info(f"💚 Adaptive Learning Health: "
                           f"Tracker: {health_status['performance_tracker_health']}, "
                           f"Drift: {health_status['drift_detector_health']}, "
                           f"Retrainer: {health_status['retrainer_health']}")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logging.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(300)
    
    def _generate_simulated_metrics(self, model_type: str) -> ModelPerformanceMetrics:
        """Generate simulated performance metrics"""
        import random
        
        # Simulate gradual performance decay
        base_accuracy = 0.85
        time_decay = random.uniform(-0.02, 0.01)  # Slight decay over time
        noise = random.uniform(-0.05, 0.05)
        
        accuracy = max(0.5, min(0.99, base_accuracy + time_decay + noise))
        
        return ModelPerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            model_type=model_type,
            accuracy=accuracy,
            precision=accuracy * random.uniform(0.95, 1.05),
            recall=accuracy * random.uniform(0.95, 1.05),
            f1_score=accuracy * random.uniform(0.95, 1.05),
            prediction_confidence=random.uniform(0.6, 0.9),
            prediction_count=random.randint(50, 200),
            correct_predictions=int(accuracy * random.randint(50, 200)),
            profit_attribution=random.uniform(-0.02, 0.08),
            sharpe_ratio=random.uniform(0.5, 2.5),
            max_drawdown=random.uniform(0.02, 0.15)
        )
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary"""
        summary = {
            'status': 'ACTIVE' if self.is_monitoring else 'STOPPED',
            'config': asdict(self.config),
            'performance_tracking': {
                'models_tracked': len(self.performance_tracker.performance_history),
                'total_metrics': sum(len(metrics) for metrics in 
                                   self.performance_tracker.performance_history.values())
            },
            'drift_detection': {
                'baseline_established': len(self.drift_detector.baseline_stats) > 0,
                'features_monitored': len(self.drift_detector.baseline_stats.get('feature_stats', {}))
            },
            'retraining': self.retrainer.get_retraining_summary(),
            'supported_models': [model.value for model in self.supported_models]
        }
        
        return summary

def main():
    """Adaptive Learning System Demo"""
    print("🧠 PRIORITY 3: ADAPTIVE LEARNING LOOP")
    print("=" * 80)
    
    # Create configuration
    config = RetrainingConfig(
        retrain_frequency_hours=24,  # Daily for demo
        min_performance_threshold=0.70,
        drift_threshold=0.10,
        enable_online_learning=True
    )
    
    # Initialize system
    adaptive_system = AdaptiveLearningSystem(config)
    
    # Demo menu
    while True:
        print(f"\n🧠 ADAPTIVE LEARNING SYSTEM")
        print(f"Retrain Frequency: {config.retrain_frequency_hours}h")
        print(f"Performance Threshold: {config.min_performance_threshold:.1%}")
        print(f"Drift Threshold: {config.drift_threshold:.1%}")
        
        print(f"\nOptions:")
        print(f"1. Generate performance metrics")
        print(f"2. Test drift detection")
        print(f"3. Simulate retraining decision")
        print(f"4. Start adaptive monitoring (demo)")
        print(f"5. View system summary")
        print(f"6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            print(f"\n📊 Generating performance metrics...")
            for model_type in adaptive_system.supported_models:
                metrics = adaptive_system._generate_simulated_metrics(model_type.value)
                adaptive_system.performance_tracker.record_performance(metrics)
                print(f"   {model_type.value}: Accuracy {metrics.accuracy:.3f}, "
                     f"F1 {metrics.f1_score:.3f}")
        
        elif choice == '2':
            print(f"\n🔍 Testing drift detection...")
            synthetic_data = np.random.normal(0, 1, (100, 5))
            feature_names = [f"feature_{i}" for i in range(5)]
            
            # Update baseline
            adaptive_system.drift_detector.update_baseline(synthetic_data, feature_names)
            
            # Add some drift
            drifted_data = synthetic_data + np.random.normal(0, 0.5, synthetic_data.shape)
            drift_result = adaptive_system.drift_detector.detect_data_drift(
                drifted_data, feature_names
            )
            
            print(f"   Drift Detected: {drift_result.drift_detected}")
            print(f"   Drift Magnitude: {drift_result.drift_magnitude:.3f}")
            print(f"   Recommendation: {drift_result.recommendation}")
        
        elif choice == '3':
            print(f"\n🤔 Testing retraining decision...")
            for model_type in adaptive_system.supported_models:
                should_retrain, reason = adaptive_system.retrainer.should_retrain(
                    adaptive_system.performance_tracker,
                    adaptive_system.drift_detector,
                    model_type.value
                )
                
                print(f"   {model_type.value}: {'RETRAIN' if should_retrain else 'CONTINUE'}")
                if should_retrain:
                    print(f"     Reason: {reason}")
        
        elif choice == '4':
            print(f"\n🔄 Starting adaptive monitoring demo (30 seconds)...")
            async def demo_monitoring():
                # Create a task that runs for 30 seconds
                monitor_task = asyncio.create_task(adaptive_system.start_adaptive_learning())
                await asyncio.sleep(30)
                adaptive_system.is_monitoring = False
                try:
                    await monitor_task
                except:
                    pass
            
            asyncio.run(demo_monitoring())
            print("Demo monitoring completed!")
        
        elif choice == '5':
            summary = adaptive_system.get_system_summary()
            print(f"\n📊 System Summary:")
            for key, value in summary.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"     {sub_key}: {sub_value}")
                else:
                    print(f"   {key}: {value}")
        
        elif choice == '6':
            break
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
