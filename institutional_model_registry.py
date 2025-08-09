#!/usr/bin/env python3
"""
🗄️ INSTITUTIONAL MODEL REGISTRY
Professional model lifecycle management with versioning and champion-challenger testing

Priority 2 Implementation: Transform from ad-hoc model files to enterprise-grade
model management with automated validation and deployment.
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    REGISTERED = "registered"
    TESTING = "testing"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class DeploymentDecision(Enum):
    """Deployment decision types"""
    PROMOTE = "promote"
    REJECT = "reject"
    CONTINUE_TESTING = "continue_testing"
    ROLLBACK = "rollback"

@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    model_name: str
    version: str
    model_type: str
    status: ModelStatus
    created_at: datetime
    training_data_period: dict[str, str]
    training_parameters: dict[str, Any]
    performance_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    feature_importance: dict[str, float]
    model_size_mb: float
    training_duration_seconds: float
    artifact_path: str
    checksum: str
    creator: str
    notes: str

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        return result

@dataclass
class ModelComparison:
    """Model comparison results"""
    champion_id: str
    challenger_id: str
    comparison_metrics: dict[str, dict[str, float]]  # metric -> {champion: val, challenger: val}
    statistical_tests: dict[str, dict[str, Any]]      # test -> results
    recommendation: DeploymentDecision
    confidence_level: float
    summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        result = asdict(self)
        result['recommendation'] = self.recommendation.value
        return result

@dataclass
class DeploymentPlan:
    """Champion-challenger deployment plan"""
    plan_id: str
    champion_model: ModelMetadata
    challenger_model: ModelMetadata
    start_date: datetime
    validation_period_days: int
    success_criteria: dict[str, float]
    current_metrics: dict[str, float]
    status: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['start_date'] = self.start_date.isoformat()
        result['champion_model'] = self.champion_model.to_dict()
        result['challenger_model'] = self.challenger_model.to_dict()
        return result

class ModelRegistry:
    """Enterprise-grade model lifecycle management"""

    def __init__(self, storage_path: str = "model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.storage_path / "model_registry.db"
        self._init_database()

        # Initialize storage directories
        self.artifacts_path = self.storage_path / "artifacts"
        self.backups_path = self.storage_path / "backups"
        self.artifacts_path.mkdir(exist_ok=True)
        self.backups_path.mkdir(exist_ok=True)

        logger.info(f"✅ Model Registry initialized at {self.storage_path}")

    def _init_database(self):
        """Initialize SQLite database for metadata"""

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    training_data_period TEXT NOT NULL,
                    training_parameters TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    validation_metrics TEXT NOT NULL,
                    feature_importance TEXT NOT NULL,
                    model_size_mb REAL NOT NULL,
                    training_duration_seconds REAL NOT NULL,
                    artifact_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    notes TEXT,
                    UNIQUE(model_name, version)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    plan_id TEXT PRIMARY KEY,
                    champion_id TEXT NOT NULL,
                    challenger_id TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    validation_period_days INTEGER NOT NULL,
                    success_criteria TEXT NOT NULL,
                    current_metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (champion_id) REFERENCES models (model_id),
                    FOREIGN KEY (challenger_id) REFERENCES models (model_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')

            conn.commit()

    def register_model(self,
                      model_name: str,
                      model_artifact: Any,
                      model_type: str,
                      training_metadata: dict,
                      performance_metrics: dict,
                      validation_metrics: dict | None = None,
                      notes: str = "") -> str:
        """Register new model version"""

        try:
            logger.info(f"📝 Registering model: {model_name}")

            # Generate version ID and model ID
            version = self._generate_version_id(model_name)
            model_id = f"{model_name}_{version}"

            # Save model artifact
            artifact_path, checksum, size_mb = self._save_model_artifact(
                model_artifact, model_name, version
            )

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                model_type=model_type,
                status=ModelStatus.REGISTERED,
                created_at=datetime.now(),
                training_data_period=training_metadata.get('data_period', {}),
                training_parameters=training_metadata.get('parameters', {}),
                performance_metrics=performance_metrics,
                validation_metrics=validation_metrics or {},
                feature_importance=training_metadata.get('feature_importance', {}),
                model_size_mb=size_mb,
                training_duration_seconds=training_metadata.get('training_time', 0),
                artifact_path=str(artifact_path),
                checksum=checksum,
                creator=training_metadata.get('creator', 'system'),
                notes=notes
            )

            # Store in database
            self._store_model_metadata(metadata)

            logger.info(f"✅ Registered {model_name} v{version} as {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"❌ Model registration failed: {e}")
            raise

    def _generate_version_id(self, model_name: str) -> str:
        """Generate semantic version ID"""

        try:
            # Get existing versions for this model
            versions = self.get_model_versions(model_name)

            if not versions:
                return "1.0.0"

            # Parse latest version and increment
            latest_version = max(versions, key=lambda x: self._parse_version(x.version))
            major, minor, patch = self._parse_version(latest_version.version)

            # Increment patch version
            return f"{major}.{minor}.{patch + 1}"

        except Exception as e:
            logger.warning(f"Version generation failed: {e}")
            return f"1.0.{int(datetime.now().timestamp())}"

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string"""
        try:
            parts = version.split('.')
            return int(parts[0]), int(parts[1]), int(parts[2])
        except:
            return 1, 0, 0

    def _save_model_artifact(self, model_artifact: Any,
                           model_name: str, version: str) -> tuple[Path, str, float]:
        """Save model artifact and return path, checksum, and size"""

        # Create versioned directory
        model_dir = self.artifacts_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension based on model type
        if hasattr(model_artifact, 'save'):  # TensorFlow/Keras model
            model_path = model_dir / "model.h5"
            model_artifact.save(str(model_path))
        elif hasattr(model_artifact, 'predict'):  # Scikit-learn model
            model_path = model_dir / "model.pkl"
            joblib.dump(model_artifact, str(model_path))
        else:
            # Generic pickle
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_artifact, f)

        # Calculate checksum and size
        checksum = self._calculate_checksum(model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)

        return model_path, checksum, size_mb

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT INTO models (
                    model_id, model_name, version, model_type, status,
                    created_at, training_data_period, training_parameters,
                    performance_metrics, validation_metrics, feature_importance,
                    model_size_mb, training_duration_seconds, artifact_path,
                    checksum, creator, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id,
                metadata.model_name,
                metadata.version,
                metadata.model_type,
                metadata.status.value,
                metadata.created_at.isoformat(),
                json.dumps(metadata.training_data_period),
                json.dumps(metadata.training_parameters),
                json.dumps(metadata.performance_metrics),
                json.dumps(metadata.validation_metrics),
                json.dumps(metadata.feature_importance),
                metadata.model_size_mb,
                metadata.training_duration_seconds,
                metadata.artifact_path,
                metadata.checksum,
                metadata.creator,
                metadata.notes
            ))
            conn.commit()

    def get_model_versions(self, model_name: str) -> list[ModelMetadata]:
        """Get all versions of a model"""

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT * FROM models WHERE model_name = ? ORDER BY created_at DESC',
                (model_name,)
            )

            versions = []
            for row in cursor.fetchall():
                metadata = self._row_to_metadata(row)
                versions.append(metadata)

            return versions

    def get_model_by_id(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata by ID"""

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT * FROM models WHERE model_id = ?',
                (model_id,)
            )

            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
            return None

    def get_current_champion(self, model_name: str) -> ModelMetadata | None:
        """Get current champion model"""

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT * FROM models WHERE model_name = ? AND status = ? ORDER BY created_at DESC LIMIT 1',
                (model_name, ModelStatus.CHAMPION.value)
            )

            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
            return None

    def get_best_model(self, model_name: str,
                      metric: str = "sharpe_ratio") -> ModelMetadata | None:
        """Get best performing model version"""

        versions = self.get_model_versions(model_name)
        if not versions:
            return None

        # Find best model based on metric
        best_model = None
        best_score = float('-inf')

        for model in versions:
            score = model.performance_metrics.get(metric, float('-inf'))
            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    def load_model(self, model_id: str) -> tuple[Any, ModelMetadata]:
        """Load model artifact and metadata"""

        metadata = self.get_model_by_id(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        # Load model artifact
        artifact_path = Path(metadata.artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        # Verify checksum
        current_checksum = self._calculate_checksum(artifact_path)
        if current_checksum != metadata.checksum:
            logger.warning(f"⚠️ Checksum mismatch for {model_id}")

        # Load based on file extension
        if artifact_path.suffix == '.h5':
            # TensorFlow/Keras model
            try:
                from tensorflow.keras.models import load_model
                model = load_model(str(artifact_path))
            except ImportError:
                raise ImportError("TensorFlow not available for loading .h5 model")
        else:
            # Pickle/Joblib model
            if artifact_path.name == 'model.pkl':
                try:
                    model = joblib.load(str(artifact_path))
                except:
                    with open(artifact_path, 'rb') as f:
                        model = pickle.load(f)
            else:
                with open(artifact_path, 'rb') as f:
                    model = pickle.load(f)

        return model, metadata

    def set_champion(self, model_name: str, version: str) -> bool:
        """Set a specific version as champion"""

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Demote current champion
                conn.execute(
                    'UPDATE models SET status = ? WHERE model_name = ? AND status = ?',
                    (ModelStatus.DEPRECATED.value, model_name, ModelStatus.CHAMPION.value)
                )

                # Promote new champion
                result = conn.execute(
                    'UPDATE models SET status = ? WHERE model_name = ? AND version = ?',
                    (ModelStatus.CHAMPION.value, model_name, version)
                )

                conn.commit()

                if result.rowcount > 0:
                    logger.info(f"✅ Promoted {model_name} v{version} to champion")
                    return True
                else:
                    logger.warning(f"⚠️ Model {model_name} v{version} not found")
                    return False

        except Exception as e:
            logger.error(f"❌ Champion promotion failed: {e}")
            return False

    def compare_models(self, champion_id: str, challenger_id: str) -> ModelComparison:
        """Statistical comparison of model versions"""

        champion = self.get_model_by_id(champion_id)
        challenger = self.get_model_by_id(challenger_id)

        if not champion or not challenger:
            raise ValueError("Invalid model IDs for comparison")

        # Compare performance metrics
        comparison_metrics = {}
        all_metrics = set(champion.performance_metrics.keys()) | set(challenger.performance_metrics.keys())

        for metric in all_metrics:
            champ_val = champion.performance_metrics.get(metric, 0.0)
            chall_val = challenger.performance_metrics.get(metric, 0.0)

            comparison_metrics[metric] = {
                'champion': champ_val,
                'challenger': chall_val,
                'improvement': chall_val - champ_val,
                'improvement_pct': ((chall_val - champ_val) / champ_val * 100) if champ_val != 0 else 0
            }

        # Statistical significance testing (simplified)
        statistical_tests = self._perform_statistical_tests(champion, challenger)

        # Make recommendation
        recommendation, confidence, summary = self._make_deployment_recommendation(
            comparison_metrics, statistical_tests
        )

        return ModelComparison(
            champion_id=champion_id,
            challenger_id=challenger_id,
            comparison_metrics=comparison_metrics,
            statistical_tests=statistical_tests,
            recommendation=recommendation,
            confidence_level=confidence,
            summary=summary
        )

    def _perform_statistical_tests(self, champion: ModelMetadata,
                                 challenger: ModelMetadata) -> dict[str, dict[str, Any]]:
        """Perform statistical significance tests"""

        # Simplified statistical testing
        # In production, this would use actual performance data

        tests = {}

        # Sharpe ratio test
        champ_sharpe = champion.performance_metrics.get('sharpe_ratio', 0)
        chall_sharpe = challenger.performance_metrics.get('sharpe_ratio', 0)

        sharpe_improvement = chall_sharpe - champ_sharpe
        sharpe_significance = abs(sharpe_improvement) > 0.1  # Simplified threshold

        tests['sharpe_ratio'] = {
            'test_type': 'improvement_threshold',
            'improvement': sharpe_improvement,
            'is_significant': sharpe_significance,
            'p_value': 0.05 if sharpe_significance else 0.5,
            'confidence_level': 0.95 if sharpe_significance else 0.5
        }

        # Win rate test
        champ_winrate = champion.performance_metrics.get('win_rate', 0.5)
        chall_winrate = challenger.performance_metrics.get('win_rate', 0.5)

        winrate_improvement = chall_winrate - champ_winrate
        winrate_significance = abs(winrate_improvement) > 0.05

        tests['win_rate'] = {
            'test_type': 'improvement_threshold',
            'improvement': winrate_improvement,
            'is_significant': winrate_significance,
            'p_value': 0.05 if winrate_significance else 0.5,
            'confidence_level': 0.95 if winrate_significance else 0.5
        }

        return tests

    def _make_deployment_recommendation(self, comparison_metrics: dict,
                                      statistical_tests: dict) -> tuple[DeploymentDecision, float, str]:
        """Make deployment recommendation based on comparison"""

        # Check key metrics
        sharpe_improvement = comparison_metrics.get('sharpe_ratio', {}).get('improvement', 0)
        sharpe_significant = statistical_tests.get('sharpe_ratio', {}).get('is_significant', False)

        winrate_improvement = comparison_metrics.get('win_rate', {}).get('improvement', 0)
        winrate_significant = statistical_tests.get('win_rate', {}).get('is_significant', False)

        # Decision logic
        if sharpe_improvement > 0.2 and sharpe_significant:
            return DeploymentDecision.PROMOTE, 0.95, "Significant Sharpe ratio improvement"
        elif sharpe_improvement > 0.1 and winrate_improvement > 0.05:
            return DeploymentDecision.PROMOTE, 0.80, "Moderate improvement in key metrics"
        elif sharpe_improvement < -0.1 or winrate_improvement < -0.05:
            return DeploymentDecision.REJECT, 0.90, "Performance degradation detected"
        else:
            return DeploymentDecision.CONTINUE_TESTING, 0.60, "Insufficient evidence for decision"

    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata object"""

        return ModelMetadata(
            model_id=row[0],
            model_name=row[1],
            version=row[2],
            model_type=row[3],
            status=ModelStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            training_data_period=json.loads(row[6]),
            training_parameters=json.loads(row[7]),
            performance_metrics=json.loads(row[8]),
            validation_metrics=json.loads(row[9]),
            feature_importance=json.loads(row[10]),
            model_size_mb=row[11],
            training_duration_seconds=row[12],
            artifact_path=row[13],
            checksum=row[14],
            creator=row[15],
            notes=row[16] or ""
        )

    def record_performance(self, model_id: str, metrics: dict[str, float],
                         context: str = "live_trading"):
        """Record real-time performance metrics"""

        with sqlite3.connect(str(self.db_path)) as conn:
            timestamp = datetime.now().isoformat()

            for metric_name, metric_value in metrics.items():
                conn.execute(
                    'INSERT INTO model_performance (model_id, timestamp, metric_name, metric_value, context) VALUES (?, ?, ?, ?, ?)',
                    (model_id, timestamp, metric_name, metric_value, context)
                )

            conn.commit()

    def get_model_performance_history(self, model_id: str,
                                    days: int = 30) -> pd.DataFrame:
        """Get performance history for a model"""

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(
                '''
                SELECT timestamp, metric_name, metric_value, context
                FROM model_performance
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                ''',
                conn,
                params=(model_id, cutoff_date)
            )

        return df

    def cleanup_old_models(self, model_name: str, keep_versions: int = 10):
        """Clean up old model versions, keeping only the most recent"""

        versions = self.get_model_versions(model_name)

        if len(versions) <= keep_versions:
            return

        # Keep champion and most recent versions
        to_remove = []
        for model in versions[keep_versions:]:
            if model.status != ModelStatus.CHAMPION:
                to_remove.append(model)

        for model in to_remove:
            self._delete_model_version(model)
            logger.info(f"🗑️ Cleaned up old version {model.model_id}")

    def _delete_model_version(self, model: ModelMetadata):
        """Delete a model version and its artifacts"""

        # Remove artifact files
        artifact_path = Path(model.artifact_path)
        if artifact_path.exists():
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()

        # Remove from database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('DELETE FROM models WHERE model_id = ?', (model.model_id,))
            conn.execute('DELETE FROM model_performance WHERE model_id = ?', (model.model_id,))
            conn.commit()

    def get_registry_summary(self) -> dict[str, Any]:
        """Get summary of model registry"""

        with sqlite3.connect(str(self.db_path)) as conn:
            # Model counts by status
            cursor = conn.execute(
                'SELECT status, COUNT(*) FROM models GROUP BY status'
            )
            status_counts = dict(cursor.fetchall())

            # Model counts by type
            cursor = conn.execute(
                'SELECT model_type, COUNT(*) FROM models GROUP BY model_type'
            )
            type_counts = dict(cursor.fetchall())

            # Model counts by name
            cursor = conn.execute(
                'SELECT model_name, COUNT(*) FROM models GROUP BY model_name'
            )
            name_counts = dict(cursor.fetchall())

            # Total storage size
            cursor = conn.execute('SELECT SUM(model_size_mb) FROM models')
            total_size = cursor.fetchone()[0] or 0

        return {
            'total_models': sum(status_counts.values()),
            'status_breakdown': status_counts,
            'type_breakdown': type_counts,
            'model_families': name_counts,
            'total_storage_mb': total_size,
            'registry_path': str(self.storage_path)
        }

def main():
    """Test the model registry system"""

    print("🗄️ INSTITUTIONAL MODEL REGISTRY")
    print("=" * 60)

    # Initialize registry
    registry = ModelRegistry("test_registry")

    try:
        # Create dummy models for testing
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        # Train two models
        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model1.fit(X, y)

        model2 = RandomForestClassifier(n_estimators=200, random_state=42)
        model2.fit(X, y)

        # Register models
        model1_id = registry.register_model(
            model_name="test_rf",
            model_artifact=model1,
            model_type="random_forest",
            training_metadata={
                'data_period': {'start': '2024-01-01', 'end': '2024-06-30'},
                'parameters': {'n_estimators': 100, 'random_state': 42},
                'creator': 'test_user'
            },
            performance_metrics={
                'accuracy': 0.85,
                'sharpe_ratio': 1.5,
                'win_rate': 0.55
            },
            notes="Initial test model"
        )

        model2_id = registry.register_model(
            model_name="test_rf",
            model_artifact=model2,
            model_type="random_forest",
            training_metadata={
                'data_period': {'start': '2024-01-01', 'end': '2024-06-30'},
                'parameters': {'n_estimators': 200, 'random_state': 42},
                'creator': 'test_user'
            },
            performance_metrics={
                'accuracy': 0.87,
                'sharpe_ratio': 1.8,
                'win_rate': 0.58
            },
            notes="Improved test model"
        )

        # Set champion
        registry.set_champion("test_rf", "1.0.0")

        # Compare models
        comparison = registry.compare_models(model1_id, model2_id)

        print("\n📊 MODEL COMPARISON RESULTS")
        print("-" * 30)
        print(f"Champion: {comparison.champion_id}")
        print(f"Challenger: {comparison.challenger_id}")
        print(f"Recommendation: {comparison.recommendation.value}")
        print(f"Confidence: {comparison.confidence_level:.2%}")
        print(f"Summary: {comparison.summary}")

        print("\n📈 METRIC IMPROVEMENTS")
        print("-" * 30)
        for metric, values in comparison.comparison_metrics.items():
            improvement_pct = values['improvement_pct']
            print(f"{metric}: {improvement_pct:+.1f}%")

        # Registry summary
        summary = registry.get_registry_summary()

        print("\n📋 REGISTRY SUMMARY")
        print("-" * 30)
        print(f"Total models: {summary['total_models']}")
        print(f"Storage used: {summary['total_storage_mb']:.1f} MB")
        print(f"Model families: {list(summary['model_families'].keys())}")

        print("\n✅ Model registry test completed successfully!")

        # Cleanup test registry
        shutil.rmtree("test_registry")
        print("🗑️ Test registry cleaned up")

    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
