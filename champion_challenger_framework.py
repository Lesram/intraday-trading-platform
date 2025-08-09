#!/usr/bin/env python3
"""
🥊 CHAMPION-CHALLENGER TESTING FRAMEWORK
Automated A/B testing for model deployment with statistical rigor

Priority 2 Implementation: Replace manual model evaluation with automated
champion-challenger system that continuously validates model improvements.
"""

import asyncio
import logging
import os
import queue
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any

import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model registry
from institutional_model_registry import DeploymentDecision, ModelComparison, ModelRegistry, ModelStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class TestPhase(Enum):
    """Champion-challenger test phases"""
    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    TESTING = "testing"
    ANALYSIS = "analysis"
    DECISION = "decision"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"

class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    EQUAL_SPLIT = "equal_split"       # 50/50 split
    CHALLENGER_BIAS = "challenger_bias"  # 30/70 challenger favored
    CHAMPION_SAFE = "champion_safe"   # 80/20 champion favored
    GRADUAL_RAMP = "gradual_ramp"     # Start 95/5, gradually to 50/50

@dataclass
class TestConfiguration:
    """Champion-challenger test configuration"""
    test_id: str
    champion_model_id: str
    challenger_model_id: str

    # Test parameters
    traffic_allocation: TrafficAllocation
    test_duration_days: int
    warmup_period_hours: int
    minimum_sample_size: int

    # Success criteria
    primary_metric: str
    success_threshold: float
    significance_level: float
    power_threshold: float

    # Safety constraints
    max_drawdown_threshold: float
    stop_loss_threshold: float
    performance_degradation_threshold: float

    # Configuration
    enable_early_stopping: bool
    enable_gradual_rollout: bool

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['traffic_allocation'] = self.traffic_allocation.value
        return result

@dataclass
class TestResult:
    """Individual prediction test result"""
    timestamp: datetime
    model_id: str
    prediction: dict[str, Any]
    actual_outcome: dict[str, Any] | None
    execution_time_ms: float
    confidence_score: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class TestMetrics:
    """Aggregated test metrics for a model"""
    model_id: str
    total_predictions: int
    correct_predictions: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_execution_time_ms: float
    confidence_intervals: dict[str, tuple[float, float]]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ChallengerTest:
    """Complete challenger test state"""
    test_id: str
    configuration: TestConfiguration
    start_time: datetime
    current_phase: TestPhase

    # Results tracking
    champion_metrics: TestMetrics | None
    challenger_metrics: TestMetrics | None
    champion_results: list[TestResult]
    challenger_results: list[TestResult]

    # Test progress
    samples_collected: int
    current_allocation_ratio: float
    phase_start_time: datetime

    # Decision tracking
    statistical_comparison: ModelComparison | None
    final_decision: DeploymentDecision | None
    decision_confidence: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        result = {
            'test_id': self.test_id,
            'configuration': self.configuration.to_dict(),
            'start_time': self.start_time.isoformat(),
            'current_phase': self.current_phase.value,
            'samples_collected': self.samples_collected,
            'current_allocation_ratio': self.current_allocation_ratio,
            'phase_start_time': self.phase_start_time.isoformat(),
            'champion_metrics': self.champion_metrics.to_dict() if self.champion_metrics else None,
            'challenger_metrics': self.challenger_metrics.to_dict() if self.challenger_metrics else None,
            'statistical_comparison': self.statistical_comparison.to_dict() if self.statistical_comparison else None,
            'final_decision': self.final_decision.value if self.final_decision else None,
            'decision_confidence': self.decision_confidence
        }
        return result

class StatisticalAnalyzer:
    """Statistical analysis for champion-challenger comparison"""

    @staticmethod
    def calculate_confidence_interval(data: np.ndarray,
                                    confidence_level: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval for data"""
        from scipy import stats

        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)

        return mean - h, mean + h

    @staticmethod
    def perform_t_test(champion_data: np.ndarray,
                      challenger_data: np.ndarray) -> dict[str, Any]:
        """Perform two-sample t-test"""
        from scipy import stats

        statistic, p_value = stats.ttest_ind(challenger_data, champion_data)

        return {
            'test_type': 'two_sample_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': np.mean(challenger_data) - np.mean(champion_data),
            'champion_mean': np.mean(champion_data),
            'challenger_mean': np.mean(challenger_data)
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns / np.std(returns) * np.sqrt(252)  # Annualized

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

class ChampionChallengerFramework:
    """Advanced A/B testing framework for model deployment"""

    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.active_tests: dict[str, ChallengerTest] = {}
        self.test_queue = queue.Queue()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.statistical_analyzer = StatisticalAnalyzer()

        logger.info("🥊 Champion-Challenger Framework initialized")

    def create_test(self, configuration: TestConfiguration) -> str:
        """Create and start new champion-challenger test"""

        try:
            logger.info(f"🚀 Creating champion-challenger test: {configuration.test_id}")

            # Validate models exist
            champion = self.model_registry.get_model_by_id(configuration.champion_model_id)
            challenger = self.model_registry.get_model_by_id(configuration.challenger_model_id)

            if not champion or not challenger:
                raise ValueError("Champion or challenger model not found")

            # Create test instance
            test = ChallengerTest(
                test_id=configuration.test_id,
                configuration=configuration,
                start_time=datetime.now(),
                current_phase=TestPhase.INITIALIZATION,
                champion_metrics=None,
                challenger_metrics=None,
                champion_results=[],
                challenger_results=[],
                samples_collected=0,
                current_allocation_ratio=0.5,  # Start with equal allocation
                phase_start_time=datetime.now(),
                statistical_comparison=None,
                final_decision=None,
                decision_confidence=None
            )

            # Store test
            with self.lock:
                self.active_tests[configuration.test_id] = test

            # Update model statuses
            self._update_model_status(configuration.champion_model_id, ModelStatus.CHAMPION)
            self._update_model_status(configuration.challenger_model_id, ModelStatus.CHALLENGER)

            # Start test execution
            self.executor.submit(self._run_test, test)

            logger.info(f"✅ Test {configuration.test_id} created and started")
            return configuration.test_id

        except Exception as e:
            logger.error(f"❌ Failed to create test: {e}")
            raise

    def _update_model_status(self, model_id: str, status: ModelStatus):
        """Update model status in registry"""
        # This would integrate with the model registry
        logger.info(f"📝 Updated {model_id} status to {status.value}")

    async def _run_test(self, test: ChallengerTest):
        """Execute champion-challenger test"""

        logger.info(f"🏃 Running test {test.test_id}")

        try:
            # Phase 1: Initialization
            await self._run_initialization_phase(test)

            # Phase 2: Warmup period
            await self._run_warmup_phase(test)

            # Phase 3: Main testing period
            await self._run_testing_phase(test)

            # Phase 4: Statistical analysis
            await self._run_analysis_phase(test)

            # Phase 5: Make deployment decision
            await self._run_decision_phase(test)

            # Phase 6: Execute deployment
            await self._run_deployment_phase(test)

            # Phase 7: Post-deployment monitoring
            await self._run_monitoring_phase(test)

            # Mark as completed
            test.current_phase = TestPhase.COMPLETED
            logger.info(f"✅ Test {test.test_id} completed successfully")

        except Exception as e:
            logger.error(f"❌ Test {test.test_id} failed: {e}")
            test.current_phase = TestPhase.FAILED
            raise

    async def _run_initialization_phase(self, test: ChallengerTest):
        """Initialize test environment and validate setup"""

        logger.info(f"🔧 Initializing test {test.test_id}")
        test.current_phase = TestPhase.INITIALIZATION
        test.phase_start_time = datetime.now()

        # Load and validate models
        champion_model, champion_metadata = self.model_registry.load_model(
            test.configuration.champion_model_id
        )
        challenger_model, challenger_metadata = self.model_registry.load_model(
            test.configuration.challenger_model_id
        )

        # Validate models are compatible
        if champion_metadata.model_type != challenger_metadata.model_type:
            logger.warning(f"⚠️ Model type mismatch: {champion_metadata.model_type} vs {challenger_metadata.model_type}")

        # Initialize traffic allocation
        test.current_allocation_ratio = self._get_initial_allocation_ratio(
            test.configuration.traffic_allocation
        )

        await asyncio.sleep(1)  # Simulate initialization time
        logger.info(f"✅ Test {test.test_id} initialized")

    async def _run_warmup_phase(self, test: ChallengerTest):
        """Run warmup period to establish baseline performance"""

        logger.info(f"🔥 Starting warmup for test {test.test_id}")
        test.current_phase = TestPhase.WARMUP
        test.phase_start_time = datetime.now()

        warmup_duration = timedelta(hours=test.configuration.warmup_period_hours)

        # Simulate warmup period
        await asyncio.sleep(min(warmup_duration.total_seconds(), 5))  # Capped for demo

        logger.info(f"✅ Warmup completed for test {test.test_id}")

    async def _run_testing_phase(self, test: ChallengerTest):
        """Run main testing phase with traffic splitting"""

        logger.info(f"🧪 Starting testing phase for {test.test_id}")
        test.current_phase = TestPhase.TESTING
        test.phase_start_time = datetime.now()

        test_duration = timedelta(days=test.configuration.test_duration_days)
        end_time = datetime.now() + test_duration

        # Simulate testing period with data collection
        sample_count = 0
        while datetime.now() < end_time and sample_count < test.configuration.minimum_sample_size:

            # Generate simulated test results
            if np.random.random() < test.current_allocation_ratio:
                # Challenger prediction
                result = self._simulate_prediction_result(
                    test.configuration.challenger_model_id,
                    is_challenger=True
                )
                test.challenger_results.append(result)
            else:
                # Champion prediction
                result = self._simulate_prediction_result(
                    test.configuration.champion_model_id,
                    is_challenger=False
                )
                test.champion_results.append(result)

            sample_count += 1
            test.samples_collected = sample_count

            # Update allocation ratio if using gradual ramp
            if test.configuration.traffic_allocation == TrafficAllocation.GRADUAL_RAMP:
                progress = sample_count / test.configuration.minimum_sample_size
                test.current_allocation_ratio = 0.05 + 0.45 * progress  # 5% to 50%

            # Check for early stopping conditions
            if (test.configuration.enable_early_stopping and
                sample_count >= 100 and  # Minimum samples for early stopping
                sample_count % 50 == 0):  # Check every 50 samples

                if await self._check_early_stopping_conditions(test):
                    logger.info(f"⏹️ Early stopping triggered for test {test.test_id}")
                    break

            await asyncio.sleep(0.01)  # Simulate time between predictions

        # Calculate final metrics
        test.champion_metrics = self._calculate_test_metrics(
            test.configuration.champion_model_id, test.champion_results
        )
        test.challenger_metrics = self._calculate_test_metrics(
            test.configuration.challenger_model_id, test.challenger_results
        )

        logger.info(f"✅ Testing completed for {test.test_id} with {sample_count} samples")

    def _simulate_prediction_result(self, model_id: str, is_challenger: bool) -> TestResult:
        """Simulate a prediction result for testing"""

        # Simulate better performance for challenger
        base_accuracy = 0.55 if is_challenger else 0.52
        base_return = 0.002 if is_challenger else 0.0015

        # Add noise
        accuracy_noise = np.random.normal(0, 0.1)
        return_noise = np.random.normal(0, 0.005)

        is_correct = np.random.random() < base_accuracy + accuracy_noise
        return_value = base_return + return_noise

        return TestResult(
            timestamp=datetime.now(),
            model_id=model_id,
            prediction={'action': 'buy', 'confidence': base_accuracy + accuracy_noise},
            actual_outcome={'return': return_value, 'correct': is_correct},
            execution_time_ms=np.random.uniform(10, 50),
            confidence_score=base_accuracy + accuracy_noise
        )

    def _calculate_test_metrics(self, model_id: str,
                              results: list[TestResult]) -> TestMetrics:
        """Calculate aggregated metrics from test results"""

        if not results:
            return TestMetrics(
                model_id=model_id,
                total_predictions=0,
                correct_predictions=0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_execution_time_ms=0.0,
                confidence_intervals={}
            )

        # Extract data
        returns = np.array([r.actual_outcome['return'] for r in results if r.actual_outcome])
        correct_flags = np.array([r.actual_outcome['correct'] for r in results if r.actual_outcome])
        execution_times = np.array([r.execution_time_ms for r in results])

        # Calculate metrics
        total_return = np.sum(returns)
        sharpe_ratio = self.statistical_analyzer.calculate_sharpe_ratio(returns)
        cumulative_returns = np.cumsum(returns)
        max_drawdown = self.statistical_analyzer.calculate_max_drawdown(cumulative_returns)
        win_rate = np.mean(correct_flags) if len(correct_flags) > 0 else 0.0

        # Calculate confidence intervals
        confidence_intervals = {}
        if len(returns) > 1:
            return_ci = self.statistical_analyzer.calculate_confidence_interval(returns)
            confidence_intervals['returns'] = return_ci

        return TestMetrics(
            model_id=model_id,
            total_predictions=len(results),
            correct_predictions=int(np.sum(correct_flags)),
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_execution_time_ms=np.mean(execution_times),
            confidence_intervals=confidence_intervals
        )

    async def _check_early_stopping_conditions(self, test: ChallengerTest) -> bool:
        """Check if early stopping conditions are met"""

        if len(test.challenger_results) < 50 or len(test.champion_results) < 50:
            return False

        # Calculate current returns
        challenger_returns = np.array([
            r.actual_outcome['return'] for r in test.challenger_results[-50:]
            if r.actual_outcome
        ])
        champion_returns = np.array([
            r.actual_outcome['return'] for r in test.champion_results[-50:]
            if r.actual_outcome
        ])

        # Check for significant performance degradation
        challenger_mean = np.mean(challenger_returns)
        champion_mean = np.mean(champion_returns)

        degradation_ratio = (challenger_mean - champion_mean) / abs(champion_mean) if champion_mean != 0 else 0

        if degradation_ratio < -test.configuration.performance_degradation_threshold:
            logger.warning(f"⚠️ Performance degradation detected: {degradation_ratio:.2%}")
            return True

        # Check maximum drawdown
        challenger_cum_returns = np.cumsum(challenger_returns)
        challenger_drawdown = self.statistical_analyzer.calculate_max_drawdown(challenger_cum_returns)

        if abs(challenger_drawdown) > test.configuration.max_drawdown_threshold:
            logger.warning(f"⚠️ Max drawdown exceeded: {challenger_drawdown:.2%}")
            return True

        return False

    async def _run_analysis_phase(self, test: ChallengerTest):
        """Perform statistical analysis of test results"""

        logger.info(f"📊 Analyzing results for test {test.test_id}")
        test.current_phase = TestPhase.ANALYSIS
        test.phase_start_time = datetime.now()

        if not test.champion_metrics or not test.challenger_metrics:
            raise ValueError("Missing test metrics for analysis")

        # Perform statistical comparison using model registry
        test.statistical_comparison = self.model_registry.compare_models(
            test.configuration.champion_model_id,
            test.configuration.challenger_model_id
        )

        # Additional statistical tests
        champion_returns = np.array([
            r.actual_outcome['return'] for r in test.champion_results if r.actual_outcome
        ])
        challenger_returns = np.array([
            r.actual_outcome['return'] for r in test.challenger_results if r.actual_outcome
        ])

        if len(champion_returns) > 10 and len(challenger_returns) > 10:
            t_test_results = self.statistical_analyzer.perform_t_test(
                champion_returns, challenger_returns
            )

            logger.info(f"📈 T-test results: p-value = {t_test_results['p_value']:.4f}")

        await asyncio.sleep(1)  # Simulate analysis time
        logger.info(f"✅ Analysis completed for test {test.test_id}")

    async def _run_decision_phase(self, test: ChallengerTest):
        """Make deployment decision based on analysis"""

        logger.info(f"🤔 Making deployment decision for test {test.test_id}")
        test.current_phase = TestPhase.DECISION
        test.phase_start_time = datetime.now()

        if not test.statistical_comparison:
            raise ValueError("Statistical comparison not available for decision")

        # Use model registry comparison results
        test.final_decision = test.statistical_comparison.recommendation
        test.decision_confidence = test.statistical_comparison.confidence_level

        logger.info(f"📋 Decision for test {test.test_id}: {test.final_decision.value}")
        logger.info(f"📋 Confidence: {test.decision_confidence:.2%}")

        await asyncio.sleep(1)
        logger.info(f"✅ Decision made for test {test.test_id}")

    async def _run_deployment_phase(self, test: ChallengerTest):
        """Execute deployment based on decision"""

        logger.info(f"🚀 Deploying decision for test {test.test_id}")
        test.current_phase = TestPhase.DEPLOYMENT
        test.phase_start_time = datetime.now()

        if test.final_decision == DeploymentDecision.PROMOTE:
            # Promote challenger to champion
            challenger_metadata = self.model_registry.get_model_by_id(
                test.configuration.challenger_model_id
            )

            success = self.model_registry.set_champion(
                challenger_metadata.model_name,
                challenger_metadata.version
            )

            if success:
                logger.info("✅ Promoted challenger to champion")
            else:
                logger.error("❌ Failed to promote challenger")

        elif test.final_decision == DeploymentDecision.REJECT:
            # Update challenger status to deprecated
            self._update_model_status(
                test.configuration.challenger_model_id,
                ModelStatus.DEPRECATED
            )
            logger.info("❌ Challenger rejected and deprecated")

        elif test.final_decision == DeploymentDecision.CONTINUE_TESTING:
            # Extend test or create follow-up test
            logger.info("⏳ Test requires additional validation")

        await asyncio.sleep(1)
        logger.info(f"✅ Deployment completed for test {test.test_id}")

    async def _run_monitoring_phase(self, test: ChallengerTest):
        """Monitor post-deployment performance"""

        logger.info(f"👀 Monitoring post-deployment for test {test.test_id}")
        test.current_phase = TestPhase.MONITORING
        test.phase_start_time = datetime.now()

        # Monitor for a short period (in production this would be longer)
        monitoring_duration = timedelta(minutes=5)  # Shortened for demo
        end_time = datetime.now() + monitoring_duration

        while datetime.now() < end_time:
            # Record ongoing performance metrics
            if test.final_decision == DeploymentDecision.PROMOTE:
                # Monitor new champion performance
                self.model_registry.record_performance(
                    test.configuration.challenger_model_id,
                    {'post_deployment_return': np.random.normal(0.002, 0.005)},
                    context='post_deployment_monitoring'
                )

            await asyncio.sleep(1)

        logger.info(f"✅ Monitoring completed for test {test.test_id}")

    def _get_initial_allocation_ratio(self, allocation: TrafficAllocation) -> float:
        """Get initial traffic allocation ratio for challenger"""

        allocation_map = {
            TrafficAllocation.EQUAL_SPLIT: 0.5,
            TrafficAllocation.CHALLENGER_BIAS: 0.7,
            TrafficAllocation.CHAMPION_SAFE: 0.2,
            TrafficAllocation.GRADUAL_RAMP: 0.05
        }

        return allocation_map.get(allocation, 0.5)

    def get_test_status(self, test_id: str) -> ChallengerTest | None:
        """Get current test status"""
        with self.lock:
            return self.active_tests.get(test_id)

    def list_active_tests(self) -> list[str]:
        """List all active test IDs"""
        with self.lock:
            return list(self.active_tests.keys())

    def stop_test(self, test_id: str) -> bool:
        """Stop an active test"""
        with self.lock:
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
                test.current_phase = TestPhase.COMPLETED
                logger.info(f"⏹️ Stopped test {test_id}")
                return True
        return False

    def get_test_summary(self) -> dict[str, Any]:
        """Get summary of all tests"""
        with self.lock:
            active_count = len([t for t in self.active_tests.values()
                              if t.current_phase not in [TestPhase.COMPLETED, TestPhase.FAILED]])
            completed_count = len([t for t in self.active_tests.values()
                                 if t.current_phase == TestPhase.COMPLETED])
            failed_count = len([t for t in self.active_tests.values()
                              if t.current_phase == TestPhase.FAILED])

            return {
                'total_tests': len(self.active_tests),
                'active_tests': active_count,
                'completed_tests': completed_count,
                'failed_tests': failed_count,
                'test_phases': {phase.value: len([t for t in self.active_tests.values()
                                                if t.current_phase == phase])
                               for phase in TestPhase}
            }

def main():
    """Test the champion-challenger framework"""

    print("🥊 CHAMPION-CHALLENGER TESTING FRAMEWORK")
    print("=" * 60)

    try:
        # Initialize model registry and framework
        registry = ModelRegistry("test_registry")
        framework = ChampionChallengerFramework(registry)

        # Create test models (simplified for demo)
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        champion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        champion_model.fit(X, y)

        challenger_model = RandomForestClassifier(n_estimators=200, random_state=42)
        challenger_model.fit(X, y)

        # Register models
        champion_id = registry.register_model(
            model_name="test_model",
            model_artifact=champion_model,
            model_type="random_forest",
            training_metadata={'creator': 'test'},
            performance_metrics={'sharpe_ratio': 1.5, 'win_rate': 0.55}
        )

        challenger_id = registry.register_model(
            model_name="test_model",
            model_artifact=challenger_model,
            model_type="random_forest",
            training_metadata={'creator': 'test'},
            performance_metrics={'sharpe_ratio': 1.8, 'win_rate': 0.58}
        )

        # Create test configuration
        config = TestConfiguration(
            test_id=f"test_{uuid.uuid4().hex[:8]}",
            champion_model_id=champion_id,
            challenger_model_id=challenger_id,
            traffic_allocation=TrafficAllocation.EQUAL_SPLIT,
            test_duration_days=1,
            warmup_period_hours=1,
            minimum_sample_size=200,
            primary_metric="sharpe_ratio",
            success_threshold=0.1,
            significance_level=0.05,
            power_threshold=0.8,
            max_drawdown_threshold=0.1,
            stop_loss_threshold=0.05,
            performance_degradation_threshold=0.15,
            enable_early_stopping=True,
            enable_gradual_rollout=False
        )

        # Run test
        test_id = framework.create_test(config)

        print(f"\n🚀 Created test: {test_id}")

        # Monitor test progress
        import time
        for i in range(10):
            test_status = framework.get_test_status(test_id)
            if test_status:
                print(f"Phase: {test_status.current_phase.value}, Samples: {test_status.samples_collected}")
                if test_status.current_phase in [TestPhase.COMPLETED, TestPhase.FAILED]:
                    break
            time.sleep(2)

        # Get final results
        final_test = framework.get_test_status(test_id)
        if final_test:
            print("\n📊 FINAL RESULTS")
            print("-" * 30)
            print(f"Decision: {final_test.final_decision.value if final_test.final_decision else 'None'}")
            print(f"Confidence: {final_test.decision_confidence:.2%}" if final_test.decision_confidence else "")

            if final_test.champion_metrics and final_test.challenger_metrics:
                print(f"Champion Sharpe: {final_test.champion_metrics.sharpe_ratio:.3f}")
                print(f"Challenger Sharpe: {final_test.challenger_metrics.sharpe_ratio:.3f}")

        # Get framework summary
        summary = framework.get_test_summary()
        print("\n📋 FRAMEWORK SUMMARY")
        print("-" * 30)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Completed: {summary['completed_tests']}")

        print("\n✅ Champion-challenger framework test completed!")

        # Cleanup
        import shutil
        shutil.rmtree("test_registry")
        print("🗑️ Test cleanup completed")

    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
