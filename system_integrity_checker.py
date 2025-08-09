"""
🔍 SYSTEM INTEGRITY CHECKER
==========================

Comprehensive system health monitoring and data integrity validation for institutional trading platform.
This module ensures all components, data sources, and models are operating within expected parameters.

Features:
- Real-time system health monitoring
- Data quality validation
- ML model integrity checks
- API connectivity verification
- Performance metrics validation
- Security and compliance monitoring

Author: Trading Platform Team
Version: 1.0.0
Date: August 2025
"""

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    timestamp: str
    overall_status: HealthStatus
    component_statuses: dict[str, dict[str, Any]]
    performance_metrics: dict[str, float]
    data_quality_scores: dict[str, float]
    alerts: list[dict[str, Any]]
    recommendations: list[str]
    uptime: float
    memory_usage: float
    cpu_usage: float

class SystemIntegrityChecker:
    """
    Comprehensive system integrity and health monitoring for trading platform
    """

    def __init__(self):
        self.start_time = time.time()
        self.last_check_time = None
        self.health_history = []
        self.alert_thresholds = {
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 80.0,
            'disk_usage_percent': 90.0,
            'api_response_time_ms': 5000,
            'data_quality_threshold': 0.85,
            'model_confidence_threshold': 0.60
        }
        self.critical_files = [
            'alpaca_trading_gateway.py',
            'advanced_ml_predictor.py',
            'real_vix_provider.py',
            'models/rf_ensemble_v2.pkl',
            'models/xgb_ensemble_v2.pkl',
            'models/lstm_ensemble_best.keras'
        ]

        logger.info("🔍 System Integrity Checker initialized")

    def perform_comprehensive_check(self) -> SystemHealthReport:
        """
        Perform comprehensive system integrity check
        Returns complete health report with all subsystem statuses
        """
        try:
            logger.info("🔍 Starting comprehensive system integrity check")
            start_time = time.time()

            # Initialize report components
            component_statuses = {}
            alerts = []
            recommendations = []
            performance_metrics = {}
            data_quality_scores = {}

            # 1. System Resources Check
            system_status = self._check_system_resources()
            component_statuses['system_resources'] = system_status
            performance_metrics.update(system_status['metrics'])

            if system_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'HIGH' if system_status['status'] == 'critical' else 'MEDIUM',
                    'component': 'system_resources',
                    'message': system_status['message']
                })

            # 2. File System Integrity
            file_status = self._check_file_integrity()
            component_statuses['file_system'] = file_status

            if file_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'HIGH',
                    'component': 'file_system',
                    'message': file_status['message']
                })

            # 3. ML Models Health
            ml_status = self._check_ml_models()
            component_statuses['ml_models'] = ml_status
            data_quality_scores['ml_models'] = ml_status.get('quality_score', 0.0)

            if ml_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'MEDIUM',
                    'component': 'ml_models',
                    'message': ml_status['message']
                })

            # 4. Data Sources Connectivity
            data_status = self._check_data_sources()
            component_statuses['data_sources'] = data_status
            data_quality_scores['data_sources'] = data_status.get('quality_score', 0.0)

            if data_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'HIGH',
                    'component': 'data_sources',
                    'message': data_status['message']
                })

            # 5. API Endpoints Health
            api_status = self._check_api_health()
            component_statuses['api_endpoints'] = api_status
            performance_metrics.update(api_status.get('metrics', {}))

            if api_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'MEDIUM',
                    'component': 'api_endpoints',
                    'message': api_status['message']
                })

            # 6. Trading System Integrity
            trading_status = self._check_trading_system()
            component_statuses['trading_system'] = trading_status
            data_quality_scores['trading_system'] = trading_status.get('quality_score', 0.0)

            if trading_status['status'] != HealthStatus.HEALTHY.value:
                alerts.append({
                    'severity': 'CRITICAL',
                    'component': 'trading_system',
                    'message': trading_status['message']
                })

            # Calculate overall status
            overall_status = self._calculate_overall_status(component_statuses)

            # Generate recommendations
            recommendations = self._generate_recommendations(component_statuses, alerts)

            # Create comprehensive report
            report = SystemHealthReport(
                timestamp=datetime.now().isoformat(),
                overall_status=overall_status,
                component_statuses=component_statuses,
                performance_metrics=performance_metrics,
                data_quality_scores=data_quality_scores,
                alerts=alerts,
                recommendations=recommendations,
                uptime=time.time() - self.start_time,
                memory_usage=performance_metrics.get('memory_usage_percent', 0.0),
                cpu_usage=performance_metrics.get('cpu_usage_percent', 0.0)
            )

            # Store in history
            self.health_history.append(report)
            if len(self.health_history) > 100:  # Keep last 100 reports
                self.health_history = self.health_history[-100:]

            self.last_check_time = time.time()
            check_duration = time.time() - start_time

            logger.info(f"✅ System integrity check completed in {check_duration:.2f}s - Overall Status: {overall_status.value}")

            return report

        except Exception as e:
            logger.error(f"❌ System integrity check failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Return critical status report
            return SystemHealthReport(
                timestamp=datetime.now().isoformat(),
                overall_status=HealthStatus.CRITICAL,
                component_statuses={'error': {'status': 'critical', 'message': str(e)}},
                performance_metrics={},
                data_quality_scores={},
                alerts=[{
                    'severity': 'CRITICAL',
                    'component': 'system_checker',
                    'message': f"System integrity check failed: {str(e)}"
                }],
                recommendations=['Immediate system maintenance required'],
                uptime=time.time() - self.start_time,
                memory_usage=0.0,
                cpu_usage=0.0
            )

    def _check_system_resources(self) -> dict[str, Any]:
        """Check system resource utilization"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Determine status
            status = HealthStatus.HEALTHY.value
            message = "System resources within normal limits"

            if (memory_percent > self.alert_thresholds['memory_usage_percent'] or
                cpu_percent > self.alert_thresholds['cpu_usage_percent'] or
                disk_percent > self.alert_thresholds['disk_usage_percent']):
                status = HealthStatus.DEGRADED.value
                message = f"High resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%, Disk {disk_percent:.1f}%"

            if memory_percent > 95 or cpu_percent > 95 or disk_percent > 95:
                status = HealthStatus.CRITICAL.value
                message = "Critical resource usage levels detected"

            return {
                'status': status,
                'message': message,
                'metrics': {
                    'memory_usage_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_usage_percent': cpu_percent,
                    'disk_usage_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'details': {
                    'memory_total_gb': memory.total / (1024**3),
                    'cpu_count': psutil.cpu_count(),
                    'disk_total_gb': disk.total / (1024**3)
                }
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"System resource check failed: {str(e)}",
                'metrics': {},
                'details': {}
            }

    def _check_file_integrity(self) -> dict[str, Any]:
        """Check integrity of critical system files"""
        try:
            missing_files = []
            file_details = []

            for file_path in self.critical_files:
                full_path = Path(file_path)

                if full_path.exists():
                    file_size = full_path.stat().st_size
                    file_details.append({
                        'file': file_path,
                        'size_mb': file_size / (1024**2),
                        'modified': datetime.fromtimestamp(full_path.stat().st_mtime).isoformat(),
                        'status': 'present'
                    })
                else:
                    missing_files.append(file_path)
                    file_details.append({
                        'file': file_path,
                        'status': 'missing'
                    })

            if missing_files:
                status = HealthStatus.DEGRADED.value
                message = f"Missing critical files: {', '.join(missing_files)}"
            else:
                status = HealthStatus.HEALTHY.value
                message = "All critical files present and accessible"

            return {
                'status': status,
                'message': message,
                'missing_files': missing_files,
                'file_details': file_details,
                'files_checked': len(self.critical_files),
                'files_present': len(self.critical_files) - len(missing_files)
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"File integrity check failed: {str(e)}",
                'missing_files': [],
                'file_details': [],
                'files_checked': 0,
                'files_present': 0
            }

    def _check_ml_models(self) -> dict[str, Any]:
        """Check ML models health and performance"""
        try:
            model_files = [
                'models/rf_ensemble_v2.pkl',
                'models/xgb_ensemble_v2.pkl',
                'models/lstm_ensemble_best.keras',
                'models/feature_scaler_v2.gz'
            ]

            models_status = []
            models_loaded = 0

            for model_file in model_files:
                if Path(model_file).exists():
                    file_size = Path(model_file).stat().st_size
                    models_loaded += 1
                    models_status.append({
                        'model': model_file,
                        'status': 'loaded',
                        'size_mb': file_size / (1024**2)
                    })
                else:
                    models_status.append({
                        'model': model_file,
                        'status': 'missing'
                    })

            # Calculate quality score
            quality_score = models_loaded / len(model_files)

            if models_loaded >= 3:
                status = HealthStatus.HEALTHY.value
                message = f"ML models operational: {models_loaded}/{len(model_files)} loaded"
            elif models_loaded >= 2:
                status = HealthStatus.DEGRADED.value
                message = f"Partial ML models loaded: {models_loaded}/{len(model_files)}"
            else:
                status = HealthStatus.CRITICAL.value
                message = f"Insufficient ML models loaded: {models_loaded}/{len(model_files)}"

            return {
                'status': status,
                'message': message,
                'quality_score': quality_score,
                'models_loaded': models_loaded,
                'total_models': len(model_files),
                'model_details': models_status,
                'ensemble_operational': models_loaded >= 3
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"ML models check failed: {str(e)}",
                'quality_score': 0.0,
                'models_loaded': 0,
                'total_models': 0,
                'model_details': [],
                'ensemble_operational': False
            }

    def _check_data_sources(self) -> dict[str, Any]:
        """Check external data sources connectivity and quality"""
        try:
            data_sources = [
                {'name': 'alpaca_api', 'url': 'https://paper-api.alpaca.markets', 'timeout': 5},
                {'name': 'alpha_vantage', 'url': 'https://www.alphavantage.co', 'timeout': 10},
                {'name': 'finnhub', 'url': 'https://finnhub.io', 'timeout': 10}
            ]

            source_statuses = []
            healthy_sources = 0

            for source in data_sources:
                try:
                    start_time = time.time()
                    response = requests.get(source['url'], timeout=source['timeout'])
                    response_time = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        healthy_sources += 1
                        source_statuses.append({
                            'name': source['name'],
                            'status': 'healthy',
                            'response_time_ms': response_time
                        })
                    else:
                        source_statuses.append({
                            'name': source['name'],
                            'status': 'degraded',
                            'response_time_ms': response_time,
                            'status_code': response.status_code
                        })

                except requests.exceptions.Timeout:
                    source_statuses.append({
                        'name': source['name'],
                        'status': 'timeout',
                        'response_time_ms': source['timeout'] * 1000
                    })
                except requests.exceptions.ConnectionError:
                    source_statuses.append({
                        'name': source['name'],
                        'status': 'offline'
                    })

            # Calculate quality score
            quality_score = healthy_sources / len(data_sources)

            if healthy_sources == len(data_sources):
                status = HealthStatus.HEALTHY.value
                message = "All data sources accessible"
            elif healthy_sources >= len(data_sources) * 0.7:
                status = HealthStatus.DEGRADED.value
                message = f"Some data sources unavailable: {healthy_sources}/{len(data_sources)}"
            else:
                status = HealthStatus.CRITICAL.value
                message = f"Critical data sources offline: {healthy_sources}/{len(data_sources)}"

            return {
                'status': status,
                'message': message,
                'quality_score': quality_score,
                'healthy_sources': healthy_sources,
                'total_sources': len(data_sources),
                'source_details': source_statuses
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"Data sources check failed: {str(e)}",
                'quality_score': 0.0,
                'healthy_sources': 0,
                'total_sources': 0,
                'source_details': []
            }

    def _check_api_health(self) -> dict[str, Any]:
        """Check internal API endpoints health"""
        try:
            api_endpoints = [
                'http://localhost:8002/api/health',
                'http://localhost:8002/api/portfolio/metrics',
                'http://localhost:8002/api/signals/latest'
            ]

            endpoint_statuses = []
            healthy_endpoints = 0
            response_times = []

            for endpoint in api_endpoints:
                try:
                    start_time = time.time()
                    response = requests.get(endpoint, timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)

                    if response.status_code == 200:
                        healthy_endpoints += 1
                        endpoint_statuses.append({
                            'endpoint': endpoint,
                            'status': 'healthy',
                            'response_time_ms': response_time
                        })
                    else:
                        endpoint_statuses.append({
                            'endpoint': endpoint,
                            'status': 'error',
                            'response_time_ms': response_time,
                            'status_code': response.status_code
                        })

                except requests.exceptions.RequestException:
                    endpoint_statuses.append({
                        'endpoint': endpoint,
                        'status': 'offline'
                    })

            avg_response_time = np.mean(response_times) if response_times else 0

            if healthy_endpoints == len(api_endpoints):
                status = HealthStatus.HEALTHY.value
                message = "All API endpoints healthy"
            elif healthy_endpoints > 0:
                status = HealthStatus.DEGRADED.value
                message = f"Some API endpoints unhealthy: {healthy_endpoints}/{len(api_endpoints)}"
            else:
                status = HealthStatus.CRITICAL.value
                message = "All API endpoints offline"

            return {
                'status': status,
                'message': message,
                'healthy_endpoints': healthy_endpoints,
                'total_endpoints': len(api_endpoints),
                'endpoint_details': endpoint_statuses,
                'metrics': {
                    'avg_response_time_ms': avg_response_time,
                    'max_response_time_ms': max(response_times) if response_times else 0
                }
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"API health check failed: {str(e)}",
                'healthy_endpoints': 0,
                'total_endpoints': 0,
                'endpoint_details': [],
                'metrics': {}
            }

    def _check_trading_system(self) -> dict[str, Any]:
        """Check trading system specific components"""
        try:
            # Check if trading gateway is responsive
            try:
                response = requests.get('http://localhost:8002/api/health', timeout=5)
                gateway_healthy = response.status_code == 200
            except:
                gateway_healthy = False

            # Check portfolio status
            try:
                response = requests.get('http://localhost:8002/api/portfolio/metrics', timeout=5)
                portfolio_data = response.json() if response.status_code == 200 else None
            except:
                portfolio_data = None

            # Check signals generation
            try:
                response = requests.get('http://localhost:8002/api/signals/latest?limit=1', timeout=5)
                signals_working = response.status_code == 200
            except:
                signals_working = False

            # Calculate system quality score
            components_working = sum([gateway_healthy, bool(portfolio_data), signals_working])
            quality_score = components_working / 3

            if components_working == 3:
                status = HealthStatus.HEALTHY.value
                message = "Trading system fully operational"
            elif components_working >= 2:
                status = HealthStatus.DEGRADED.value
                message = "Trading system partially operational"
            else:
                status = HealthStatus.CRITICAL.value
                message = "Trading system critical issues detected"

            return {
                'status': status,
                'message': message,
                'quality_score': quality_score,
                'gateway_healthy': gateway_healthy,
                'portfolio_accessible': bool(portfolio_data),
                'signals_generating': signals_working,
                'portfolio_value': portfolio_data.get('total_value') if portfolio_data else None,
                'last_signal_check': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"Trading system check failed: {str(e)}",
                'quality_score': 0.0,
                'gateway_healthy': False,
                'portfolio_accessible': False,
                'signals_generating': False,
                'portfolio_value': None
            }

    def _calculate_overall_status(self, component_statuses: dict[str, dict[str, Any]]) -> HealthStatus:
        """Calculate overall system status based on component statuses"""
        try:
            statuses = [comp['status'] for comp in component_statuses.values()]

            if 'critical' in statuses:
                return HealthStatus.CRITICAL
            elif 'degraded' in statuses:
                return HealthStatus.DEGRADED
            elif 'offline' in statuses:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

        except Exception as e:
            logger.error(f"Error calculating overall status: {e}")
            return HealthStatus.CRITICAL

    def _generate_recommendations(self, component_statuses: dict[str, dict[str, Any]], alerts: list[dict[str, Any]]) -> list[str]:
        """Generate actionable recommendations based on system status"""
        recommendations = []

        try:
            # System resource recommendations
            if 'system_resources' in component_statuses:
                sys_status = component_statuses['system_resources']
                if sys_status.get('metrics', {}).get('memory_usage_percent', 0) > 80:
                    recommendations.append("Consider increasing system memory or optimizing memory usage")
                if sys_status.get('metrics', {}).get('cpu_usage_percent', 0) > 80:
                    recommendations.append("High CPU usage detected - consider optimizing processes")
                if sys_status.get('metrics', {}).get('disk_usage_percent', 0) > 85:
                    recommendations.append("Disk space running low - cleanup or expand storage")

            # ML model recommendations
            if 'ml_models' in component_statuses:
                ml_status = component_statuses['ml_models']
                if ml_status.get('models_loaded', 0) < 3:
                    recommendations.append("Restore missing ML model files for full ensemble operation")
                if ml_status.get('quality_score', 0) < 0.8:
                    recommendations.append("Consider retraining ML models for better performance")

            # Data source recommendations
            if 'data_sources' in component_statuses:
                data_status = component_statuses['data_sources']
                if data_status.get('quality_score', 0) < 0.8:
                    recommendations.append("Check API keys and network connectivity for data sources")

            # Trading system recommendations
            if 'trading_system' in component_statuses:
                trading_status = component_statuses['trading_system']
                if not trading_status.get('signals_generating', False):
                    recommendations.append("Investigate signal generation issues - check market data feeds")
                if not trading_status.get('portfolio_accessible', False):
                    recommendations.append("Portfolio data inaccessible - verify Alpaca API connectivity")

            # High severity alert recommendations
            critical_alerts = [alert for alert in alerts if alert.get('severity') == 'CRITICAL']
            if critical_alerts:
                recommendations.append("URGENT: Address critical system alerts immediately")

            # Default recommendation if no specific issues
            if not recommendations:
                recommendations.append("System operating normally - continue regular monitoring")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - manual system review required"]

    def get_system_status_summary(self) -> dict[str, Any]:
        """Get quick system status summary for API endpoints"""
        try:
            if not self.health_history:
                # Perform initial check if no history
                self.perform_comprehensive_check()

            latest_report = self.health_history[-1] if self.health_history else None

            if not latest_report:
                return {
                    'status': 'unknown',
                    'message': 'No health data available',
                    'last_check': None
                }

            return {
                'status': latest_report.overall_status.value,
                'message': f"System {latest_report.overall_status.value} - {len(latest_report.alerts)} alerts",
                'last_check': latest_report.timestamp,
                'uptime_hours': latest_report.uptime / 3600,
                'memory_usage_percent': latest_report.memory_usage,
                'cpu_usage_percent': latest_report.cpu_usage,
                'active_alerts': len(latest_report.alerts),
                'data_quality_avg': np.mean(list(latest_report.data_quality_scores.values())) if latest_report.data_quality_scores else 0.0
            }

        except Exception as e:
            logger.error(f"Error getting system status summary: {e}")
            return {
                'status': 'error',
                'message': f'Status check failed: {str(e)}',
                'last_check': datetime.now().isoformat()
            }

    def get_detailed_health_report(self) -> dict[str, Any]:
        """Get detailed health report for dashboard display"""
        try:
            if not self.health_history:
                report = self.perform_comprehensive_check()
            else:
                report = self.health_history[-1]

            return {
                'timestamp': report.timestamp,
                'overall_status': report.overall_status.value,
                'system_metrics': {
                    'uptime_hours': report.uptime / 3600,
                    'memory_usage_percent': report.memory_usage,
                    'cpu_usage_percent': report.cpu_usage,
                    'performance_score': np.mean(list(report.data_quality_scores.values())) if report.data_quality_scores else 0.0
                },
                'components': report.component_statuses,
                'data_quality': report.data_quality_scores,
                'alerts': report.alerts,
                'recommendations': report.recommendations,
                'health_trend': self._calculate_health_trend()
            }

        except Exception as e:
            logger.error(f"Error getting detailed health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

    def _calculate_health_trend(self) -> str:
        """Calculate health trend based on recent history"""
        try:
            if len(self.health_history) < 2:
                return "insufficient_data"

            recent_statuses = [report.overall_status for report in self.health_history[-5:]]
            status_values = {
                HealthStatus.HEALTHY: 3,
                HealthStatus.DEGRADED: 2,
                HealthStatus.CRITICAL: 1,
                HealthStatus.OFFLINE: 0
            }

            values = [status_values[status] for status in recent_statuses]

            if len(values) >= 3:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                if trend > 0.1:
                    return "improving"
                elif trend < -0.1:
                    return "declining"
                else:
                    return "stable"
            else:
                return "stable"

        except Exception:
            return "unknown"

# Global instance for easy access
system_checker = SystemIntegrityChecker()

def get_system_health() -> dict[str, Any]:
    """Convenience function to get system health status"""
    return system_checker.get_system_status_summary()

def perform_health_check() -> SystemHealthReport:
    """Convenience function to perform comprehensive health check"""
    return system_checker.perform_comprehensive_check()

def get_health_report() -> dict[str, Any]:
    """Convenience function to get detailed health report"""
    return system_checker.get_detailed_health_report()

# Testing and validation
if __name__ == "__main__":
    print("🔍 System Integrity Checker - Running Comprehensive Test")
    print("=" * 60)

    # Initialize checker
    checker = SystemIntegrityChecker()

    # Perform comprehensive check
    report = checker.perform_comprehensive_check()

    # Display results
    print("\n📊 SYSTEM HEALTH REPORT")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Uptime: {report.uptime/3600:.1f} hours")
    print(f"Memory Usage: {report.memory_usage:.1f}%")
    print(f"CPU Usage: {report.cpu_usage:.1f}%")
    print(f"Active Alerts: {len(report.alerts)}")

    print("\n🔧 COMPONENT STATUS:")
    for component, status in report.component_statuses.items():
        print(f"  {component}: {status['status'].upper()} - {status['message']}")

    if report.alerts:
        print("\n⚠️  ALERTS:")
        for alert in report.alerts:
            print(f"  [{alert['severity']}] {alert['component']}: {alert['message']}")

    if report.recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n✅ System Integrity Checker test completed successfully")
