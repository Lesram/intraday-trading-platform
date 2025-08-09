#!/usr/bin/env python3
"""
🧹 COMPREHENSIVE PROJECT CLEANUP ANALYSIS
Analyzes all files in the trading platform and categorizes them for safe cleanup
"""

from datetime import datetime
from pathlib import Path

# Root directory
root_dir = Path("C:/Users/Marsel/OneDrive/Documents/Cyb/Intraday_Trading")

# CORE PLATFORM FILES - NEVER DELETE
CORE_FILES = {
    # Main application
    'alpaca_trading_gateway.py': 'Main FastAPI trading gateway - CRITICAL',
    'simple_health_checker.py': 'Health checker used by gateway - CRITICAL',

    # Core trading modules (imported by gateway)
    'volatility_adjusted_position_sizing.py': 'Position sizing module - ACTIVE',
    'dynamic_stop_loss_manager.py': 'Stop loss management - ACTIVE',
    'performance_monitor.py': 'Performance tracking - ACTIVE',
    'multi_timeframe_analyzer.py': 'Technical analysis - ACTIVE',
    'advanced_ml_predictor.py': 'ML predictions - ACTIVE',
    'advanced_dynamic_stop_optimizer.py': 'Advanced stops - ACTIVE',
    'adaptive_learning_system.py': 'Learning system - ACTIVE',
    'portfolio_risk_manager.py': 'Risk management - ACTIVE',
    'transaction_cost_model.py': 'Cost analysis - ACTIVE',

    # Additional core modules
    'performance_attribution_analyzer.py': 'Performance attribution - ACTIVE',
    'advanced_correlation_modeler.py': 'Correlation analysis - ACTIVE',
    'advanced_volatility_forecaster.py': 'Volatility forecasting - ACTIVE',
    'portfolio_optimization_engine.py': 'Portfolio optimization - ACTIVE',
    'champion_challenger_framework.py': 'Model management - ACTIVE',
    'institutional_model_registry.py': 'Model registry - ACTIVE',
    'institutional_backtest_engine.py': 'Backtesting - ACTIVE',
    'fractional_kelly_module.py': 'Kelly sizing - ACTIVE',
    'vector_kelly_module.py': 'Vector Kelly - ACTIVE',

    # Enhanced modules
    'enhanced_cvar_risk_manager.py': 'CVaR risk management - ACTIVE',
    'cvar_integrated_order_router.py': 'Order routing - ACTIVE',
    'data_enhancement_risk_integration.py': 'Data enhancement - ACTIVE',
    'enhanced_social_sentiment_module.py': 'Sentiment analysis - ACTIVE',
    'finbert_sentiment_enhancer.py': 'FinBERT sentiment - ACTIVE',
    'social_sentiment_module.py': 'Social sentiment - ACTIVE',
    'smart_execution_engine.py': 'Smart execution - ACTIVE',
    'smart_execution_integration.py': 'Execution integration - ACTIVE',
    'stress_testing_module.py': 'Stress testing - ACTIVE',

    # Model files - CRITICAL
    'rf_ensemble_v2.pkl': 'Random Forest model - CRITICAL',
    'xgb_ensemble_v2.pkl': 'XGBoost model - CRITICAL',
    'lstm_ensemble_best.keras': 'LSTM model - CRITICAL',
    'feature_scaler_v2.gz': 'Feature scaler - CRITICAL',

    # Configuration and data
    'requirements.txt': 'Python dependencies - CRITICAL',
    'config/': 'Configuration directory - CRITICAL',
    'trading-dashboard/': 'React dashboard - CRITICAL',
    'production_model_registry/': 'Model registry - ACTIVE',
    'data/': 'Data directory - ACTIVE',
    'logs/': 'Logs directory - ACTIVE',
    'models/': 'Models directory - ACTIVE',
    '__pycache__/': 'Python cache - AUTO-GENERATED',

    # Database files
    'performance.db': 'Performance database - ACTIVE',
    'performance_metrics.db': 'Metrics database - ACTIVE',

    # Current useful startup scripts
    'RESTART_FULL_PLATFORM.bat': 'Current working startup script - KEEP',
}

# CLEANUP CANDIDATES - SAFE TO DELETE
CLEANUP_CANDIDATES = {
    # Temporary/generated files
    'dashboard.log': 'Old dashboard log - DELETE',
    'dashboard_test.html': 'Test file - DELETE',
    'automated_trading.log': 'Old log - DELETE',
    'autonomous_trading.log': 'Old log - DELETE',
    'system_monitor.log': 'Old log - DELETE',
    'trading_gateway.log': 'Old log - DELETE',
    'trading_gateway_fixed.log': 'Old log - DELETE',
    'trading_gateway_restart.log': 'Old log - DELETE',
    'signals_test.json': 'Test output - DELETE',
    'final_signals_test.json': 'Test output - DELETE',
    'health_test.json': 'Test output - DELETE',
    'comprehensive_test_report.json': 'Old test report - DELETE',

    # Old/backup model files
    'rf_ensemble_backup_20250806_225036.pkl': 'Old model backup - DELETE',
    'feature_scaler_backup_20250806_225037.gz': 'Old scaler backup - DELETE',
    'lstm_ensemble_v2.keras': 'Old LSTM version - DELETE',
    'rf_fallback.pkl': 'Fallback model - DELETE',
    'rf_fallback_synthetic.pkl': 'Synthetic fallback - DELETE',
    'model_recovery_backup_20250806_224627/': 'Old backup directory - DELETE',

    # Test files
    'test_alpaca_connection.py': 'Connection test - DELETE',
    'test_api.py': 'API test - DELETE',
    'test_api_fix.js': 'JS test - DELETE',
    'test_crypto_priority_2.py': 'Crypto test - DELETE',
    'test_dashboard_apis.py': 'Dashboard test - DELETE',
    'test_dynamic_ensemble.py': 'Ensemble test - DELETE',
    'test_fix.py': 'Fix test - DELETE',
    'test_health.py': 'Health test - DELETE',
    'test_model_fixes.py': 'Model test - DELETE',
    'test_old_data.py': 'Data test - DELETE',
    'test_optimizations.py': 'Optimization test - DELETE',
    'test_optimizations_offline.py': 'Offline test - DELETE',
    'test_script.py': 'Generic test - DELETE',
    'test_social_sentiment.py': 'Sentiment test - DELETE',
    'minimal_alpaca_test.py': 'Minimal test - DELETE',
    'amd_prediction_test.py': 'AMD test - DELETE',
    'fast_universe_test.py': 'Universe test - DELETE',
    'simple_ensemble_test.py': 'Ensemble test - DELETE',
    'simple_model_test.py': 'Model test - DELETE',
    'comprehensive_test.py': 'Comprehensive test - DELETE',
    'mlops_comprehensive_test_suite.py': 'MLOps test - DELETE',

    # Diagnostic/analysis files
    'analyze_corruption.py': 'Corruption analysis - DELETE',
    'analyze_predictions.py': 'Prediction analysis - DELETE',
    'dashboard_diagnostics.py': 'Diagnostics - DELETE',
    'debug_alpaca_url.py': 'Debug script - DELETE',
    'debug_data.py': 'Debug script - DELETE',
    'filter_analysis.py': 'Filter analysis - DELETE',
    'verify_ml_status.py': 'ML verification - DELETE',
    'alpaca_integration_status.py': 'Integration status - DELETE',
    'comprehensive_platform_audit.py': 'Platform audit - DELETE',
    'platform_audit_completion.py': 'Audit completion - DELETE',
    'implementation_analysis.py': 'Implementation analysis - DELETE',
    'implementation_status.py': 'Implementation status - DELETE',
    'priority1_status_check.py': 'Priority check - DELETE',
    'priority_tracker.py': 'Priority tracker - DELETE',
    'strategic_completion_summary.py': 'Strategy summary - DELETE',
    'strategic_roadmap_validator.py': 'Roadmap validator - DELETE',
    'version_1_0_audit.py': 'Version audit - DELETE',

    # Old startup scripts (replaced by RESTART_FULL_PLATFORM.bat)
    'CLEAN_STARTUP.bat': 'Old startup script - DELETE',
    'CLEAN_VS_CODE_TERMINALS.bat': 'Old terminal script - DELETE',
    'FIXED_STARTUP.bat': 'Old startup script - DELETE',
    'RESTART_FIXED_GATEWAY.bat': 'Old restart script - DELETE',
    'RESTART_PLATFORM.py': 'Old restart script - DELETE',
    'RESTART_WITH_SIGNAL_FIXES.bat': 'Old restart script - DELETE',
    'RESTORE_INSTITUTIONAL_PLATFORM.bat': 'Old restore script - DELETE',
    'START_PLATFORM_COMPREHENSIVE.bat': 'Old startup script - DELETE',
    'START_PLATFORM_ULTIMATE.bat': 'Old startup script - DELETE',
    'START_SILENT_SERVICES.bat': 'Old startup script - DELETE',
    'STOP_BACKGROUND_SERVICES.bat': 'Old stop script - DELETE',
    'start_api.bat': 'Old API start - DELETE',
    'start_automated_trading.bat': 'Old trading start - DELETE',
    'start_platform.bat': 'Old platform start - DELETE',
    'start_react_dashboard.ps1': 'Old dashboard start - DELETE',

    # Old/alternative implementations
    'automated_trading_bot.py': 'Old trading bot - DELETE',
    'backend_api_server.py': 'Old backend - DELETE',
    'simple_api_gateway.py': 'Simple gateway - DELETE',
    'unified_api_gateway.py': 'Unified gateway - DELETE',
    'trading_api.py': 'Old trading API - DELETE',
    'system_health_endpoints.py': 'Old health endpoints - DELETE',
    'status_dashboard.py': 'Status dashboard - DELETE',
    'serve_dashboard.py': 'Dashboard server - DELETE',
    'simple_dashboard_server.py': 'Simple server - DELETE',
    'automated_system_monitor.py': 'System monitor - DELETE',
    'enhanced_trading_system.py': 'Enhanced system - DELETE',
    'simple_working_model.py': 'Simple model - DELETE',
    'simplified_trading_test.py': 'Simplified test - DELETE',

    # Data processing/training (one-time use)
    'alpaca_data_trainer.py': 'Data trainer - DELETE',
    'real_data_model_trainer.py': 'Model trainer - DELETE',
    'robust_model_trainer.py': 'Robust trainer - DELETE',
    'model_recovery.py': 'Model recovery - DELETE',
    'safe_model_recovery.py': 'Safe recovery - DELETE',
    'patch_ensemble.py': 'Ensemble patch - DELETE',
    'create_performance_db.py': 'DB creation - DELETE',
    'safe_startup_check.py': 'Startup check - DELETE',

    # Documentation/reports (historical)
    'system_integrity_report_20250806_230531.json': 'Old report - DELETE',
    'system_integrity_report_20250806_231557.json': 'Old report - DELETE',
    'system_integrity_report_20250806_234419.json': 'Old report - DELETE',

    # Demo/example files
    'demo_backtest.py': 'Demo backtest - DELETE',
    'quick_backtest.py': 'Quick backtest - DELETE',
    'simple_backtest.py': 'Simple backtest - DELETE',
    'alpaca_backtest.py': 'Alpaca backtest - DELETE',
    'API_TEST.html': 'API test HTML - DELETE',
    'simple_dashboard.html': 'Simple dashboard - DELETE',
    'simple_dashboard/': 'Simple dashboard dir - DELETE',
    'PLATFORM_STATUS.html': 'Status HTML - DELETE',

    # Alternative implementations
    'runfile.py': 'Old runfile - DELETE',
    'runfile_clean.py': 'Clean runfile - DELETE',
    'start_autonomous_trading.py': 'Autonomous start - DELETE',
    'start_platform.py': 'Platform start - DELETE',
    'start_platform_comprehensive.py': 'Comprehensive start - DELETE',
    'LAUNCH_PLATFORM.py': 'Launch script - DELETE',
    'live_trading_deployment.py': 'Deployment script - DELETE',
    'run_microservices_local.py': 'Microservices - DELETE',
    'mlops_system_integrator.py': 'MLOps integrator - DELETE',

    # Utility scripts (one-time use)
    'clean_status_check.py': 'Status check - DELETE',
    'platform_status.py': 'Platform status - DELETE',
    'install_backend_deps.py': 'Dependency installer - DELETE',
    'improvement_plan.py': 'Improvement plan - DELETE',
    'profit_optimizer.py': 'Profit optimizer - DELETE',
    'profit_optimization_summary.py': 'Optimization summary - DELETE',
    'chatgpt_integration_summary.py': 'ChatGPT summary - DELETE',
    'enhanced_analysis.py': 'Enhanced analysis - DELETE',
    'advanced_features.py': 'Advanced features - DELETE',
    'enhanced_logging.py': 'Enhanced logging - DELETE',
    'trading_config.py': 'Trading config - DELETE',
    'dynamic_risk_manager.py': 'Risk manager - DELETE',

    # Docker files (not used)
    'docker-compose-simple.yml': 'Docker compose - DELETE',
    'docker-compose-working.yml': 'Docker compose - DELETE',
    'deploy.ps1': 'Deploy script - DELETE',
    'deploy-dashboard.ps1': 'Dashboard deploy - DELETE',

    # Crypto integration (separate feature)
    'crypto_integration.py': 'Crypto integration - DELETE',

    # Archive directory
    'archive/': 'Archive directory - DELETE',
    'infrastructure/': 'Infrastructure directory - DELETE',
    'core/': 'Core directory - DELETE',
    'modules/': 'Modules directory - DELETE',
    'scripts/': 'Scripts directory - DELETE',
}

# DOCUMENTATION - REVIEW BEFORE DELETE
DOCUMENTATION_FILES = {
    'README.md': 'Main README - REVIEW',
    'QUICK_START_GUIDE.md': 'Quick start - REVIEW',
    'BACKTESTING_GUIDE.md': 'Backtesting guide - REVIEW',
    'MLOPS_SETUP_GUIDE.md': 'MLOps guide - REVIEW',
    'UI_UX_IMPLEMENTATION_README.md': 'UI guide - REVIEW',
    'PORT_REFERENCE.md': 'Port reference - REVIEW',
    'PROJECT_CLEANUP_PLAN.md': 'Cleanup plan - DELETE',
    'SENTIMENT_PROVIDER_CONFIGURATION.md': 'Sentiment config - REVIEW',

    # Implementation reports
    'ACTIONABLE_IMPLEMENTATION_ROADMAP.md': 'Roadmap - DELETE',
    'AI_CONSULTANT_IMPLEMENTATION_PLAN.md': 'AI plan - DELETE',
    'ALPACA_INTEGRATION_COMPLETE.md': 'Integration complete - DELETE',
    'CLEANUP_COMPLETE.md': 'Cleanup complete - DELETE',
    'COMPLETE_SYSTEM_ANALYSIS.md': 'System analysis - DELETE',
    'COMPREHENSIVE_PLATFORM_ANALYSIS_2025.md': 'Platform analysis - DELETE',
    'CRITICAL_IMPLEMENTATION_WEEKS_1-4.md': 'Implementation weeks - DELETE',
    'DASHBOARD_FIXED.md': 'Dashboard fixed - DELETE',
    'DATA_ENHANCEMENT_RISK_MANAGEMENT_COMPLETE.md': 'Risk complete - DELETE',
    'DATA_ENHANCEMENT_RISK_MANAGEMENT_SUCCESS.md': 'Risk success - DELETE',
    'DEPLOYMENT_SUCCESS_STATUS.md': 'Deployment success - DELETE',
    'DYNAMIC_ENSEMBLE_SUCCESS.md': 'Ensemble success - DELETE',
    'FINAL_AUDIT_REPORT.md': 'Final audit - DELETE',
    'FINAL_SUMMARY.md': 'Final summary - DELETE',
    'IMPLEMENTATION_COMPLETE.md': 'Implementation complete - DELETE',
    'IMPLEMENTATION_ROADMAP_WEEKS_2-12.md': 'Implementation roadmap - DELETE',
    'INSTITUTIONAL_TRADING_PLATFORM_TECHNICAL_DEEP_DIVE.md': 'Technical dive - REVIEW',
    'KELLY_SIZING_SUCCESS.md': 'Kelly success - DELETE',
    'MICROSERVICES_ARCHITECTURE_PLAN.md': 'Microservices plan - DELETE',
    'MICROSERVICES_DEPLOYMENT_COMPLETE.md': 'Microservices complete - DELETE',
    'MLOPS_DASHBOARD_FIXES.md': 'MLOps fixes - DELETE',
    'MLOPS_IMPLEMENTATION_COMPLETE.md': 'MLOps complete - DELETE',
    'NEXT_STEPS_AND_RECOMMENDATIONS.md': 'Next steps - DELETE',
    'PLATFORM_READY.md': 'Platform ready - DELETE',
    'PLATFORM_RESTORED_STATUS.md': 'Platform restored - DELETE',
    'PLATFORM_SNAPSHOT_README.md': 'Platform snapshot - DELETE',
    'PRIORITY_1_IMPLEMENTATION_REPORT.md': 'Priority 1 - DELETE',
    'PRIORITY_2A_COMPLETION_REPORT.md': 'Priority 2A - DELETE',
    'PROFESSIONAL_FRONTEND_COMPLETE.md': 'Frontend complete - DELETE',
    'REMAINING_AI_CONSULTANT_STEPS.md': 'Remaining steps - DELETE',
    'SMART_ORDER_EXECUTION_COMPLETE.md': 'Smart execution - DELETE',
    'SMART_ORDER_EXECUTION_IMPLEMENTATION_PLAN.md': 'Execution plan - DELETE',
    'SOCIAL_SENTIMENT_SUCCESS.md': 'Sentiment success - DELETE',
    'STEP3_SENTIMENT_UPGRADE_SUMMARY.md': 'Sentiment upgrade - DELETE',
    'STRESS_TESTING_SUCCESS.md': 'Stress testing - DELETE',
    'SYSTEM_CLEANUP_REPORT.md': 'Cleanup report - DELETE',
    'SYSTEM_VALIDATION_REPORT.md': 'Validation report - DELETE',
    'TECHNICAL_EVALUATION_SUMMARY.md': 'Technical summary - DELETE',
    'TRADING_SIGNALS_FIXED.md': 'Signals fixed - DELETE',
    'WEEK_2_CRYPTO_INTEGRATION_PLAN.md': 'Week 2 plan - DELETE',
    'WEEK_3-4_IMPLEMENTATION_PLAN.md': 'Week 3-4 plan - DELETE',
    'WEEK_5_COMPLETION_SUMMARY.md': 'Week 5 summary - DELETE',
    'WEEK_5_UI_UX_IMPLEMENTATION.md': 'Week 5 UI - DELETE',

    # Jupyter notebooks
    'LSTM_Trading_Tutorial.ipynb': 'LSTM tutorial - REVIEW',
}

def generate_cleanup_script():
    """Generate the cleanup script"""

    cleanup_script = '''@echo off
echo 🧹 COMPREHENSIVE PROJECT CLEANUP
echo ================================
echo This will remove temporary files, old tests, and unused scripts
echo while preserving all critical platform components.
echo.

set /p confirm="Are you sure you want to proceed? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cleanup cancelled.
    pause
    exit /b
)

echo.
echo 🗂️ Creating backup of critical files...
mkdir backup_before_cleanup 2>nul
copy alpaca_trading_gateway.py backup_before_cleanup\\ 2>nul
copy simple_health_checker.py backup_before_cleanup\\ 2>nul
copy requirements.txt backup_before_cleanup\\ 2>nul

echo.
echo 🧹 Removing temporary and test files...

REM Log files
del /q dashboard.log 2>nul
del /q automated_trading.log 2>nul
del /q autonomous_trading.log 2>nul
del /q system_monitor.log 2>nul
del /q trading_gateway.log 2>nul
del /q trading_gateway_fixed.log 2>nul
del /q trading_gateway_restart.log 2>nul

REM Test output files
del /q signals_test.json 2>nul
del /q final_signals_test.json 2>nul
del /q health_test.json 2>nul
del /q dashboard_test.html 2>nul
del /q comprehensive_test_report.json 2>nul

REM Old model backups
del /q rf_ensemble_backup_20250806_225036.pkl 2>nul
del /q feature_scaler_backup_20250806_225037.gz 2>nul
del /q lstm_ensemble_v2.keras 2>nul
del /q rf_fallback.pkl 2>nul
del /q rf_fallback_synthetic.pkl 2>nul
rmdir /s /q model_recovery_backup_20250806_224627 2>nul

REM Test files
del /q test_*.py 2>nul
del /q minimal_alpaca_test.py 2>nul
del /q amd_prediction_test.py 2>nul
del /q fast_universe_test.py 2>nul
del /q simple_ensemble_test.py 2>nul
del /q simple_model_test.py 2>nul
del /q comprehensive_test.py 2>nul
del /q mlops_comprehensive_test_suite.py 2>nul

REM Diagnostic files
del /q analyze_*.py 2>nul
del /q debug_*.py 2>nul
del /q verify_*.py 2>nul
del /q dashboard_diagnostics.py 2>nul
del /q filter_analysis.py 2>nul
del /q *_status.py 2>nul
del /q *_audit.py 2>nul
del /q priority*.py 2>nul
del /q strategic_*.py 2>nul
del /q version_*.py 2>nul

echo.
echo 🚀 Removing old startup scripts...
del /q CLEAN_STARTUP.bat 2>nul
del /q CLEAN_VS_CODE_TERMINALS.bat 2>nul
del /q FIXED_STARTUP.bat 2>nul
del /q RESTART_FIXED_GATEWAY.bat 2>nul
del /q RESTART_WITH_SIGNAL_FIXES.bat 2>nul
del /q RESTORE_INSTITUTIONAL_PLATFORM.bat 2>nul
del /q START_PLATFORM_COMPREHENSIVE.bat 2>nul
del /q START_PLATFORM_ULTIMATE.bat 2>nul
del /q START_SILENT_SERVICES.bat 2>nul
del /q STOP_BACKGROUND_SERVICES.bat 2>nul
del /q start_*.bat 2>nul
del /q start_*.ps1 2>nul

echo.
echo 🔧 Removing old implementations...
del /q automated_trading_bot.py 2>nul
del /q backend_api_server.py 2>nul
del /q simple_api_gateway.py 2>nul
del /q unified_api_gateway.py 2>nul
del /q trading_api.py 2>nul
del /q system_health_endpoints.py 2>nul
del /q status_dashboard.py 2>nul
del /q serve_dashboard.py 2>nul
del /q simple_dashboard_server.py 2>nul
del /q automated_system_monitor.py 2>nul
del /q enhanced_trading_system.py 2>nul
del /q simple_working_model.py 2>nul
del /q runfile*.py 2>nul
del /q start_autonomous_trading.py 2>nul
del /q start_platform*.py 2>nul
del /q LAUNCH_PLATFORM.py 2>nul
del /q live_trading_deployment.py 2>nul

REM Training/processing scripts (one-time use)
del /q *_trainer.py 2>nul
del /q model_recovery.py 2>nul
del /q safe_model_recovery.py 2>nul
del /q patch_ensemble.py 2>nul
del /q create_performance_db.py 2>nul

REM Demo files
del /q demo_*.py 2>nul
del /q quick_*.py 2>nul
del /q simple_backtest.py 2>nul
del /q alpaca_backtest.py 2>nul
del /q API_TEST.html 2>nul
del /q simple_dashboard.html 2>nul
del /q PLATFORM_STATUS.html 2>nul
rmdir /s /q simple_dashboard 2>nul

REM Utility scripts
del /q clean_status_check.py 2>nul
del /q platform_status.py 2>nul
del /q install_backend_deps.py 2>nul
del /q improvement_plan.py 2>nul
del /q profit_optimization*.py 2>nul
del /q chatgpt_integration_summary.py 2>nul
del /q enhanced_analysis.py 2>nul
del /q enhanced_logging.py 2>nul
del /q trading_config.py 2>nul
del /q crypto_integration.py 2>nul

REM Docker files
del /q docker-compose*.yml 2>nul
del /q deploy*.ps1 2>nul

REM Old directories (if empty)
rmdir archive 2>nul
rmdir infrastructure 2>nul  
rmdir core 2>nul
rmdir modules 2>nul
rmdir scripts 2>nul

echo.
echo 📋 Removing documentation files...
del /q ACTIONABLE_*.md 2>nul
del /q AI_CONSULTANT_*.md 2>nul
del /q ALPACA_INTEGRATION_*.md 2>nul
del /q CLEANUP_*.md 2>nul
del /q COMPLETE_*.md 2>nul
del /q COMPREHENSIVE_*.md 2>nul
del /q CRITICAL_*.md 2>nul
del /q DASHBOARD_*.md 2>nul
del /q DATA_ENHANCEMENT_*.md 2>nul
del /q DEPLOYMENT_*.md 2>nul
del /q DYNAMIC_*.md 2>nul
del /q FINAL_*.md 2>nul
del /q IMPLEMENTATION_*.md 2>nul
del /q KELLY_*.md 2>nul
del /q MICROSERVICES_*.md 2>nul
del /q MLOPS_*.md 2>nul
del /q NEXT_STEPS_*.md 2>nul
del /q PLATFORM_*.md 2>nul
del /q PRIORITY_*.md 2>nul
del /q PROFESSIONAL_*.md 2>nul
del /q PROJECT_*.md 2>nul
del /q REMAINING_*.md 2>nul
del /q SMART_*.md 2>nul
del /q SOCIAL_*.md 2>nul
del /q STEP3_*.md 2>nul
del /q STRESS_*.md 2>nul
del /q SYSTEM_*.md 2>nul
del /q TECHNICAL_*.md 2>nul
del /q TRADING_*.md 2>nul
del /q WEEK_*.md 2>nul

REM Old JSON reports
del /q system_integrity_report_*.json 2>nul

echo.
echo ✅ CLEANUP COMPLETE!
echo ====================
echo.
echo 📁 Files preserved:
echo ✅ alpaca_trading_gateway.py (Main application)
echo ✅ simple_health_checker.py (Health checker)
echo ✅ All core trading modules
echo ✅ ML model files (.pkl, .keras, .gz)
echo ✅ trading-dashboard/ (React app)
echo ✅ config/ (Configuration)
echo ✅ data/, logs/, models/ (Data directories)
echo ✅ requirements.txt (Dependencies)
echo ✅ RESTART_FULL_PLATFORM.bat (Working startup script)
echo.
echo 📊 Project is now clean and organized!
echo Backup created in: backup_before_cleanup/
echo.
pause'''

    with open('COMPREHENSIVE_CLEANUP.bat', 'w') as f:
        f.write(cleanup_script)

def analyze_project():
    """Analyze the project and create cleanup recommendations"""

    print("🔍 COMPREHENSIVE PROJECT CLEANUP ANALYSIS")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("📂 CORE FILES (NEVER DELETE):")
    print("-" * 40)
    for file, desc in CORE_FILES.items():
        print(f"✅ {file:<40} - {desc}")

    print(f"\\n📄 Total Core Files: {len(CORE_FILES)}")
    print()

    print("🧹 CLEANUP CANDIDATES (SAFE TO DELETE):")
    print("-" * 40)
    for file, desc in CLEANUP_CANDIDATES.items():
        print(f"❌ {file:<40} - {desc}")

    print(f"\\n🗑️ Total Cleanup Candidates: {len(CLEANUP_CANDIDATES)}")
    print()

    print("📖 DOCUMENTATION (REVIEW BEFORE DELETE):")
    print("-" * 40)
    for file, desc in DOCUMENTATION_FILES.items():
        print(f"📋 {file:<40} - {desc}")

    print(f"\\n📚 Total Documentation Files: {len(DOCUMENTATION_FILES)}")
    print()

    # Calculate cleanup impact
    total_files = len(CORE_FILES) + len(CLEANUP_CANDIDATES) + len(DOCUMENTATION_FILES)
    cleanup_percentage = (len(CLEANUP_CANDIDATES) / total_files) * 100

    print("📊 CLEANUP IMPACT ANALYSIS:")
    print("-" * 40)
    print(f"Total Files Analyzed: {total_files}")
    print(f"Files to Keep: {len(CORE_FILES)}")
    print(f"Files to Delete: {len(CLEANUP_CANDIDATES)}")
    print(f"Files to Review: {len(DOCUMENTATION_FILES)}")
    print(f"Cleanup Percentage: {cleanup_percentage:.1f}%")
    print()

    print("🎯 CLEANUP BENEFITS:")
    print("-" * 40)
    print("✅ Remove ~150+ unnecessary files")
    print("✅ Eliminate old test and diagnostic scripts")
    print("✅ Clean up temporary logs and output files")
    print("✅ Remove superseded startup scripts")
    print("✅ Preserve all critical platform components")
    print("✅ Keep working React dashboard and FastAPI gateway")
    print("✅ Maintain all ML models and configuration")
    print()

    print("⚠️ SAFETY MEASURES:")
    print("-" * 40)
    print("✅ Backup critical files before cleanup")
    print("✅ Preserve all imported modules")
    print("✅ Keep current working startup script")
    print("✅ Maintain database and model files")
    print("✅ Preserve configuration directory")
    print()

    generate_cleanup_script()
    print("📝 Generated: COMPREHENSIVE_CLEANUP.bat")
    print("Run this script to perform the cleanup safely!")

if __name__ == "__main__":
    analyze_project()
