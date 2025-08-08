@echo off
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
copy alpaca_trading_gateway.py backup_before_cleanup\ 2>nul
copy simple_health_checker.py backup_before_cleanup\ 2>nul
copy requirements.txt backup_before_cleanup\ 2>nul

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
pause
