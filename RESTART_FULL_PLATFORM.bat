@echo off
echo 🔄 RESTARTING FULL PLATFORM - Trading Gateway + React Dashboard
echo ============================================================

REM Stop any existing processes
echo Stopping existing processes...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
timeout /t 3 /nobreak >nul

echo.
echo 🚀 Starting Trading Gateway (Port 8002)...
REM Start gateway in background with log redirection
start /min "Trading Gateway" cmd /c "python alpaca_trading_gateway.py > trading_gateway_restart.log 2>&1"

REM Wait for gateway to initialize
echo Waiting for Trading Gateway to initialize...
timeout /t 8 /nobreak >nul

echo.
echo 📊 Starting React Dashboard (Port 3003)...
REM Change to dashboard directory and start
cd trading-dashboard
start /min "React Dashboard" cmd /c "npm run dev > dashboard_restart.log 2>&1"
cd ..

echo.
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

echo.
echo 🧪 TESTING SERVICES...
echo =====================

REM Test Trading Gateway
curl -s http://localhost:8002/api/health > health_test.json 2>nul
if %errorlevel% equ 0 (
    echo ✅ Trading Gateway: RUNNING on http://localhost:8002
) else (
    echo ❌ Trading Gateway: NOT RESPONDING
)

REM Test React Dashboard
curl -s http://localhost:3003 > dashboard_test.html 2>nul
if %errorlevel% equ 0 (
    echo ✅ React Dashboard: RUNNING on http://localhost:3003
    echo 📊 Opening dashboard in browser...
    start http://localhost:3003
) else (
    echo ❌ React Dashboard: NOT RESPONDING
    echo.
    echo 🔍 Checking React logs...
    type trading-dashboard\dashboard_restart.log | findstr /i "error" || echo No errors found
)

echo.
echo 📋 QUICK ACCESS LINKS:
echo ======================
echo 🌐 Dashboard:     http://localhost:3003
echo ⚙️ Trading API:   http://localhost:8002
echo 📖 API Docs:     http://localhost:8002/docs
echo 🔍 Health Check: http://localhost:8002/api/health
echo.
echo 📁 LOG FILES:
echo Trading Gateway: trading_gateway_restart.log
echo React Dashboard: trading-dashboard\dashboard_restart.log
echo.

REM Final signal test
echo 🎯 Testing Signals Endpoint...
curl -s "http://localhost:8002/api/signals/latest?limit=3" > final_signals_test.json
findstr /c:"count" final_signals_test.json >nul
if %errorlevel% equ 0 (
    echo ✅ Signals endpoint working - Latest Trading Signals should appear in dashboard
) else (
    echo ⚠️ Signals endpoint may need more time to initialize
)

echo.
echo ✅ PLATFORM RESTART COMPLETE!
echo Both services should now be running properly.
pause
