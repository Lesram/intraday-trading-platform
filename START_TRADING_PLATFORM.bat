@echo off
REM ============================================================================
REM                   AUTOMATED TRADING PLATFORM STARTUP
REM ============================================================================
REM This batch file automates the startup of both backend and frontend servers
REM Solves PowerShell stalling issues by using native Windows commands
REM ============================================================================

REM Set UTF-8 encoding for proper Unicode support
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

color 0A
echo.
echo 🚀 AUTOMATED TRADING PLATFORM STARTUP
echo ====================================
echo Date: %date%
echo Time: %time%
echo.

REM Set working directory
cd /d "C:\Users\Marsel\OneDrive\Documents\Cyb\Intraday_Trading"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Kill any existing processes
echo 🔄 Cleaning up existing processes...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
taskkill /f /im npm.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clear previous logs
del /q logs\backend_startup.log 2>nul
del /q logs\frontend_startup.log 2>nul
del /q logs\startup_errors.log 2>nul

echo ✅ Process cleanup complete
echo.

REM ============================================================================
REM                           START BACKEND SERVER
REM ============================================================================

echo 🖥️ Starting Trading Gateway (Backend - Port 8002)...
echo Starting backend at %time% > logs\backend_startup.log

REM Start backend in minimized window with logging and UTF-8 encoding
start /min "Trading Gateway Backend" cmd /c "set PYTHONIOENCODING=utf-8 && chcp 65001 >nul 2>&1 && python alpaca_trading_gateway.py >> logs\backend_startup.log 2>>&1"

echo ⏳ Waiting for backend initialization (10 seconds)...
timeout /t 10 /nobreak >nul

REM Check if backend is responding
echo 🔍 Testing backend health...
curl -s --connect-timeout 5 http://localhost:8002/api/health > logs\backend_health.json 2>logs\startup_errors.log

if %errorlevel% equ 0 (
    echo ✅ Backend: ONLINE at http://localhost:8002
    echo Backend health check passed at %time% >> logs\backend_startup.log
) else (
    echo ❌ Backend: FAILED TO START
    echo Backend failed at %time% >> logs\startup_errors.log
    echo.
    echo 🔍 Checking backend logs for errors...
    findstr /i "error\|exception\|failed" logs\backend_startup.log
    echo.
    pause
    exit /b 1
)

echo.

REM ============================================================================
REM                          START FRONTEND SERVER
REM ============================================================================

echo 📊 Starting React Dashboard (Frontend - Port 3003)...
echo Starting frontend at %time% > logs\frontend_startup.log

REM Navigate to frontend directory
cd trading-dashboard

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    npm install >> ..\logs\frontend_startup.log 2>>&1
)

REM Start frontend in minimized window with logging
start /min "React Dashboard Frontend" cmd /c "npm run dev >> ..\logs\frontend_startup.log 2>>&1"

REM Return to main directory
cd ..

echo ⏳ Waiting for frontend initialization (15 seconds)...
timeout /t 15 /nobreak >nul

REM Check if frontend is responding
echo 🔍 Testing frontend accessibility...
curl -s --connect-timeout 5 http://localhost:3003 > logs\frontend_health.html 2>>logs\startup_errors.log

if %errorlevel% equ 0 (
    echo ✅ Frontend: ONLINE at http://localhost:3003
    echo Frontend health check passed at %time% >> logs\frontend_startup.log
) else (
    echo ⚠️ Frontend: Starting (may need more time)
    echo Frontend test at %time% - needs more time >> logs\frontend_startup.log
)

echo.

REM ============================================================================
REM                            SYSTEM VALIDATION
REM ============================================================================

echo 🧪 RUNNING SYSTEM VALIDATION...
echo ==============================

REM Test API endpoints
echo 📡 Testing API endpoints...
curl -s "http://localhost:8002/api/signals/latest?limit=1" > logs\signals_test.json 2>nul
if %errorlevel% equ 0 (
    echo ✅ Signals API: Working
) else (
    echo ⚠️ Signals API: May need initialization time
)

curl -s "http://localhost:8002/api/portfolio/summary" > logs\portfolio_test.json 2>nul
if %errorlevel% equ 0 (
    echo ✅ Portfolio API: Working
) else (
    echo ⚠️ Portfolio API: May need initialization time
)

echo.

REM ============================================================================
REM                              STATUS SUMMARY
REM ============================================================================

echo 📋 STARTUP SUMMARY
echo ==================
echo.
echo 🌐 Frontend Dashboard: http://localhost:3003
echo ⚙️  Backend API:        http://localhost:8002
echo 📖 API Documentation:  http://localhost:8002/docs
echo 🔍 Health Check:       http://localhost:8002/api/health
echo.
echo 📁 LOG FILES:
echo - Backend:     logs\backend_startup.log
echo - Frontend:    logs\frontend_startup.log
echo - Errors:      logs\startup_errors.log
echo.

REM Check for any TypeScript compilation errors
if exist "trading-dashboard\logs" (
    findstr /i "error\|failed" trading-dashboard\*.log > logs\typescript_errors.log 2>nul
    if not %errorlevel% equ 1 (
        echo ⚠️  TypeScript compilation errors detected
        echo Check logs\typescript_errors.log for details
    )
)

echo ✅ AUTOMATED STARTUP COMPLETE!
echo.
echo Opening dashboard in default browser...
timeout /t 3 /nobreak >nul
start http://localhost:3003

echo.
echo 💡 TIP: Keep this window open to monitor the startup process
echo Press any key to view detailed logs or close this window
pause >nul

REM Show recent log entries
echo.
echo 📊 RECENT BACKEND LOG:
echo =====================
tail -n 10 logs\backend_startup.log 2>nul || echo No recent backend logs

echo.
echo 📊 RECENT FRONTEND LOG:
echo ======================
tail -n 10 logs\frontend_startup.log 2>nul || echo No recent frontend logs

echo.
echo Startup script completed at %time%
pause
