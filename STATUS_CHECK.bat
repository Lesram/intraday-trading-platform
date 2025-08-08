@echo off
REM ============================================================================
REM                      TRADING PLATFORM STATUS CHECKER
REM ============================================================================
REM This batch file checks the current status of all trading platform services
REM ============================================================================

color 0B
echo.
echo 📊 TRADING PLATFORM STATUS CHECK
echo ================================
echo Date: %date%
echo Time: %time%
echo.

REM Set working directory
cd /d "C:\Users\Marsel\OneDrive\Documents\Cyb\Intraday_Trading"

echo 🔍 PROCESS STATUS
echo ================

REM Check Python processes (Backend)
tasklist /fi "imagename eq python.exe" 2>nul | findstr /i python.exe >nul
if %errorlevel% equ 0 (
    echo ✅ Backend (Python): RUNNING
    tasklist /fi "imagename eq python.exe" | findstr python.exe
) else (
    echo ❌ Backend (Python): NOT RUNNING
)

echo.

REM Check Node.js processes (Frontend)
tasklist /fi "imagename eq node.exe" 2>nul | findstr /i node.exe >nul
if %errorlevel% equ 0 (
    echo ✅ Frontend (Node.js): RUNNING
    tasklist /fi "imagename eq node.exe" | findstr node.exe
) else (
    echo ❌ Frontend (Node.js): NOT RUNNING
)

echo.
echo 🌐 SERVICE CONNECTIVITY
echo ======================

REM Test Backend API
echo Testing Backend API (Port 8002)...
curl -s --connect-timeout 3 http://localhost:8002/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend API: ACCESSIBLE at http://localhost:8002
    echo ✅ API Documentation: http://localhost:8002/docs
) else (
    echo ❌ Backend API: NOT ACCESSIBLE
)

echo.

REM Test Frontend
echo Testing Frontend (Port 3003)...
curl -s --connect-timeout 3 http://localhost:3003 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend: ACCESSIBLE at http://localhost:3003
) else (
    echo ❌ Frontend: NOT ACCESSIBLE
)

echo.
echo 🔌 PORT STATUS
echo =============

REM Check port usage
netstat -ano | findstr :8002 >nul
if %errorlevel% equ 0 (
    echo ✅ Port 8002 (Backend): IN USE
    netstat -ano | findstr :8002
) else (
    echo ❌ Port 8002 (Backend): FREE
)

echo.

netstat -ano | findstr :3003 >nul
if %errorlevel% equ 0 (
    echo ✅ Port 3003 (Frontend): IN USE
    netstat -ano | findstr :3003
) else (
    echo ❌ Port 3003 (Frontend): FREE
)

echo.
echo 📋 QUICK ACCESS LINKS
echo ====================
echo 🌐 Dashboard:     http://localhost:3003
echo ⚙️  API:          http://localhost:8002
echo 📖 API Docs:     http://localhost:8002/docs
echo 🔍 Health:       http://localhost:8002/api/health

echo.
echo 📁 LOG FILES (if available)
echo ==========================
if exist "logs\backend_startup.log" (
    echo ✅ Backend logs: logs\backend_startup.log
) else (
    echo ❌ No backend logs found
)

if exist "logs\frontend_startup.log" (
    echo ✅ Frontend logs: logs\frontend_startup.log
) else (
    echo ❌ No frontend logs found
)

echo.
echo 💡 ACTIONS AVAILABLE:
echo - To start services: START_TRADING_PLATFORM.bat
echo - To stop services:  STOP_TRADING_PLATFORM.bat
echo - To check status:   STATUS_CHECK.bat (this file)
echo.

pause
