@echo off
REM ============================================================================
REM                    STOP TRADING PLATFORM SERVICES
REM ============================================================================
REM This batch file cleanly stops all trading platform services
REM ============================================================================

color 0C
echo.
echo 🛑 STOPPING TRADING PLATFORM SERVICES
echo ====================================
echo Date: %date%
echo Time: %time%
echo.

REM Set working directory
cd /d "C:\Users\Marsel\OneDrive\Documents\Cyb\Intraday_Trading"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo 🔍 Checking running processes...

REM Check for Python processes (Backend)
tasklist /fi "imagename eq python.exe" 2>nul | findstr /i python.exe >nul
if %errorlevel% equ 0 (
    echo 🐍 Found Python processes - stopping backend...
    taskkill /f /im python.exe
    echo ✅ Python processes terminated
    echo Backend stopped at %time% >> logs\shutdown.log
) else (
    echo ℹ️  No Python processes found
)

echo.

REM Check for Node.js processes (Frontend)
tasklist /fi "imagename eq node.exe" 2>nul | findstr /i node.exe >nul
if %errorlevel% equ 0 (
    echo 🟢 Found Node.js processes - stopping frontend...
    taskkill /f /im node.exe
    echo ✅ Node.js processes terminated
    echo Frontend stopped at %time% >> logs\shutdown.log
) else (
    echo ℹ️  No Node.js processes found
)

echo.

REM Check for NPM processes
tasklist /fi "imagename eq npm.exe" 2>nul | findstr /i npm.exe >nul
if %errorlevel% equ 0 (
    echo 📦 Found NPM processes - stopping...
    taskkill /f /im npm.exe
    echo ✅ NPM processes terminated
) else (
    echo ℹ️  No NPM processes found
)

echo.

REM Wait a moment for processes to fully terminate
echo ⏳ Waiting for processes to fully terminate...
timeout /t 3 /nobreak >nul

echo.
echo 🧹 CLEANUP VERIFICATION
echo ======================

REM Verify no processes are still running
set "processes_found=0"

tasklist /fi "imagename eq python.exe" 2>nul | findstr /i python.exe >nul
if %errorlevel% equ 0 (
    echo ⚠️  Warning: Some Python processes may still be running
    set "processes_found=1"
)

tasklist /fi "imagename eq node.exe" 2>nul | findstr /i node.exe >nul
if %errorlevel% equ 0 (
    echo ⚠️  Warning: Some Node.js processes may still be running  
    set "processes_found=1"
)

if %processes_found% equ 0 (
    echo ✅ All trading platform processes stopped successfully
) else (
    echo ⚠️  Some processes may still be running - check Task Manager if needed
)

echo.

REM Test that ports are freed
echo 🔍 Verifying ports are freed...
netstat -ano | findstr :8002 >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 8002 still in use (Backend may still be running)
) else (
    echo ✅ Port 8002 freed (Backend stopped)
)

netstat -ano | findstr :3003 >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 3003 still in use (Frontend may still be running)
) else (
    echo ✅ Port 3003 freed (Frontend stopped)
)

echo.
echo 📋 SHUTDOWN SUMMARY
echo ==================
echo Platform shutdown completed at %time%
echo All services should now be stopped
echo.
echo 💡 To restart the platform, run: START_TRADING_PLATFORM.bat
echo.

pause
