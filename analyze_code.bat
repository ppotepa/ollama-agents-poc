# Simple batch file wrapper for Windows Command Prompt users
@echo off
echo Ollama Agents - Code Quality Analysis
echo.

REM Check if PowerShell is available
powershell -Command "exit 0" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: PowerShell is required to run this analysis tool
    echo Please use PowerShell or install it from Microsoft
    pause
    exit /b 1
)

REM Check for parameters and run PowerShell script
if "%1"=="--help" (
    powershell -ExecutionPolicy Bypass -File analyze_code.ps1 -Help
) else if "%1"=="--quick" (
    powershell -ExecutionPolicy Bypass -File analyze_code.ps1 -Quick
) else if "%1"=="--full" (
    powershell -ExecutionPolicy Bypass -File analyze_code.ps1 -Full
) else if "%1"=="--coverage" (
    powershell -ExecutionPolicy Bypass -File analyze_code.ps1 -Coverage
) else (
    echo Usage: analyze_code.bat [--quick] [--full] [--coverage] [--help]
    echo.
    echo Running quick analysis by default...
    echo.
    powershell -ExecutionPolicy Bypass -File analyze_code.ps1 -Quick
)

pause
