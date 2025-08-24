@echo off
REM Generic Ollama Agent Launcher for Windows
echo Starting Generic Ollama Agent...

REM Try to find and activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist "ollama-agents\Scripts\activate.bat" (
    echo Activating virtual environment (ollama-agents)...
    call ollama-agents\Scripts\activate.bat
) else (
    echo No Windows virtual environment found, using system Python
)

echo Launching Generic Ollama Agent...
python main.py

pause
