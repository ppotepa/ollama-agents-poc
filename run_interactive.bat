@echo off
REM Generic Ollama Agent - Interactive Model Selection
echo Starting Generic Ollama Agent with interactive model selection...

REM Try to find and activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

echo Choose your AI model...
python main.py

pause
