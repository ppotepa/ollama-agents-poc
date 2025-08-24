@echo off
REM Generic Ollama Agent - DeepCoder Model Launcher
echo Starting Generic Ollama Agent with DeepCoder 14B...

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

echo Launching with DeepCoder 14B model...
python main.py --model deepcoder

pause
