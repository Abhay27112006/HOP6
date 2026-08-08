@echo off
echo Setting up Hop6 Engine...
if exist .venv (
    call .venv\Scripts\activate
) else if exist venv (
    call venv\Scripts\activate
) else (
    echo Error: Virtual environment not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)
pip install -r requirements.txt
python src\cli.py
pause
