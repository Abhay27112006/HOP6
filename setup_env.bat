@echo off
echo =========================================
echo Setting up Hop6 Engine Environment...
echo =========================================

set VENV_DIR=.venv

IF NOT EXIST "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
) ELSE (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers networkx questionary rich huggingface_hub accelerate safetensors sentencepiece psutil

echo.
echo =========================================
echo Setup Complete! 
echo You can now run Hop6_Launcher.exe or start via cli.py
echo =========================================
pause
