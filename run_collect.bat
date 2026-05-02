@echo off
cd /d %~dp0

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch CPU...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing other requirements...
pip install -r requirements_collect.txt

echo Starting data collection...
python app\collect_webcam_spoof_data.py

pause