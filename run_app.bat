@echo off
echo Starting Brain Tumor Detection App...
cd /d "%~dp0"
python -m streamlit run app.py --server.headless true
pause