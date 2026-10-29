@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python scripts\download_data.py
if errorlevel 1 goto :error
python src\train_model.py
if errorlevel 1 goto :error
python -m streamlit run app.py
goto :end
:error
echo.
echo A command failed. Read the error above, then check README.md.
pause
:end
endlocal
