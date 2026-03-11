@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title PRIMORDIAL CORE v17.1 - RL TRAINING LOOP

pushd "%~dp0"

echo ======================================================
echo PRIMORDIAL CORE: EGITIM v17.1
echo Mod: Explicit Bridge Async-Compatible Training
echo ======================================================
echo.

set "PYTHONPATH=%~dp0"
set "PYTHON_EXE=%~dp0primordial_venv\Scripts\python.exe"
set "TRAIN_PATH=primordial/training/train_ppo.py"

if not exist "%PYTHON_EXE%" (
    echo [HATA] Sanal ortam bulunamadi.
    pause
    popd
    exit /b 1
)

echo >>> Egitim Motoru Baslatiliyor...
"%PYTHON_EXE%" "%TRAIN_PATH%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [HATA] Egitim dongusu calisma zamaninda coktu.
    pause
)

popd
endlocal
