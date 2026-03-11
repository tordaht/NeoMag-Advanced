@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title PRIMORDIAL CORE v12.5 - HIZLI SIMULASYON

pushd "%~dp0"

echo ======================================================
echo PRIMORDIAL CORE: SIMULASYON v12.7.0 [ABSOLUTE ZERO-COPY]
echo Not: Simulasyon Gozlemevi Uzerinden Yonetilir.
echo ======================================================
echo.

set "PYTHONPATH=%~dp0"
set "PYTHON_EXE=%~dp0primordial_venv\Scripts\python.exe"
set "APP_PATH=primordial/apps/observatory/app.py"

if not exist "%PYTHON_EXE%" (
    echo [HATA] Sanal ortam bulunamadi.
    pause
    popd
    exit /b 1
)

:: Gozlemevi'ni baslat (v12.5 Standart)
"%PYTHON_EXE%" "%APP_PATH%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [HATA] Simulasyon motoru baslatilamadi.
    pause
)

popd
endlocal
