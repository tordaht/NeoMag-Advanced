@echo off
cd /d "%~dp0"
title PRIMORDIAL CORE v17.1 - SYNTHETIC LIFE OBSERVATORY
echo ========================================================
echo PRIMORDIAL CORE: GOZLEMEVI v17.1
echo Durum: Explicit Bridge ^| Async Worker Ready
echo ========================================================
echo.
echo [SISTEM] Sanal ortam yukleniyor...
if not exist "%~dp0primordial_venv\Scripts\python.exe" (
    echo [HATA] primordial_venv bulunamadi!
    pause
    exit /b
)
echo [SISTEM] Uygulama baslatiliyor (v17.1)...
"%~dp0primordial_venv\Scripts\python.exe" "%~dp0primordial\apps\observatory\app.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HATA] Uygulama beklenmedik bir sekilde kapandi. (Error Code: %ERRORLEVEL%)
    pause
)
exit
