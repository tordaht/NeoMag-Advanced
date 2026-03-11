@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0build_setup.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HATA] Build/setup islemi basarisiz oldu. Kod: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo [TAMAM] Build/setup tamamlandi.
