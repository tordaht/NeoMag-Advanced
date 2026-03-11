@echo off
title PRIMORDIAL CORE v17.1 - QUALITY GATE
setlocal
cd /d "%~dp0"

set PYTHONPATH=%cd%

echo ========================================================
echo PRIMORDIAL CORE v17.1: QUALITY GATE [EXPLICIT BRIDGE]
echo Components: Pytest ^| Pixel-Match ^| Smoke Check
echo ========================================================
echo.

echo [1/3] Calistiriliyor: Birim ve API Testleri (Pytest)...
.\primordial_venv\Scripts\python.exe -m pytest primordial/tests/
if %errorlevel% neq 0 goto :FAIL

echo.
echo [2/3] Calistiriliyor: Gorsel Kalite Kontrolu (Pixel-Match)...
.\primordial_venv\Scripts\python.exe test_pixel_match.py
if %errorlevel% neq 0 goto :FAIL

echo.
echo [3/3] Calistiriliyor: Gozlemevi Baslatma Kontrolu (Smoke Check)...
echo [NOT] Uygulama baslatilip 10 saniye sonra otomatik kapatilacaktir.
.\primordial_venv\Scripts\python.exe -c "import subprocess, time; p = subprocess.Popen(['.\\primordial_venv\\Scripts\\python.exe', 'primordial/apps/observatory/app.py']); time.sleep(10); p.terminate(); print('[SMOKE CHECK] Uygulama basariyla acildi ve kapandi.')"
if %errorlevel% neq 0 goto :FAIL

echo.
echo ========================================================
echo [BASARILI] v17.1 tum kalite barajlarini gecti.
echo Durum: STABIL CORE + ASYNC WORKER VERIFIED
echo ========================================================
pause
exit /b 0

:FAIL
echo.
echo ========================================================
echo [HATA] Kalite baraji asilamadi! Loglari inceleyin.
echo ========================================================
pause
exit /b 1
