@echo off
echo ------------------------------------------
echo       DIO Demonstration Launcher
echo ------------------------------------------

echo [1/3] Cleaning up previous processes...
taskkill /F /IM dio-manager.exe 2>NUL
taskkill /F /IM mock-worker.exe 2>NUL
taskkill /F /IM demo.exe 2>NUL

echo [2/3] Setting environment variables...
set "DIO_ROOT=%CD%\.."
echo DIO_ROOT is set to: %DIO_ROOT%

echo [3/3] Starting demo server...
echo.
echo Open http://localhost:9090 in your browser.
echo.
.\demo.exe
pause
