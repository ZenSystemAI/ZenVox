@echo off
echo ========================================
echo  Building ZenVox standalone app...
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Cleaning previous build...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo [2/3] Running PyInstaller...
pyinstaller zenvox.spec --noconfirm

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check errors above.
    pause
    exit /b 1
)

echo [3/3] Copying icon...
copy /y whisper.ico "dist\ZenVox\whisper.ico" >nul 2>&1

echo.
echo ========================================
echo  BUILD COMPLETE
echo  Output: dist\ZenVox\ZenVox.exe
echo ========================================
echo.
echo First run will download the Whisper model (~1.5GB).
echo.
pause
