@echo off
echo ========================================================
echo Pushing DeepLure Saree Recognition Project to GitHub...
echo ========================================================
echo.
cd /d "%~dp0"
"..\.git_bin\cmd\git.exe" push -u origin main
echo.
echo ========================================================
echo Done! Check your repo at:
echo https://github.com/riteeshkumargudipudi/color-invariant-saree-recognition
echo ========================================================
pause
