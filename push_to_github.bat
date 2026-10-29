@echo off
echo ====================================================================
echo Pushing OptiQueue AI to GitHub (https://github.com/ishant212)
echo ====================================================================
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================================
    echo [SUCCESS] Project successfully pushed to GitHub!
    echo URL: https://github.com/ishant212/Retail-Checkout-Queue-Optimizer-RL-Project
    echo ====================================================================
) else (
    echo.
    echo [NOTE] Failed to push. Please verify your credentials and network connection.
)

pause
