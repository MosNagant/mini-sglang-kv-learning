@echo off
title WSL 2 Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-WSL.ps1"
echo.
echo If a restart is required, restart Windows and run this file as administrator again.
echo Logs are saved in the logs folder beside this file.
pause
