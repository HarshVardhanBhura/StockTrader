@echo off
REM Weekly health report on the paper accounts.
REM
REM Evaluates the five stop conditions and the behaviour checks specified in
REM docs/DECISION_RULE.md, which were prose that nothing read until this
REM existed. Exits 2 when a stop condition triggers, so the scheduler records
REM a failure rather than a silent success.

setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" ( echo cannot find %PY% & exit /b 9 )
"%PY%" -u scripts\weekly_report.py --save >> "data\paper\weekly.log" 2>&1
exit /b %errorlevel%
