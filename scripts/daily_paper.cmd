@echo off
REM Daily driver for every paper-trading account.
REM
REM Registered as a Windows scheduled task, weekdays at 19:30 IST - after NSE
REM publishes the bhavcopy. Both accounts stalled for five days once because
REM nothing ran them; that is what this exists to prevent.
REM
REM One runner over the whole registry (backend/live/accounts.py), not a
REM script per account. Six accounts running slightly different code could not
REM be compared, and comparison is the only reason there is more than one.
REM
REM --catchup means a missed day is not a lost day. If the machine is asleep
REM at 19:30 the task runs at the next opportunity and steps through every
REM trading day it missed, in order.
REM
REM Deliberately does NOT commit or push. Account state is written to disk and
REM committed by hand, so an automated job never publishes anything on its own.

setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
set "LOG=data\paper\daily.log"

if not exist "%PY%" (
    echo cannot find %PY% - is the venv present?
    exit /b 9
)

echo.>> "%LOG%"
echo ======================================================== >> "%LOG%"
echo RUN %date% %time% >> "%LOG%"
echo ======================================================== >> "%LOG%"

"%PY%" -u scripts\paper_accounts.py --catchup >> "%LOG%" 2>&1
set RC=%errorlevel%

echo exit code: %RC% >> "%LOG%"
exit /b %RC%
