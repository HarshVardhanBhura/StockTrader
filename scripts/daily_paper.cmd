@echo off
REM Daily paper-trading driver for both accounts.
REM
REM Registered as a Windows scheduled task, weekdays at 19:30 IST - after NSE
REM publishes the bhavcopy. Both accounts stalled for five days once because
REM nothing ran them; that is what this exists to prevent.
REM
REM --catchup means a missed day is not a lost day. If the machine is asleep
REM at 19:30 the task runs at the next opportunity and steps through every
REM trading day it missed, in order.
REM
REM The second account passes --no-ingest because the first already pulled
REM today's bhavcopy into the shared store.
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

"%PY%" -u scripts\paper_trade.py --catchup >> "%LOG%" 2>&1
set RC1=%errorlevel%

"%PY%" -u scripts\paper_trade_band.py --catchup --no-ingest >> "%LOG%" 2>&1
set RC2=%errorlevel%

echo exit codes: top60=%RC1% band61_300=%RC2% >> "%LOG%"

if not "%RC1%"=="0" exit /b %RC1%
exit /b %RC2%
