@echo off
title JARVIS_B3 - Brain V2 (LIVE MODE)
cd /d E:\PROJETO_JARVIS_B3

echo ============================================
echo   JARVIS_B3 - Watchlist: MGLU3 B3SA3 VALE3 ABEV3 RENT3
echo   Mode: LIVE (orders will be sent to MT5)
echo   Market hours: 10:00 - 17:00 BRT
echo ============================================

:loop
echo [%date% %time%] Starting JARVIS_B3 in LIVE mode...
py "BRAIN (DL)\brain_engine_v2.py" --live
echo [%date% %time%] Stopped. Restarting in 10s...
timeout /t 10
goto loop
