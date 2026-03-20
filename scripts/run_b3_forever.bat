@echo off
title JARVIS_B3 - Brain V2 (DRY RUN)
cd /d E:\PROJETO_JARVIS_B3

echo ============================================
echo   JARVIS_B3 - Watchlist: MGLU3 B3SA3 VALE3 ABEV3 RENT3
echo   Mode: DRY_RUN (logs trades, no real orders)
echo   Market hours: 10:00 - 17:00 BRT
echo ============================================

:loop
echo [%date% %time%] Starting JARVIS_B3...
py "BRAIN (DL)\brain_engine_v2.py"
echo [%date% %time%] Stopped. Restarting in 10s...
timeout /t 10
goto loop
