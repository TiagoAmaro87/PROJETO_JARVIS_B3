import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime

# Paths & Imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
try:
    from check_xp_config import MT5_LOGIN, MT5_PASS, MT5_SERVER
except ImportError:
    pass

def backtest_mini(ticker, df):
    """Backtest for WIN/WDO specific edge."""
    df = df.copy()
    # Strategy: Anchored VWAP + SMA 20
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
    df['sma20'] = df['close'].rolling(20).mean()
    
    # Sig: Price > VWAP and Price > SMA20 (Trend follow)
    df['signal'] = np.where((df['close'] > df['vwap']) & (df['close'] > df['sma20']), 1,
                            np.where((df['close'] < df['vwap']) & (df['close'] < df['sma20']), -1, 0))
    
    df['ret'] = df['close'].pct_change()
    df['strat_ret'] = df['signal'].shift(1) * df['ret']
    
    return df['strat_ret'].sum(), df['strat_ret'].mean() / (df['strat_ret'].std() + 1e-10) * np.sqrt(252 * 24) # M5 factor

def run_mini_edge():
    print("🚀 Conectando ao MT5 da XP para analisar Mini Índice e Mini Dólar...")
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
        print("Erro: Não foi possível conectar ao terminal XP.")
        return
    
    # Try Continuous symbols or current contract
    contracts = ["WIN$", "WDO$"] # Common aliases in XP
    results = []
    
    for symbol in contracts:
        print(f"📊 Analisando {symbol} (M5)...")
        if not mt5.symbol_select(symbol, True):
            # Try specific month contract if continuous fails
            symbol = "WINJ24" if "WIN" in symbol else "WDOJ24" # Mocking current
            if not mt5.symbol_select(symbol, True):
                print(f"Símbolo {symbol} não encontrado.")
                continue
                
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 5000)
        if rates is None or len(rates) == 0:
            print(f"Sem dados para {symbol}")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        ret, sharpe = backtest_mini(symbol, df)
        results.append({"symbol": symbol, "ret": ret, "sharpe": sharpe})
        print(f"✅ {symbol}: Sharpe {sharpe:.2f} | Ret Total {ret*100:.2f}%")

    mt5.shutdown()
    
    # Report back to Brain/Obsidian
    print("\n--- O VERDITO DO MINI ---")
    for res in results:
        print(f"{res['symbol']}: {res['sharpe']:.2f} (Eficácia)")

if __name__ == "__main__":
    run_mini_edge()
