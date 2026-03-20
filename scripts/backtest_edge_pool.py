import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import importlib.util

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import Brain logic
spec = importlib.util.spec_from_file_location("brain_engine_v2", os.path.join(BASE_DIR, "BRAIN (DL)", "brain_engine_v2.py"))
brain_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brain_mod)
compute_features = brain_mod.compute_features

# TOP 40 B3 STOCKS
POOL_40 = [
    "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "B3SA3", "ITSA4", "JBSS3", "SUZB3",
    "RENT3", "MGLU3", "WEGE3", "RADL3", "RAIL3", "ELET3", "GGBR4", "EQTL3", "LREN3", "VIVT3",
    "CSNA3", "CPLE6", "VBBR3", "HYPE3", "RDOR3", "SBSP3", "EMBR3", "BRFS3", "BRAP4", "TOTS3",
    "CIEL3", "USIM5", "GOAU4", "ENGI11", "BPAC11", "BEEF3", "MRFG3", "CYRE3", "MULT3", "AZUL4"
]

def backtest_strategy(ticker, df, strat_name):
    """Simplified backtest for a single strategy."""
    df = df.copy()
    if strat_name == "TREND":
        df['ema200'] = df['Close'].ewm(span=200).mean()
        df['ema20'] = df['Close'].ewm(span=20).mean()
        df['signal'] = np.where((df['Close'] > df['ema200']) & (df['Close'] > df['ema20']), 1, 
                                np.where((df['Close'] < df['ema200']) & (df['Close'] < df['ema20']), -1, 0))
    elif strat_name == "REVERSION":
        df['ma20'] = df['Close'].rolling(20).mean()
        df['std'] = df['Close'].rolling(20).std()
        df['signal'] = np.where(df['Close'] < df['ma20'] - 2*df['std'], 1,
                                np.where(df['Close'] > df['ma20'] + 2*df['std'], -1, 0))
    elif strat_name == "BREAKOUT":
        df['high10'] = df['High'].rolling(10).max()
        df['low10'] = df['Low'].rolling(10).min()
        df['signal'] = np.where(df['Close'] > df['high10'].shift(1), 1,
                                np.where(df['Close'] < df['low10'].shift(1), -1, 0))
    
    # Simple Returns
    df['ret'] = df['Close'].pct_change()
    df['strat_ret'] = df['signal'].shift(1) * df['ret']
    
    total_ret = df['strat_ret'].sum()
    sharpe = df['strat_ret'].mean() / (df['strat_ret'].std() + 1e-10) * np.sqrt(252)
    return total_ret, sharpe

def run_pool_backtest():
    results = []
    print(f"\n🚀 Iniciando Varredura de Backtest em 40 Papéis...")
    
    for i, ticker in enumerate(POOL_40):
        print(f"[{i+1}/40] Analisando {ticker}...")
        try:
            df = yf.download(f"{ticker}.SA", period="2y", interval="1d", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Test all 3
            best_strat = "TREND"
            best_sharpe = -999
            
            for strat in ["TREND", "REVERSION", "BREAKOUT"]:
                ret, sharpe = backtest_strategy(ticker, df, strat)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_strat = strat
            
            results.append({"ticker": ticker, "best_strat": best_strat, "sharpe": best_sharpe})
        except Exception as e:
            pass

    res_df = pd.DataFrame(results)
    print("\n✅ Backtest Concluído!")
    
    # Generate STRAT_MAP code
    print("\n--- NOVO STRAT_MAP (Copie para o brain_engine_v2.py) ---")
    mapping = {row['ticker']: row['best_strat'] for _, row in res_df.iterrows()}
    print("STRAT_MAP = " + str(mapping))
    print("----------------------------------------------------------")
    
    # Save to CSV for reference
    res_df.to_csv(os.path.join(BASE_DIR, "backtest_results_pool.csv"), index=False)

if __name__ == "__main__":
    run_pool_backtest()
