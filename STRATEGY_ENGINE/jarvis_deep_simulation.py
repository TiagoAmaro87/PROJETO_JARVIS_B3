import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

CAPITAL_INICIAL = 1000.0  # Foco total em R$ 1000
TAXA_B3_PCT = 0.0005

# Lista completa de papéis solicitada
ATIVOS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", 
    "PRIO3.SA", "MGLU3.SA", "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "RADL3.SA", "RAIL3.SA", "SUZB3.SA", "GGBR4.SA", "CSNA3.SA",
    "B3SA3.SA", "HYPE3.SA", "VIVT3.SA", "ELET3.SA", "JBSS3.SA",
    "GOAU4.SA", "CPLE6.SA", "EQTL3.SA", "LREN3.SA"
]

class JarvisDeepSimulation:
    def __init__(self):
        print("🧠 [JARVIS DEEP SIM] Carregando cérebro para simulação histórica...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def run_dl(self, df_orig):
        df = df_orig.copy()
        # Features
        df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
        df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df = df.dropna()
        if df.empty: return -100, 0, 0
        
        X = self.scaler.transform(df[self.features_cols].values)
        probs = self.model.predict(X, verbose=0).flatten()
        
        saldo = CAPITAL_INICIAL
        trades = 0
        for i in range(len(df)-1):
            sinal = 1 if probs[i] > 0.55 else (-1 if probs[i] < 0.45 else 0)
            if sinal != 0:
                lote = int(saldo / df["Close"].iloc[i])
                if lote > 0:
                    ret = (df["Close"].iloc[i+1] / df["Close"].iloc[i]) - 1
                    taxa = (lote * df["Close"].iloc[i]) * TAXA_B3_PCT
                    saldo += (saldo * (sinal * ret)) - taxa
                    trades += 1
        return (saldo / CAPITAL_INICIAL - 1) * 100, trades, saldo

    def run_momentum(self, df_orig):
        df = df_orig.copy()
        df['Max_10'] = df['High'].rolling(10).max().shift(1); df['Min_10'] = df['Low'].rolling(10).min().shift(1); df['SMA_50'] = df['Close'].rolling(50).mean()
        saldo = CAPITAL_INICIAL; pos = 0; p_in = 0; trades = 0
        for i in range(50, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0 and close > df['Max_10'].iloc[i] and close > df['SMA_50'].iloc[i]:
                pos = 1; p_in = close; trades += 1
            elif pos == 1 and close < df['Min_10'].iloc[i]:
                ret = (close / p_in) - 1
                saldo += (saldo * ret) - (CAPITAL_INICIAL * TAXA_B3_PCT * 2)
                pos = 0
        return (saldo / CAPITAL_INICIAL - 1) * 100, trades, saldo

    def run_reversion(self, df_orig):
        df = df_orig.copy()
        df["SMA_20"] = df["Close"].rolling(20).mean(); df["std"] = df["Close"].rolling(20).std(); df["BB_Lower"] = df["SMA_20"] - (2 * df["std"])
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        saldo = CAPITAL_INICIAL; pos = 0; p_in = 0; trades = 0
        for i in range(20, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0 and close < df['BB_Lower'].iloc[i] and df['RSI'].iloc[i] < 30:
                pos = 1; p_in = close; trades += 1
            elif pos == 1 and close > df['SMA_20'].iloc[i]:
                ret = (close / p_in) - 1
                saldo += (saldo * ret) - (CAPITAL_INICIAL * TAXA_B3_PCT * 2)
                pos = 0
        return (saldo / CAPITAL_INICIAL - 1) * 100, trades, saldo

    def run_full_sim(self):
        start_t = datetime.now()
        print(f"\n{'='*100}")
        print(f"📈 JARVIS DEEP SIMULATION - CAPITAL INICIAL: R$ {CAPITAL_INICIAL:,.2f}")
        print(f"{'='*100}")
        print(f"{'ATIVO':<12} | {'ESTRATÉGIA':<15} | {'P&L %':<10} | {'TRADES':<8} | {'SALDO FINAL'}")
        print("-" * 100)
        
        relatorio = []
        for ticker in ATIVOS:
            df = self.get_data(ticker)
            if df is None: continue
            
            # Testa as 3 e pega a melhor
            dl_p, dl_t, dl_s = self.run_dl(df)
            mom_p, mom_t, mom_s = self.run_momentum(df)
            rev_p, rev_t, rev_s = self.run_reversion(df)
            
            res_list = [
                {"name": "DeepLearning", "p": dl_p, "t": dl_t, "s": dl_s},
                {"name": "Momentum", "p": mom_p, "t": mom_t, "s": mom_s},
                {"name": "Reversion", "p": rev_p, "t": rev_t, "s": rev_s}
            ]
            
            best = max(res_list, key=lambda x: x['p'])
            
            print(f"{ticker:<12} | {best['name']:<15} | {best['p']:>8.2f}% | {best['t']:>8} | R$ {best['s']:,.2f}")
            relatorio.append({"Ativo": ticker, **best})
            
        end_t = datetime.now()
        tot_final = sum([x['s'] for x in relatorio])
        tot_investido = CAPITAL_INICIAL * len(relatorio)
        
        print(f"\n{'='*100}")
        print(f"🏁 CONCLUSÃO DA SIMULAÇÃO")
        print(f"{'='*100}")
        print(f"⌛ Tempo Processamento: {end_t - start_t}")
        print(f"💰 Investimento Total Virt.: R$ {tot_investido:,.2f}")
        print(f"🚀 Patrimônio Final Virt.:   R$ {tot_final:,.2f}")
        print(f"📈 Rentabilidade Média:      {((tot_final/tot_investido)-1)*100:.2f}%")
        print(f"{'='*100}")

if __name__ == "__main__":
    sim = JarvisDeepSimulation()
    sim.run_full_sim()
