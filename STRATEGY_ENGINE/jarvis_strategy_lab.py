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

CAPITAL_TESTE = 1000.0  # Trava solicitada pelo usuário
TAXA_B3_PCT = 0.0005

ATIVOS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", 
    "PRIO3.SA", "MGLU3.SA", "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "RADL3.SA", "RAIL3.SA", "SUZB3.SA", "GGBR4.SA", "CSNA3.SA",
    "B3SA3.SA", "HYPE3.SA", "VIVT3.SA", "ELET3.SA", "JBSS3.SA"
]

class JarvisStrategyLab:
    def __init__(self):
        print("🧠 [JARVIS LAB] Inicializando Motores de Estratégia...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    # --- SETUP 1: Deep Learning (Brain Core) ---
    def strategy_dl(self, df_orig):
        df = df_orig.copy()
        df["SMA_9"] = df["Close"].rolling(9).mean()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100
        df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df = df.dropna()
        if df.empty: return -100
        
        X = self.scaler.transform(df[self.features_cols].values)
        probs = self.model.predict(X, verbose=0).flatten()
        
        saldo = CAPITAL_TESTE
        for i in range(len(df)-1):
            sinal = 1 if probs[i] > 0.55 else (-1 if probs[i] < 0.45 else 0)
            if sinal != 0:
                lote = int(saldo / df["Close"].iloc[i])
                if lote > 0:
                    ret = (df["Close"].iloc[i+1] / df["Close"].iloc[i]) - 1
                    taxa = (lote * df["Close"].iloc[i]) * TAXA_B3_PCT
                    saldo += (saldo * (sinal * ret)) - taxa
        return (saldo / CAPITAL_TESTE - 1) * 100

    # --- SETUP 2: Momentum Breakout (Trend) ---
    def strategy_momentum(self, df_orig):
        df = df_orig.copy()
        df['Max_10'] = df['High'].rolling(10).max().shift(1)
        df['Min_10'] = df['Low'].rolling(10).min().shift(1)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        saldo = CAPITAL_TESTE
        pos = 0; p_in = 0
        for i in range(50, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0 and close > df['Max_10'].iloc[i] and close > df['SMA_50'].iloc[i]:
                pos = 1; p_in = close
            elif pos == 1 and close < df['Min_10'].iloc[i]:
                ret = (close / p_in) - 1
                saldo += (saldo * ret) - (CAPITAL_TESTE * TAXA_B3_PCT * 2)
                pos = 0
        return (saldo / CAPITAL_TESTE - 1) * 100

    # --- SETUP 3: Mean Reversion (RSI + BB) ---
    def strategy_reversion(self, df_orig):
        df = df_orig.copy()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["std"] = df["Close"].rolling(20).std()
        df["BB_Lower"] = df["SMA_20"] - (2 * df["std"])
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        
        saldo = CAPITAL_TESTE
        pos = 0; p_in = 0
        for i in range(20, len(df)):
            close = df['Close'].iloc[i]
            # Compra: Preço abaixo da banda inferior + RSI sobrevendido
            if pos == 0 and close < df['BB_Lower'].iloc[i] and df['RSI'].iloc[i] < 30:
                pos = 1; p_in = close
            # Venda: Preço voltou à média
            elif pos == 1 and close > df['SMA_20'].iloc[i]:
                ret = (close / p_in) - 1
                saldo += (saldo * ret) - (CAPITAL_TESTE * TAXA_B3_PCT * 2)
                pos = 0
        return (saldo / CAPITAL_TESTE - 1) * 100

    def run_all(self):
        print(f"\n📊 [ULTIMATE BACKTEST] Capital: R$ {CAPITAL_TESTE} | Período: 1 Ano")
        print(f"{'ATIVO':<10} | {'DL (%)':<10} | {'MOM (%)':<10} | {'REV (%)':<10} | {'VENCEDOR'}")
        print("-" * 75)
        
        final_map = {}
        for ticker in ATIVOS:
            df = self.get_data(ticker)
            if df is None: continue
            
            res_dl = self.strategy_dl(df)
            res_mom = self.strategy_momentum(df)
            res_rev = self.strategy_reversion(df)
            
            scores = {"DeepLearning": res_dl, "Momentum": res_mom, "Reversion": res_rev}
            winner = max(scores, key=scores.get)
            
            print(f"{ticker:<10} | {res_dl:>8.1f}% | {res_mom:>8.1f}% | {res_rev:>8.1f}% | {winner}")
            final_map[ticker] = winner
            
        return final_map

if __name__ == "__main__":
    lab = JarvisStrategyLab()
    mapping = lab.run_all()
    print("\n✅ Mapeamento para o Robô Híbrido Finalizado.")
