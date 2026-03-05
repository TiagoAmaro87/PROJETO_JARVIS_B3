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

# --- CONFIGURAÇÃO GLOBAL ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

CAPITAL_INICIAL = 10000.0
TAXA_B3_PCT = 0.0005

ATIVOS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", 
    "PRIO3.SA", "MGLU3.SA", "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "RADL3.SA", "RAIL3.SA", "SUZB3.SA", "GGBR4.SA", "CSNA3.SA"
]

class JarvisMassTester:
    def __init__(self):
        print("🧠 [JARVIS UNIFIED] Carregando Modelos para Teste Massivo...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def run_strategy_dl(self, df_orig):
        df = df_orig.copy()
        # Prepara Features
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
        
        if df.empty: return 0, 0
        
        X = self.scaler.transform(df[self.features_cols].values)
        probs = self.model.predict(X, verbose=0).flatten()
        
        saldo = CAPITAL_INICIAL
        trades = 0
        for i in range(len(df)-1):
            sinal = 1 if probs[i] > 0.55 else (-1 if probs[i] < 0.45 else 0)
            if sinal != 0:
                lote = int(saldo / df["Close"].iloc[i])
                if lote > 0:
                    retorno = (df["Close"].iloc[i+1] / df["Close"].iloc[i]) - 1
                    taxa = (lote * df["Close"].iloc[i]) * TAXA_B3_PCT
                    # Ganho = sinal * retorno
                    saldo += (saldo * (sinal * retorno)) - taxa
                    trades += 1
        
        lucro_pct = (saldo / CAPITAL_INICIAL - 1) * 100
        return round(lucro_pct, 2), trades

    def run_strategy_momentum(self, df_orig):
        df = df_orig.copy()
        df['Max_10'] = df['High'].rolling(10).max().shift(1)
        df['Min_10'] = df['Low'].rolling(10).min().shift(1)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        saldo = CAPITAL_INICIAL
        pos = 0
        p_in = 0
        trades = 0
        
        for i in range(50, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0:
                if close > df['Max_10'].iloc[i] and close > df['SMA_50'].iloc[i]:
                    pos = 1
                    p_in = close
                    trades += 1
            elif pos == 1:
                if close < df['Min_10'].iloc[i]:
                    ret = (close / p_in) - 1
                    taxa = (CAPITAL_INICIAL * TAXA_B3_PCT) * 2 # Compra e Venda
                    saldo += (saldo * ret) - taxa
                    pos = 0
        
        lucro_pct = (saldo / CAPITAL_INICIAL - 1) * 100
        return round(lucro_pct, 2), trades

    def execute_all(self):
        print("\n" + "="*80)
        print(f"📊 JARVIS UNIFIED MASS BACKTEST - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        print("="*80)
        print(f"{'ATIVO':<12} | {'DL (%)':<10} | {'MOM (%)':<10} | {'VENCEDOR':<10}")
        print("-" * 80)
        
        results = []
        for ticker in ATIVOS:
            df = self.get_data(ticker)
            if df is None: continue
            
            dl_res, dl_trades = self.run_strategy_dl(df)
            mom_res, mom_trades = self.run_strategy_momentum(df)
            
            winner = "DeepLearning" if dl_res > mom_res else "Momentum"
            if dl_res == mom_res: winner = "Empate"
            
            print(f"{ticker:<12} | {dl_res:>9}% | {mom_res:>9}% | {winner:<10}")
            results.append({
                "Ativo": ticker,
                "DL_Profit_%": dl_res,
                "MOM_Profit_%": mom_res,
                "Winner": winner
            })
            
        print("="*80)
        df_res = pd.DataFrame(results)
        dl_wins = len(df_res[df_res['Winner'] == 'DeepLearning'])
        mom_wins = len(df_res[df_res['Winner'] == 'Momentum'])
        
        print(f"\n🏆 PLACAR FINAL: Deep Learning {dl_wins} vs {mom_wins} Momentum")
        print(f"📈 Média DL: {df_res['DL_Profit_%'].mean():.2f}%")
        print(f"📈 Média MOM: {df_res['MOM_Profit_%'].mean():.2f}%")
        print("="*80)

if __name__ == "__main__":
    tester = JarvisMassTester()
    tester.execute_all()
