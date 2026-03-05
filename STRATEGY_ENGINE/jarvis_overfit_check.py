import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

class JarvisOverfitAuditor:
    def __init__(self):
        print("🔍 [AUDITORIA] Iniciando Teste de Robustez do Modelo...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker, start_date, end_date):
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def prepare_data(self, df):
        df = df.copy()
        df["SMA_9"] = df["Close"].rolling(9).mean()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100
        df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        
        # Target: 1 se o preço subir no dia seguinte, 0 se cair
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()
        return df

    def analyze(self, ticker):
        print(f"\n--- Analisando Ativo: {ticker} ---")
        
        # 1. Dados In-Sample (Onde o modelo provavelmente treinou - 2023)
        df_train = self.get_data(ticker, "2023-01-01", "2023-12-31")
        # 2. Dados Out-of-Sample (Onde o modelo NUNCA viu - Jan/Fev 2026)
        df_test = self.get_data(ticker, "2026-01-01", "2026-03-04")
        
        results = []
        for name, data in [("In-Sample (2023)", df_train), ("Out-Sample (2026)", df_test)]:
            if data is None or len(data) < 20: continue
            df = self.prepare_data(data)
            X = self.scaler.transform(df[self.features_cols].values)
            y_true = df["Target"].values
            y_prob = self.model.predict(X, verbose=0).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            
            results.append({"Dataset": name, "Accuracy": acc, "Precision": prec})
        
        return results

    def run_audit(self):
        tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
        all_results = []
        for t in tickers:
            all_results.append(self.analyze(t))
        
        print(f"\n{'='*60}")
        print(f"📊 RELATÓRIO DE AUDITORIA (OVERFITTING CHECK)")
        print(f"{'='*60}")
        for i, t in enumerate(tickers):
            print(f"\nAtivo: {t}")
            for res in all_results[i]:
                print(f"  > {res['Dataset']:<18} | Acc: {res['Accuracy']:.2%} | Prec: {res['Precision']:.2%}")
        
        # Lógica de Diagnóstico
        # Se a Acc Training for >> Acc Test, é Overfitting.
        # Se ambas forem baixas (perto de 50%), é Underfitting (Aleatório).
        print(f"\n{'='*60}")
        print("🧠 DIAGNÓSTICO DO JARVIS:")
        # Exemplo baseado no primeiro ativo
        in_acc = all_results[0][0]['Accuracy']
        out_acc = all_results[0][1]['Accuracy']
        diff = in_acc - out_acc
        
        if diff > 0.15:
            print("⚠️ AVISO: Overfitting Detectado! O modelo decorou o passado e erra no presente.")
        elif in_acc < 0.53 and out_acc < 0.53:
            print("📉 AVISO: Underfitting Detectado! O modelo não está captando padrões úteis.")
        else:
            print("✅ ROBUSTEZ CONFIRMADA: O modelo mantém performance consistente em dados novos.")
        print('='*60)

if __name__ == "__main__":
    auditor = JarvisOverfitAuditor()
    auditor.run_audit()
