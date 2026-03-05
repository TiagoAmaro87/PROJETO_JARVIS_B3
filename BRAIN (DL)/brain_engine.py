import os
import time
import joblib
import pandas as pd
import tensorflow as tf
import yfinance as yf
import pyautogui

# CONFIGURAÇÃO
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "modelo_global_b3.h5")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler_global.pkl")
WATCHLIST = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "PRIO3.SA", "ABEV3.SA"]

def disparar_ordem_hotkey(ticker, sinal):
    print(f"⚠️ ATENÇÃO: Sinal de {sinal} para {ticker}!")
    print(f"Mude o gráfico para {ticker} AGORA. Disparando em 3 segundos...")
    time.sleep(3)
    if sinal == "COMPRA":
        pyautogui.hotkey('ctrl', 'alt', 'c')
    else:
        pyautogui.hotkey('ctrl', 'alt', 'v')
    print(f"✅ Comando enviado ao Profit para {ticker}")

def monitor_brain():
    print("🧠 JARVIS: Carregando Cérebro...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cols = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
    print("✅ Cérebro On-line! Iniciando Varredura...")

    while True:
        print(f"\n🔄 Varredura de Mercado: {time.strftime('%H:%M:%S')}")
        for ticker in WATCHLIST:
            try:
                # 1. Download de dados
                df = yf.download(ticker, period="60d", interval="1d", progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                # 2. Cálculo dos Indicadores (O que faltava!)
                close = df["Close"]
                df["SMA_9"] = close.rolling(9).mean()
                df["SMA_20"] = close.rolling(20).mean()
                df["BB_Width"] = (close.rolling(20).std() * 4) / df["SMA_20"]
                
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                df["RSI"] = 100 - (100 / (1 + (gain / loss)))
                
                df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
                df["ROC"] = close.pct_change(5) * 100
                df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()

                # 3. Preparar para IA
                last_row = df.dropna()[cols].tail(1)
                if last_row.empty: continue
                
                # 4. Predição
                X = scaler.transform(last_row.values)
                prob = model.predict(X, verbose=0)[0][0]
                
                ticker_puro = ticker.split('.')[0]
                print(f"🔍 {ticker_puro.ljust(6)} | Prob: {prob:.4f}")

                if prob > 0.85:
                    disparar_ordem_hotkey(ticker_puro, "COMPRA")
                    time.sleep(10) # Pausa curta
                elif prob < 0.15:
                    disparar_ordem_hotkey(ticker_puro, "VENDA")
                    time.sleep(10)
            except Exception as e:
                print(f"❌ Erro em {ticker}: {e}")
                continue
        
        print("⏸️ Aguardando 30 segundos para a próxima rodada...")
        time.sleep(30)

if __name__ == "__main__":
    monitor_brain()