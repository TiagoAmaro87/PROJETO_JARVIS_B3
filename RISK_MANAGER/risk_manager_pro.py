import os
import time
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import pyautogui
import pygetwindow as gw

# ==========================================================
# 1. CONFIGURAÇÃO DE COORDENADAS (PREENCHA APÓS CALIBRAR)
# ==========================================================
# DICA: Deixe as 6 janelas de gráfico lado a lado no Profit
BOTOES_COMPRA = {
    "PETR4.SA": (0, 0),  # Substitua (0,0) pelo X e Y real
    "VALE3.SA": (0, 0),
    "ITUB4.SA": (0, 0),
    "BBDC4.SA": (0, 0),
    "PRIO3.SA": (0, 0),
    "ABEV3.SA": (0, 0),
}

WATCHLIST = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "PRIO3.SA", "ABEV3.SA"]

# Ajuste de Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

# ==========================================================
# 2. FUNÇÃO DE DISPARO POR CLIQUE (SEM ERRO)
# ==========================================================
def disparar_por_clique(ticker):
    try:
        x, y = BOTOES_COMPRA[ticker]
        if x == 0 and y == 0:
            print(f"⚠️ Coordenada para {ticker} não configurada!")
            return

        # Traz o Profit para frente
        wins = [w for w in gw.getWindowsWithTitle('Profit') if w.visible]
        if wins: wins[0].activate()
        
        # Move e clica no botão de compra daquela janela específica
        pyautogui.click(x, y)
        time.sleep(0.2)
        pyautogui.press('enter') # Confirma a boleta do Profit
        print(f"🚀 [ORDEM EXECUTADA] Clique em {ticker} via Coordenadas.")
    except Exception as e:
        print(f"❌ Erro no clique: {e}")

# ==========================================================
# 3. MONITORAMENTO E INTELIGÊNCIA ARTIFICIAL
# ==========================================================
def monitor_final():
    print("\n" + "="*50)
    print("🏆 JARVIS MODO HITMAN: CLIQUE DIRETO POR ATIVO")
    print("="*50)

    # Carrega a IA do seu FDS
    if not os.path.exists(MODEL_PATH):
        print("❌ Modelo .h5 não encontrado!")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cols = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]

    while True:
        print(f"\n🔄 Varredura de Mercado: {time.strftime('%H:%M:%S')}")
        
        for ticker in WATCHLIST:
            try:
                # Download dos dados
                df = yf.download(ticker, period="60d", interval="1d", progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                # Cálculo dos Indicadores (Fiel ao seu Deep Learning)
                c = df["Close"]
                df["SMA_9"] = c.rolling(9).mean(); df["SMA_20"] = c.rolling(20).mean()
                df["BB_Width"] = (c.rolling(20).std() * 4) / df["SMA_20"]
                d = c.diff(); g = d.where(d > 0, 0).rolling(14).mean(); l = -d.where(d < 0, 0).rolling(14).mean()
                df["RSI"] = 100 - (100 / (1 + (g / l)))
                df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
                df["ROC"] = c.pct_change(5) * 100
                df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()

                last = df.dropna()[cols].tail(1)
                if last.empty: continue

                # Predição da IA
                prob = model.predict(scaler.transform(last.values), verbose=0)[0][0]
                print(f"🔍 {ticker.ljust(9)} | Probabilidade: {prob:.4f}")

                # GATILHO DE EXECUÇÃO
                if prob > 0.85:
                    disparar_por_clique(ticker)
                    time.sleep(20) # Pausa de segurança
            except Exception as e:
                continue

        print("⏳ Ciclo completo. Reiniciando em 30 segundos...")
        time.sleep(30)

# ==========================================================
# 4. CALIBRAGEM INICIAL (RODE UMA VEZ)
# ==========================================================
if __name__ == "__main__":
    print("\n--- MODO DE CALIBRAGEM ---")
    print("Você tem 15 segundos para colocar o mouse sobre os botões de COMPRA.")
    print("Anote os valores de X e Y que aparecerem e coloque no código!")
    print("-" * 30)
    
    try:
        # Mostra a posição do mouse em tempo real por 15 segundos
        for i in range(30): 
            x, y = pyautogui.position()
            print(f"Posição atual do Mouse -> X: {x} | Y: {y}  ", end="\r")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    print("\n\nIniciando Monitoramento real agora...")
    monitor_final()