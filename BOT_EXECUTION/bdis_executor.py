import pyautogui
import time
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf
import pygetwindow as gw
import os
import numpy as np

# Adicionando path para importar o RISK_MANAGER
import sys
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
sys.path.append(BASE_DIR)
from RISK_MANAGER.quant_framework import JarvisQuantFramework

# --- CONFIGURAÇÃO ---
MODELO_PATH = os.path.join(BASE_DIR, r"BRAIN (DL)\modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, r"BRAIN (DL)\scaler_global.pkl")
LOG_PATH = os.path.join(BASE_DIR, r"LOGS\operacoes_jarvis.txt")
ATIVO = "PETR4.SA"

# Limite Crítico de Value at Risk (VaR): Se o ativo perder mais que X% no dia histórico, bloqueamos o trade!
VAR_THRESHOLD_CRITICO = -0.04  # 4% de VaR 95%

class JarvisB3Full:
    def __init__(self):
        print(f"🧠 JARVIS_B3: Localizando arquivos em: {BASE_DIR}")
        self.model = tf.keras.models.load_model(MODELO_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.quant = JarvisQuantFramework() # Injeção de Risco
        self.features_col = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
        print("✅ Inteligência Sincronizada com Sucesso.")

    def calcular_features(self, df):
        # Limpa multi-index do yfinance
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]
        
        # Indicadores
        df["SMA_9"] = close.rolling(9).mean()
        df["SMA_20"] = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["BB_Width"] = (std * 4) / df["SMA_20"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        df["ATR"] = (high - low).rolling(14).mean()
        df["ROC"] = close.pct_change(5) * 100
        df["Vol_Shock"] = volume / volume.rolling(20).mean()
        return df.dropna()

    def focar_e_boletar(self, sinal, ativo=ATIVO):
        try:
            wins = [w for w in gw.getWindowsWithTitle('Profit') if w.visible]
            if not wins:
                print("⚠️ Profit não encontrado!")
                return
            
            win = wins[0]
            win.activate()
            time.sleep(0.3)
            
            # Clica no centro da janela para garantir o foco
            pyautogui.click(win.left + (win.width // 2), win.top + (win.height // 2))
            time.sleep(0.1)

            # --- TRAVA DE SEGURANÇA ---
            ticker_limpo = ativo.replace(".SA", "")
            print(f"🔒 [JARVIS SECURITY] Forçando Ticker no Ativo: {ticker_limpo}")
            pyautogui.write(ticker_limpo)
            time.sleep(0.3)
            pyautogui.press('enter')
            time.sleep(0.8) # Tempo de carregamento do gráfico B3
            
            if sinal == 1:
                pyautogui.press('f5') # Dispara Compra
                time.sleep(0.1)
                pyautogui.press('enter') # Confirma a boleta amarela
                print("🚀 [GATILHO] F5 + ENTER Enviado (Compra)")
            elif sinal == -1:
                pyautogui.press('f9') # Dispara Venda
                time.sleep(0.1)
                pyautogui.press('enter') # Confirma a boleta amarela
                print("🔻 [GATILHO] F9 + ENTER Enviado (Venda)")
        except Exception as e:
            print(f"⚠️ Erro no gatilho: {e}")

    def loop(self):
        print(f"⚡ JARVIS_B3: Monitorando {ATIVO}...")
        while True:
            try:
                dados = yf.download(ATIVO, period="60d", interval="1d", progress=False)
                df_final = self.calcular_features(dados.copy())
                
                # --- CAMADA QUANTITATIVA DE RISCO ---
                # Pega os retornos percentuais dos últimos 60 dias do ativo
                retornos = dados['Close'].pct_change().dropna()
                var_ativo = self.quant.value_at_risk(retornos)
                
                print(f"⚖️  Risco do Mercado Atual (VaR 95%): {var_ativo:.2%}")
                
                # Bloqueio em caso de volatilidade destrutiva
                if var_ativo < VAR_THRESHOLD_CRITICO:
                    print(f"🚫 [RISK BLOCKED] Operação cancelada. Mercado extremamente volátil (VaR: {var_ativo:.2%} pior que Mínimo: {VAR_THRESHOLD_CRITICO:.2%}).")
                    time.sleep(60)
                    continue

                # Prepara dados para predição da IA
                last_row = df_final[self.features_col].tail(1).values
                X = self.scaler.transform(last_row)
                
                prob = self.model.predict(X, verbose=0)[0][0]
                print(f"📊 {time.strftime('%H:%M:%S')} | Prob de Alta: {prob:.4f}")

                # Gatilhos
                if prob > 0.55:
                    self.focar_e_boletar(1)
                    time.sleep(300) # Espera 5 min após boleta
                elif prob < 0.45:
                    self.focar_e_boletar(-1)
                    time.sleep(300)
                
                time.sleep(20) # Intervalo de atualização
            except Exception as e:
                print(f"⚠️ Falha: {e}")
                time.sleep(10)

if __name__ == "__main__":
    JarvisB3Full().loop()