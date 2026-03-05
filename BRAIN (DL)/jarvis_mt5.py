import os
import time
import joblib
import pandas as pd
import tensorflow as tf
import yfinance as yf
import MetaTrader5 as mt5
import sys

# Adicionando o diretório base para importar o RISK_MANAGER
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
sys.path.append(BASE_DIR)

from RISK_MANAGER.quant_framework import JarvisQuantFramework

# --- CONFIGURAÇÃO DA CONTA XP ---
MT5_LOGIN = 54206952
MT5_PASS = "And1and1411208*#"
MT5_SERVER = "XPMT5-DEMO" # Servidor de simulador da XP

# --- CONFIGURAÇÃO DO MODELO ---
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")
WATCHLIST = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
VAR_THRESHOLD_CRITICO = -0.04  # 4% de VaR 95%

class JarvisMT5Bot:
    def __init__(self):
        print("🔍 [DEBUG] Entrando no construtor do Bot...")
        print("🧠 [JARVIS MT5] Carregando Modelo DL...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("🧠 [JARVIS MT5] Modelo Carregado.")
        
        self.scaler = joblib.load(SCALER_PATH)
        self.quant = JarvisQuantFramework()
        self.cols = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
        
        print(f"📡 [JARVIS MT5] Tentando Conexão com Servidor: {MT5_SERVER}...")
        # Iniciar login no MT5 de forma direta
        authorized = mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER)
        if authorized:
            print(f"✅ Conectado com Sucesso à Conta XP: {MT5_LOGIN}")
        else:
            print(f"❌ Falha na Autenticação XP. Erro: {mt5.last_error()}")
            return

    def calcular_indicadores(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
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
        return df.dropna()

    def enviar_ordem(self, symbol, sinal):
        # XP MT5 geralmente usa tickers sem .SA no simulador, mas vamos garantir
        # que o ativo esteja visível no terminal.
        mt5.symbol_select(symbol, True)
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Não foi possível obter preço para {symbol}")
            return

        lot = 100.0 # Lote padrão na B3 (100 ações)
        tipo = mt5.ORDER_TYPE_BUY if sinal == "COMPRA" else mt5.ORDER_TYPE_SELL
        preco = tick.ask if sinal == "COMPRA" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": tipo,
            "price": preco,
            "magic": 2024,
            "comment": "JARVIS_B3_BOT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Falha na Execução {symbol}: {result.comment}")
        else:
            print(f"🚀 [EXECUTADO] {sinal} de {lot} {symbol} a {preco}")

    def loop(self):
        print(f"⚡ Monitorando Watchlist: {WATCHLIST}")
        while True:
            for ticker in WATCHLIST:
                try:
                    # 1. Coleta de Dados via YFinance (para alimentar a IA)
                    ticker_yf = ticker + ".SA"
                    dados = yf.download(ticker_yf, period="60d", interval="1d", progress=False)
                    if dados.empty: continue
                    
                    # 2. Risco Quantitativo (VaR)
                    retornos = dados['Close'].pct_change().dropna()
                    var_atual = self.quant.value_at_risk(retornos)
                    
                    if var_atual < VAR_THRESHOLD_CRITICO:
                        print(f"⚖️ {ticker}: Risco Excessivo (VaR {var_atual:.2%}). Ignorando.")
                        continue

                    # 3. Predição DL (Brain)
                    df_proc = self.calcular_indicadores(dados)
                    last_features = df_proc[self.cols].tail(1)
                    X = self.scaler.transform(last_features.values)
                    prob = self.model.predict(X, verbose=0)[0][0]

                    print(f"🔍 {ticker.ljust(6)} | Prob: {prob:.4f} | VaR: {var_atual:.2%}")

                    # 4. Decisão de Trading
                    if prob > 0.80:
                        self.enviar_ordem(ticker, "COMPRA")
                    elif prob < 0.20:
                        self.enviar_ordem(ticker, "VENDA")

                except Exception as e:
                    print(f"⚠️ Erro ao processar {ticker}: {e}")
            
            print(f"--- Fim da Varredura {time.strftime('%H:%M:%S')} (Pausa 60s) ---")
            time.sleep(60)

if __name__ == "__main__":
    bot = JarvisMT5Bot()
    bot.loop()
