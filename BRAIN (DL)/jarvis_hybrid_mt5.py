import os
import time
import joblib
import pandas as pd
import tensorflow as tf
import yfinance as yf
import MetaTrader5 as mt5
import sys
from datetime import datetime

# Adicionando o diretório base para importar o RISK_MANAGER
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
sys.path.append(BASE_DIR)

from RISK_MANAGER.quant_framework import JarvisQuantFramework

# --- CONFIGURAÇÃO DA CONTA XP ---
MT5_LOGIN = 54206952
MT5_PASS = "And1and1411208*#"
MT5_SERVER = "XPMT5-DEMO"

# --- CONFIGURAÇÃO DO MODELO ---
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

# --- DICIONÁRIO DE ESPECIALISTAS (Configurado via Backtest de 05/03) ---
# Aqui definimos qual motor (DL ou MOM) é o vencedor para cada papel.
ASSET_EXPERT_CONFIG = {
    "PETR4": "DeepLearning",
    "VALE3": "DeepLearning",
    "ITUB4": "DeepLearning",
    "BBDC4": "DeepLearning",
    "ABEV3": "DeepLearning",
    "MGLU3": "DeepLearning",
    "BBAS3": "DeepLearning",
    "CSNA3": "DeepLearning",
    "RADL3": "Momentum",  # Vencedor absoluto em RADL3
    "RENT3": "DeepLearning",
    "WEGE3": "DeepLearning"
}

# --- CONFIGURAÇÃO DE RISCO ---
PERCENTUAL_POR_TRADE = 0.10  # Aloca no máximo 10% do capital por operação
VAR_THRESHOLD_CRITICO = -0.04

class JarvisHybridBot:
    def __init__(self):
        print(f"\n{'='*60}\n🚀 [JARVIS HYBRID] MODO ESPECIALISTA + RISK MANAGER\n{'='*60}")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.quant = JarvisQuantFramework()
        self.cols = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
        
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
            print(f"❌ Erro Crítico MT5: {mt5.last_error()}")
            sys.exit()
        
        info = mt5.account_info()
        if info:
            print(f"✅ Conectado à XP | Conta: {MT5_LOGIN} | Saldo: R$ {info.balance:,.2f}")
        else:
            print("❌ Não foi possível obter informações da conta.")

    def calcular_lote_dinamico(self, preco_ativo):
        """Calcula a quantidade total de ações baseado no saldo (XP B3: Volume = Ações)."""
        info = mt5.account_info()
        if not info: return 0.0
        
        capital_disponivel = info.balance
        valor_alocacao = capital_disponivel * PERCENTUAL_POR_TRADE
        
        # Na XP B3 Simulado, volume_min=100 e volume_step=100.
        # O campo 'volume' deve receber a quantidade de ações (ex: 100, 200, 5000).
        qtd_acoes = (valor_alocacao / preco_ativo)
        
        # Arredonda para baixo para o múltiplo de 100 mais próximo
        acoes_ajustadas = int(qtd_acoes // 100) * 100
        
        if acoes_ajustadas < 100: return 0.0
        return float(acoes_ajustadas)

    def get_market_data(self, ticker):
        df = yf.download(ticker + ".SA", period="60d", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def predict_dl(self, df):
        # Cálculo de indicadores
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
        
        last = df.dropna()[self.cols].tail(1)
        if last.empty: return 0.5
        X = self.scaler.transform(last.values)
        return self.model.predict(X, verbose=0)[0][0]

    def check_momentum(self, df):
        max_10 = df['High'].rolling(10).max().iloc[-2]
        min_10 = df['Low'].rolling(10).min().iloc[-2]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        close = df['Close'].iloc[-1]
        
        if close > max_10 and close > sma_50: return 1
        if close < min_10: return -1
        return 0

    def executar_ordem(self, symbol, sinal):
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return
        
        preco = tick.ask if sinal == "COMPRA" else tick.bid
        qtd_acoes = self.calcular_lote_dinamico(preco)
        
        if qtd_acoes <= 0:
            print(f"⚠️ Saldo insuficiente ou lote inválido para {symbol}")
            return

        tipo = mt5.ORDER_TYPE_BUY if sinal == "COMPRA" else mt5.ORDER_TYPE_SELL
        
        # XP usualmente suporta IOC para execuções a mercado no simulador
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(qtd_acoes),
            "type": tipo,
            "price": float(preco),
            "magic": 2024,
            "comment": f"JARVIS_{int(qtd_acoes)}UN",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, 
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"💰 [OPERADO] {sinal} de {int(qtd_acoes)} ações em {symbol}")
        else:
            print(f"❌ Falha no Roteamento {symbol}: {res.comment} (Retcode: {res.retcode})")

    def loop(self):
        while True:
            print(f"\n--- ciclo: {datetime.now().strftime('%H:%M:%S')} ---")
            for ticker, expert in ASSET_EXPERT_CONFIG.items():
                try:
                    df = self.get_market_data(ticker)
                    if df is None: continue
                    
                    var = self.quant.value_at_risk(df['Close'].pct_change().dropna())
                    if var < VAR_THRESHOLD_CRITICO:
                        print(f"⚖️  {ticker}: VaR Crítico ({var:.2%}) - Bloqueado.")
                        continue
                    
                    sinal_final = None
                    if expert == "DeepLearning":
                        prob = self.predict_dl(df)
                        print(f"🧠 {ticker}: DL Prob {prob:.4f}")
                        if prob > 0.85: sinal_final = "COMPRA"
                        elif prob < 0.15: sinal_final = "VENDA"
                    
                    elif expert == "Momentum":
                        mom = self.check_momentum(df)
                        print(f"⚡ {ticker}: Mom {mom}")
                        if mom == 1: sinal_final = "COMPRA"
                        elif mom == -1: sinal_final = "VENDA"

                    if sinal_final:
                        self.executar_ordem(ticker, sinal_final)

                except Exception as e:
                    print(f"⚠️ Erro em {ticker}: {e}")
            
            time.sleep(60)

if __name__ == "__main__":
    bot = JarvisHybridBot()
    bot.loop()

