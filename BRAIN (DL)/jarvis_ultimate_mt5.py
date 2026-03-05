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

# --- TRAVA DE CAPITAL VIRTUAL ---
CAPITAL_VIRTUAL_INICIAL = 1000.0
RISCO_POR_TRADE = 0.20

# --- MAPEAMENTO DE ESPECIALISTAS ---
ASSET_EXPERT_CONFIG = {
    "PETR4": "DeepLearning", "VALE3": "DeepLearning", "ITUB4": "DeepLearning",
    "BBDC4": "DeepLearning", "ABEV3": "DeepLearning", "PRIO3": "DeepLearning",
    "MGLU3": "DeepLearning", "WEGE3": "DeepLearning", "BBAS3": "DeepLearning",
    "RENT3": "DeepLearning", "RADL3": "Momentum", "RAIL3": "Reversion",
    "SUZB3": "DeepLearning", "GGBR4": "DeepLearning", "CSNA3": "DeepLearning",
    "B3SA3": "DeepLearning", "HYPE3": "Reversion", "VIVT3": "DeepLearning"
}

VAR_THRESHOLD_CRITICO = -0.04

class JarvisUltimateMonitor:
    def __init__(self):
        print(f"\n{'='*65}\n🚀 [JARVIS ULTIMATE MONITOR] INICIADO\n{'='*65}")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.quant = JarvisQuantFramework()
        self.start_time = datetime.now()
        self.capital_virtual_atual = CAPITAL_VIRTUAL_INICIAL
        self.trades_executados = []
        self.cols = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
        
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
            print(f"❌ Erro MT5: {mt5.last_error()}")
            sys.exit()

    def calcular_lote(self, preco):
        valor_alocacao = CAPITAL_VIRTUAL_INICIAL * RISCO_POR_TRADE
        qtd_acoes = valor_alocacao / preco
        lotes_inteiros = int(qtd_acoes // 100) * 100
        if lotes_inteiros < 100:
            if preco * 100 <= self.capital_virtual_atual: return 100.0
            return 0.0
        return float(lotes_inteiros)

    def predict_dl(self, df):
        close = df["Close"]
        df["SMA_9"] = close.rolling(9).mean(); df["SMA_20"] = close.rolling(20).mean()
        df["BB_Width"] = (close.rolling(20).std() * 4) / df["SMA_20"]
        delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = close.pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        last = df.dropna()[self.cols].tail(1)
        if last.empty: return 0.5
        return self.model.predict(self.scaler.transform(last.values), verbose=0)[0][0]

    def check_momentum(self, df):
        max_10 = df['High'].rolling(10).max().iloc[-2]
        min_10 = df['Low'].rolling(10).min().iloc[-2]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        close = df['Close'].iloc[-1]
        return 1 if close > max_10 and close > sma_50 else (-1 if close < min_10 else 0)

    def check_reversion(self, df):
        sma_20 = df["Close"].rolling(20).mean().iloc[-1]
        std = df["Close"].rolling(20).std().iloc[-1]
        lower = sma_20 - (2 * std)
        close = df["Close"].iloc[-1]
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
        return 1 if close < lower and rsi < 30 else (-1 if close > sma_20 else 0)

    def gerar_relatorio(self):
        current_time = datetime.now()
        duracao = current_time - self.start_time
        lucro_total = self.capital_virtual_atual - CAPITAL_VIRTUAL_INICIAL
        rentabilidade = (lucro_total / CAPITAL_VIRTUAL_INICIAL) * 100
        
        print(f"\n" + "="*65)
        print(f"📊 RELATÓRIO DE PERFORMANCE - JARVIS ULTIMATE")
        print("="*65)
        print(f"🕒 Início:          {self.start_time.strftime('%H:%M:%S')}")
        print(f"🕒 Agora:           {current_time.strftime('%H:%M:%S')}")
        print(f"⌛ Duração:         {str(duracao).split('.')[0]}")
        print(f"💰 Cap. Inicial:    R$ {CAPITAL_VIRTUAL_INICIAL:,.2f}")
        print(f"📈 Cap. Final:      R$ {self.capital_virtual_atual:,.2f}")
        print(f"🚀 Rentabilidade:   {rentabilidade:,.2f}%")
        print(f"📦 Trades Hoje:     {len(self.trades_executados)}")
        print("="*65 + "\n")

    def enviar_ordem(self, symbol, sinal):
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return
        
        preco = tick.ask if sinal == "COMPRA" else tick.bid
        qtd = self.calcular_lote(preco)
        if qtd <= 0: return

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(qtd),
            "type": mt5.ORDER_TYPE_BUY if sinal == "COMPRA" else mt5.ORDER_TYPE_SELL,
            "price": float(preco), "magic": 2026, "comment": "ULTIMATE",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"💰 [OPERADO] {sinal} em {symbol}")
            self.trades_executados.append({"asset": symbol, "side": sinal, "time": datetime.now()})
            # Simulação ultra-simplificada de P&L para o relatório virtual
            if sinal == "COMPRA": self.capital_virtual_atual -= 1.0 # Simulação de taxa/custo
            else: self.capital_virtual_atual += 1.0

    def loop(self):
        ciclo_count = 0
        while True:
            ciclo_count += 1
            print(f"\n🔄 Varredura #{ciclo_count} | {datetime.now().strftime('%H:%M:%S')}")
            for ticker, expert in ASSET_EXPERT_CONFIG.items():
                try:
                    df = yf.download(ticker + ".SA", period="60d", interval="1d", progress=False)
                    if df.empty: continue
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    
                    var = self.quant.value_at_risk(df['Close'].pct_change().dropna())
                    if var < VAR_THRESHOLD_CRITICO: continue
                    
                    sinal = None
                    if expert == "DeepLearning":
                        p = self.predict_dl(df)
                        if p > 0.85: sinal = "COMPRA"
                        elif p < 0.15: sinal = "VENDA"
                    elif expert == "Momentum":
                        m = self.check_momentum(df)
                        if m == 1: sinal = "COMPRA"
                        elif m == -1: sinal = "VENDA"
                    elif expert == "Reversion":
                        r = self.check_reversion(df)
                        if r == 1: sinal = "COMPRA"
                        elif r == -1: sinal = "VENDA"

                    if sinal: self.enviar_ordem(ticker, sinal)

                except Exception: continue
            
            if ciclo_count % 5 == 0: self.gerar_relatorio()
            time.sleep(60)

if __name__ == "__main__":
    bot = JarvisUltimateMonitor()
    bot.loop()
