import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf

# --- CONFIGURAÇÃO DO BACKTEST REALISTA ---
CAPITAL_INICIAL = 10000.00
RISCO_POR_TRADE = 0.02    # 2% de risco por operação
STOP_LOSS_PCT = 0.02      # Stop de 2%
TAXA_B3 = 0.0005          # 0.05% (Corretagem + Emolumentos)

# CAMINHOS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

def rodar_backtest_integrado(ticker):
    print(f"\n🧪 BACKTEST INTEGRADO (RISK + IA): {ticker}")
    
    # 1. Carregar IA
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Dados de 1 ano
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 3. Features
    df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
    df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
    delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df = df.dropna()

    # 4. Predições
    X = scaler.transform(df[["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]].values)
    df["Prob"] = model.predict(X, verbose=0)
    
    # 5. SIMULAÇÃO DE CARTEIRA (A MÁGICA)
    saldo = CAPITAL_INICIAL
    historico_saldo = []
    trades_totais = 0
    
    for i in range(len(df)-1):
        preco_atual = df["Close"].iloc[i]
        retorno_seguinte = df["Close"].pct_change().iloc[i+1]
        prob = df["Prob"].iloc[i]
        
        sinal = 0
        if prob > 0.55: sinal = 1      # Compra
        elif prob < 0.45: sinal = -1   # Venda
        
        if sinal != 0:
            # Chama a lógica do Módulo 2: Cálculo de Lote
            valor_arriscar = saldo * RISCO_POR_TRADE
            perda_por_acao = preco_atual * STOP_LOSS_PCT
            lote = int(valor_arriscar / perda_por_acao)
            
            # Resultado do trade (Simplificado: Retorno do dia seguinte)
            # Se for sinal de compra, ganha o retorno. Se for venda, ganha o inverso.
            resultado_bruto = (sinal * retorno_seguinte)
            
            # Se o retorno for pior que o Stop Loss, limita a perda
            if resultado_bruto < -STOP_LOSS_PCT:
                resultado_bruto = -STOP_LOSS_PCT
            
            lucro_financeiro = (lote * preco_atual) * resultado_bruto
            custos = (lote * preco_atual) * TAXA_B3
            
            saldo += (lucro_financeiro - custos)
            trades_totais += 1
            
        historico_saldo.append(saldo)

    # 6. RESULTADOS
    lucro_final = ((saldo / CAPITAL_INICIAL) - 1) * 100
    print("="*45)
    print(f"💰 Saldo Final: R${saldo:.2f}")
    print(f"📈 Rentabilidade: {lucro_final:.2f}%")
    print(f"🔄 Total de Trades: {trades_totais}")
    print("="*45)

if __name__ == "__main__":
    rodar_backtest_integrado("PRIO3.SA")