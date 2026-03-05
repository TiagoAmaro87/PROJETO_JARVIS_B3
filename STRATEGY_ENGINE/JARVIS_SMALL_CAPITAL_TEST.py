import os
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DE GUERRA ---
CAPITAL_POR_ATIVO = 1000.00
TAXA_B3_PCT = 0.0005
ALIQUOTA_IR = 0.15

ATIVOS_ELITE = ["MGLU3.SA", "CSNA3.SA", "BBAS3.SA", "RENT3.SA", "BBDC4.SA"]
ATIVO_MOMENTUM = "RADL3.SA"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

def engine_dl(ticker, model, scaler):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Features padrão Jarvis
    df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
    df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
    delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df = df.dropna()

    X = scaler.transform(df[["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]].values)
    probs = model.predict(X, verbose=0)
    
    saldo = CAPITAL_POR_ATIVO
    taxas = 0
    for i in range(len(df)-1):
        sinal = 1 if probs[i] > 0.55 else (-1 if probs[i] < 0.45 else 0)
        if sinal != 0:
            lote = int(saldo / df["Close"].iloc[i])
            if lote > 0:
                retorno = df["Close"].pct_change().iloc[i+1]
                v_op = lote * df["Close"].iloc[i]
                t = v_op * TAXA_B3_PCT
                saldo += (v_op * (sinal * retorno)) - t
                taxas += t
    return saldo, taxas

def engine_momentum(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['Max_10'] = df['High'].rolling(10).max().shift(1)
    df['Min_10'] = df['Low'].rolling(10).min().shift(1)
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    saldo = CAPITAL_POR_ATIVO
    taxas = 0; pos = 0
    for i in range(10, len(df)):
        if pos == 0 and df['Close'].iloc[i] > df['Max_10'].iloc[i] and df['Close'].iloc[i] > df['SMA_50'].iloc[i]:
            pos = 1; p_in = df['Close'].iloc[i]; t = (int(saldo/p_in)*p_in)*TAXA_B3_PCT; taxas += t
        elif pos == 1 and df['Close'].iloc[i] < df['Min_10'].iloc[i]:
            ret = (df['Close'].iloc[i]/p_in)-1; t = (int(saldo/p_in)*df['Close'].iloc[i])*TAXA_B3_PCT; 
            saldo += (saldo * ret) - t; taxas += t; pos = 0
    return saldo, taxas

def main():
    print("\n" + "="*115)
    print("🚀 JARVIS ULTIMATE: RELATÓRIO DE ESTRESSE CONSOLIDADO (CAPITAL R$ 1.000,00 / ATIVO)")
    print("="*115)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    results = []

    # Processa Elite (DL)
    for t in ATIVOS_ELITE:
        s_f, tax = engine_dl(t, model, scaler)
        l_b = s_f - CAPITAL_POR_ATIVO
        ir = l_b * ALIQUOTA_IR if l_b > 0 else 0
        results.append({"Ativo": t, "Cap. Inicial": CAPITAL_POR_ATIVO, "Taxas": tax, "Lucro Bruto": l_b, "IR": ir, "L. Líquido": l_b - ir, "Saldo Final": s_f - ir})

    # Processa RADL3 (Momentum)
    s_f, tax = engine_momentum(ATIVO_MOMENTUM)
    l_b = s_f - CAPITAL_POR_ATIVO
    ir = l_b * ALIQUOTA_IR if l_b > 0 else 0
    results.append({"Ativo": ATIVO_MOMENTUM + "*", "Cap. Inicial": CAPITAL_POR_ATIVO, "Taxas": tax, "Lucro Bruto": l_b, "IR": ir, "L. Líquido": l_b - ir, "Saldo Final": s_f - ir})

    df = pd.DataFrame(results)
    print(df.to_string(index=False, formatters={
        "Cap. Inicial": "R$ {:,.2f}".format, "Taxas": "R$ {:,.2f}".format, 
        "Lucro Bruto": "R$ {:,.2f}".format, "IR": "R$ {:,.2f}".format, 
        "L. Líquido": "R$ {:,.2f}".format, "Saldo Final": "R$ {:,.2f}".format
    }))
    
    tot_inv = CAPITAL_POR_ATIVO * len(results)
    tot_liq = df["L. Líquido"].sum()
    print("-" * 115)
    print(f"✅ LUCRO LÍQUIDO TOTAL: R$ {tot_liq:,.2f} | 🚀 PATRIMÔNIO FINAL: R$ {tot_inv + tot_liq:,.2f} | 📈 RENTABILIDADE: {((tot_inv+tot_liq)/tot_inv - 1)*100:.2f}%")
    print("(*) Ativo operado via Estratégia de Momentum | Demais via Deep Learning")
    print("="*115)

if __name__ == "__main__":
    main()