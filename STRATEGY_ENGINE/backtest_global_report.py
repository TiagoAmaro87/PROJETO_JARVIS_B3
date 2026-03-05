import os
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
CAPITAL_POR_ATIVO = 10000.00
RISCO_POR_TRADE = 0.02
STOP_LOSS_PCT = 0.02
TAXA_B3_PCT = 0.0005 
ALÍQUOTA_IR = 0.15    

ATIVOS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "PRIO3.SA", 
    "ABEV3.SA", "MGLU3.SA", "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "CSNA3.SA", "SUZB3.SA", "RAIL3.SA", "GGBR4.SA", "RADL3.SA"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

def rodar_backtest_financeiro(ticker, model, scaler):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Features (Mesma lógica anterior)
        df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
        df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df = df.dropna()

        X = scaler.transform(df[["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]].values)
        df["Prob"] = model.predict(X, verbose=0)
        
        saldo = CAPITAL_POR_ATIVO
        total_taxas = 0
        
        for i in range(len(df)-1):
            prob = df["Prob"].iloc[i]
            sinal = 1 if prob > 0.55 else (-1 if prob < 0.45 else 0)
            if sinal != 0:
                preco = df["Close"].iloc[i]
                retorno = df["Close"].pct_change().iloc[i+1]
                lote = int((saldo * RISCO_POR_TRADE) / (preco * STOP_LOSS_PCT))
                if lote > 0:
                    res_bruto = (sinal * retorno)
                    if res_bruto < -STOP_LOSS_PCT: res_bruto = -STOP_LOSS_PCT
                    fin_trade = lote * preco
                    taxa = fin_trade * TAXA_B3_PCT
                    saldo += (fin_trade * res_bruto) - taxa
                    total_taxas += taxa
        
        lucro_bruto = saldo - CAPITAL_POR_ATIVO
        imposto = lucro_bruto * ALÍQUOTA_IR if lucro_bruto > 0 else 0
        saldo_liquido = saldo - imposto

        return {
            "Ativo": ticker, 
            "Lucro Bruto (R$)": round(lucro_bruto, 2),
            "Taxas B3 (R$)": round(total_taxas, 2),
            "Imposto IR (R$)": round(imposto, 2),
            "Lucro Líquido (R$)": round(saldo_liquido - CAPITAL_POR_ATIVO, 2),
            "Saldo Final (R$)": round(saldo_liquido, 2),
            "Retorno %": round(((saldo_liquido/CAPITAL_POR_ATIVO)-1)*100, 2)
        }
    except: return None

def imprimir_bloco(df, titulo):
    total_investido = CAPITAL_POR_ATIVO * len(df)
    total_lucro_liquido = df["Lucro Líquido (R$)"].sum()
    patrimonio_final = total_investido + total_lucro_liquido
    total_taxas = df["Taxas B3 (R$)"].sum()
    total_ir = df["Imposto IR (R$)"].sum()

    print("\n" + "="*110)
    print(f"🏆 {titulo}")
    print("="*110)
    print(df.to_string(index=False))
    print("-"*110)
    print(f"🔸 Capital Inicial Total:      R$ {total_investido:,.2f}")
    print(f"🔸 Total Pago em Taxas B3:     R$ {total_taxas:,.2f}")
    print(f"🔸 Total Pago em Impostos IR:  R$ {total_ir:,.2f}")
    print(f"✅ LUCRO LÍQUIDO ACUMULADO:   R$ {total_lucro_liquido:,.2f}")
    print(f"🚀 PATRIMÔNIO FINAL:          R$ {patrimonio_final:,.2f}")
    print(f"📈 RENTABILIDADE REAL:        {((patrimonio_final/total_investido)-1)*100:.2f}%")
    print("="*110)

def executar_simulacao():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    resultados = []
    for ticker in ATIVOS:
        res = rodar_backtest_financeiro(ticker, model, scaler)
        if res: resultados.append(res)

    df_global = pd.DataFrame(resultados).sort_values(by="Retorno %", ascending=False)
    
    # 1. Relatório Global
    imprimir_bloco(df_global, "RELATÓRIO DE CAIXA CONSOLIDADO - PORTFÓLIO GLOBAL (15 ATIVOS)")
    
    # 2. Relatório Top 5
    df_top5 = df_global.head(5)
    imprimir_bloco(df_top5, "RELATÓRIO DE ELITE - TOP 5 ATIVOS (MÁXIMA PERFORMANCE)")

if __name__ == "__main__":
    executar_simulacao()