import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO INDUSTRIAL ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

CAPITAL_INICIAL = 1000.0
# TAXAS REAIS B3 (Emolumentos + Liquidação ~ 0.03% por perna = 0.06% total)
TAXA_B3_TOTAL = 0.0006 
# IMPOSTO DE RENDA (20% para Day Trade - assumindo que a maioria dos trades da IA são rápidos)
IMPOSTO_RENDA_PCT = 0.20

ATIVOS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", 
    "PRIO3.SA", "MGLU3.SA", "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "RADL3.SA", "RAIL3.SA", "SUZB3.SA", "GGBR4.SA", "CSNA3.SA",
    "B3SA3.SA", "HYPE3.SA", "VIVT3.SA", "GOAU4.SA", "EQTL3.SA", "LREN3.SA"
]

class JarvisRealFiscalSim:
    def __init__(self):
        print("🏛️ [JARVIS FISCAL] Inicializando Simulação com Taxas e IR...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def apply_fiscal_logic(self, saldo_bruto, lucro_bruto):
        """Aplica taxas B3 e desconta IR de 20% sobre o lucro líquido."""
        if lucro_bruto <= 0:
            return saldo_bruto # Sem IR se houver prejuízo, mas as taxas já foram deduzidas no loop
        
        imposto = lucro_bruto * IMPOSTO_RENDA_PCT
        return saldo_bruto - imposto

    def run_dl(self, df):
        df = df.copy()
        df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
        df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df = df.dropna()
        if df.empty: return -100, 0, 0, 0
        
        X = self.scaler.transform(df[self.features_cols].values)
        probs = self.model.predict(X, verbose=0).flatten()
        
        saldo = CAPITAL_INICIAL
        trades = 0
        taxas_pagas = 0
        
        for i in range(len(df)-1):
            sinal = 1 if probs[i] > 0.55 else (-1 if probs[i] < 0.45 else 0)
            if sinal != 0:
                lote = int(saldo / df["Close"].iloc[i])
                if lote > 0:
                    ret = (df["Close"].iloc[i+1] / df["Close"].iloc[i]) - 1
                    custo_b3 = (lote * df["Close"].iloc[i]) * TAXA_B3_TOTAL
                    taxas_pagas += custo_b3
                    saldo += (saldo * (sinal * ret)) - custo_b3
                    trades += 1
        
        lucro_bruto = saldo - CAPITAL_INICIAL
        saldo_pos_ir = self.apply_fiscal_logic(saldo, lucro_bruto)
        return (saldo_pos_ir / CAPITAL_INICIAL - 1) * 100, trades, saldo_pos_ir, taxas_pagas

    def run_momentum(self, df):
        df = df.copy()
        df['Max_10'] = df['High'].rolling(10).max().shift(1); df['Min_10'] = df['Low'].rolling(10).min().shift(1); df['SMA_50'] = df['Close'].rolling(50).mean()
        saldo = CAPITAL_INICIAL; pos = 0; p_in = 0; trades = 0; taxas_pagas = 0
        for i in range(50, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0 and close > df['Max_10'].iloc[i] and close > df['SMA_50'].iloc[i]:
                pos = 1; p_in = close; trades += 1
            elif pos == 1 and close < df['Min_10'].iloc[i]:
                ret = (close / p_in) - 1
                custo_b3 = (CAPITAL_INICIAL * TAXA_B3_TOTAL)
                taxas_pagas += custo_b3
                saldo += (saldo * ret) - custo_b3
                pos = 0
        lucro_bruto = saldo - CAPITAL_INICIAL
        saldo_pos_ir = self.apply_fiscal_logic(saldo, lucro_bruto)
        return (saldo_pos_ir / CAPITAL_INICIAL - 1) * 100, trades, saldo_pos_ir, taxas_pagas

    def run_reversion(self, df):
        df = df.copy()
        df["SMA_20"] = df["Close"].rolling(20).mean(); df["std"] = df["Close"].rolling(20).std(); df["BB_Lower"] = df["SMA_20"] - (2 * df["std"])
        delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        saldo = CAPITAL_INICIAL; pos = 0; p_in = 0; trades = 0; taxas_pagas = 0
        for i in range(20, len(df)):
            close = df['Close'].iloc[i]
            if pos == 0 and close < df['BB_Lower'].iloc[i] and df['RSI'].iloc[i] < 30:
                pos = 1; p_in = close; trades += 1
            elif pos == 1 and close > df['SMA_20'].iloc[i]:
                ret = (close / p_in) - 1
                custo_b3 = (CAPITAL_INICIAL * TAXA_B3_TOTAL)
                taxas_pagas += custo_b3
                saldo += (saldo * ret) - custo_b3
                pos = 0
        lucro_bruto = saldo - CAPITAL_INICIAL
        saldo_pos_ir = self.apply_fiscal_logic(saldo, lucro_bruto)
        return (saldo_pos_ir / CAPITAL_INICIAL - 1) * 100, trades, saldo_pos_ir, taxas_pagas

    def run_full_fiscal_sim(self):
        start_t = datetime.now()
        print(f"\n{'='*120}")
        print(f"💰 RELATÓRIO REALÍSTICO (COM TAXAS B3 E IMPOSTO DE RENDA 20%)")
        print(f"CAPITAL INICIAL: R$ {CAPITAL_INICIAL:,.2f} | PERÍODO: 1 ANO")
        print(f"{'='*120}")
        print(f"{'ATIVO':<12} | {'ESTRATÉGIA':<14} | {'P&L LÍQUIDO':<12} | {'TRADES':<8} | {'TAXAS B3':<10} | {'SALDO FINAL'}")
        print("-" * 120)
        
        relatorio = []
        for ticker in ATIVOS:
            df = self.get_data(ticker)
            if df is None: continue
            
            dl_p, dl_t, dl_s, dl_tax = self.run_dl(df)
            mom_p, mom_t, mom_s, mom_tax = self.run_momentum(df)
            rev_p, rev_t, rev_s, rev_tax = self.run_reversion(df)
            
            res_list = [
                {"name": "DeepLearning", "p": dl_p, "t": dl_t, "s": dl_s, "tax": dl_tax},
                {"name": "Momentum", "p": mom_p, "t": mom_t, "s": mom_s, "tax": mom_tax},
                {"name": "Reversion", "p": rev_p, "t": rev_t, "s": rev_s, "tax": rev_tax}
            ]
            
            best = max(res_list, key=lambda x: x['p'])
            print(f"{ticker:<12} | {best['name']:<14} | {best['p']:>10.2f}% | {best['t']:>8} | R$ {best['tax']:>7.2f} | R$ {best['s']:,.2f}")
            relatorio.append({"Ativo": ticker, **best})
            
        end_t = datetime.now()
        tot_final = sum([x['s'] for x in relatorio])
        tot_investido = CAPITAL_INICIAL * len(relatorio)
        tot_taxas = sum([x['tax'] for x in relatorio])
        tot_ir = sum([(x['s']/(1-IMPOSTO_RENDA_PCT) - x['s']) if x['p']>0 else 0 for x in relatorio]) # Cálculo aproximado do IR pago
        
        print(f"\n{'='*120}")
        print(f"🏁 CONCLUSÃO FISCAL")
        print(f"{'='*120}")
        print(f"⌛ Duração:             {end_t - start_t}")
        print(f"💸 Total Taxas B3:     R$ {tot_taxas:,.2f}")
        print(f"🏛️ Est. Imposto Renda: R$ {tot_ir:,.2f}")
        print(f"🚀 Patrimônio LÍQUIDO: R$ {tot_final:,.2f}")
        print(f"📉 Rentabilidade Média: {((tot_final/tot_investido)-1)*100:.2f}%")
        print(f"{'='*120}")

if __name__ == "__main__":
    sim = JarvisRealFiscalSim()
    sim.run_full_fiscal_sim()
