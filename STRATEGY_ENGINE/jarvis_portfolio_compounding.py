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

# --- CONFIGURAÇÃO DE CARTEIRA ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

CAPITAL_PORTFOLIO_INICIAL = 1000.0
PERCENTUAL_MAX_POR_ATIVO = 0.25  # No máximo 25% do capital em um único papel (Diversificação)
TAXA_B3_TOTAL = 0.0006 
IMPOSTO_RENDA_PCT = 0.20

# Mapeamento Vencedor (Expert por Ativo) detectado anteriormente
EXPERT_MAP = {
    "PETR4.SA": "DeepLearning", "VALE3.SA": "DeepLearning", "ITUB4.SA": "DeepLearning",
    "BBDC4.SA": "DeepLearning", "ABEV3.SA": "DeepLearning", "MGLU3.SA": "DeepLearning",
    "BBAS3.SA": "DeepLearning", "RENT3.SA": "DeepLearning", "RADL3.SA": "Momentum",
    "RAIL3.SA": "Reversion", "GGBR4.SA": "DeepLearning", "CSNA3.SA": "DeepLearning",
    "B3SA3.SA": "DeepLearning", "HYPE3.SA": "Reversion", "LREN3.SA": "Reversion"
}

class JarvisPortfolioCompoundingSim:
    def __init__(self):
        print("🚀 [JARVIS COMPOUND] Iniciando Simulação de Carteira Global...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']

    def get_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def get_signals(self, ticker, expert, df):
        # Gera os sinais 1 (compra), -1 (venda/saída) ou 0 (nada)
        df = df.copy()
        if expert == "DeepLearning":
            df["SMA_9"] = df["Close"].rolling(9).mean(); df["SMA_20"] = df["Close"].rolling(20).mean()
            df["BB_Width"] = (df["Close"].rolling(20).std() * 4) / df["SMA_20"]
            delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df["RSI"] = 100 - (100 / (1 + (gain / loss))); df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
            df["ROC"] = df["Close"].pct_change(5) * 100; df["Vol_Shock"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df = df.dropna()
            if df.empty: return pd.Series()
            X = self.scaler.transform(df[self.features_cols].values)
            probs = self.model.predict(X, verbose=0).flatten()
            signals = pd.Series(0, index=df.index)
            signals[probs > 0.55] = 1
            signals[probs < 0.45] = -1
            return signals
            
        elif expert == "Momentum":
            max_10 = df['High'].rolling(10).max().shift(1); min_10 = df['Low'].rolling(10).min().shift(1); sma_50 = df['Close'].rolling(50).mean()
            signals = pd.Series(0, index=df.index)
            signals[(df['Close'] > max_10) & (df['Close'] > sma_50)] = 1
            signals[df['Close'] < min_10] = -1
            return signals

        elif expert == "Reversion":
            sma_20 = df["Close"].rolling(20).mean(); std = df["Close"].rolling(20).std(); lower = sma_20 - (2 * std)
            delta = df["Close"].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))
            signals = pd.Series(0, index=df.index)
            signals[(df['Close'] < lower) & (rsi < 30)] = 1
            signals[df['Close'] > sma_20] = -1
            return signals

    def run(self):
        # 1. Carregar todos os sinais
        all_data = {}
        all_signals = {}
        for ticker, expert in EXPERT_MAP.items():
            df = self.get_data(ticker)
            if df is not None:
                all_data[ticker] = df
                all_signals[ticker] = self.get_signals(ticker, expert, df)
        
        # 2. Sincronizar datas (Timeline Global)
        dates = sorted(list(set().union(*(sig.index for sig in all_signals.values()))))
        
        equity = CAPITAL_PORTFOLIO_INICIAL
        positions = {} # {ticker: {'qty': x, 'p_in': y}}
        history = []
        lucro_acumulado_ano = 0

        print(f"\n{'='*100}")
        print(f"💰 SIMULAÇÃO DE CARTEIRA COM JUROS COMPOSTOS (CAPITAL TOTAL: R$ {CAPITAL_PORTFOLIO_INICIAL})")
        print(f"{'='*100}")
        
        for date in dates:
            # 2.1 Verificar saídas primeiro (para liberar capital)
            to_remove = []
            for ticker, pos in positions.items():
                if date in all_signals[ticker].index:
                    sinal = all_signals[ticker].loc[date]
                    if sinal == -1: # Sinal de venda ou saída
                        price_out = all_data[ticker].loc[date, 'Close']
                        ret = (price_out / pos['p_in']) - 1
                        val_bruto = (pos['qty'] * price_out)
                        taxa = val_bruto * TAXA_B3_TOTAL
                        equity += val_bruto - taxa
                        lucro_trade = (val_bruto - (pos['qty'] * pos['p_in'])) - taxa
                        if lucro_trade > 0:
                            ir = lucro_trade * IMPOSTO_RENDA_PCT
                            equity -= ir # Desconta IR real
                        to_remove.append(ticker)
            
            for t in to_remove: del positions[t]

            # 2.2 Verificar entradas
            for ticker, expert in EXPERT_MAP.items():
                if ticker not in positions and date in all_signals[ticker].index:
                    sinal = all_signals[ticker].loc[date]
                    if sinal == 1:
                        # Position Sizing: Usa no máximo 25% do equity atual por trade
                        capital_aloc = equity * PERCENTUAL_MAX_POR_ATIVO
                        price_in = all_data[ticker].loc[date, 'Close']
                        
                        # Simulação do Mercado Fracionário (pode comprar 1 ação se precisar)
                        qty = int(capital_aloc / price_in)
                        if qty > 0:
                            val_entrada = qty * price_in
                            taxa = val_entrada * TAXA_B3_TOTAL
                            equity -= (val_entrada + taxa)
                            positions[ticker] = {'qty': qty, 'p_in': price_in}
            
            # Valor da carteira no dia (Equity + Posições abertas)
            current_total = equity
            for t, p in positions.items():
                if date in all_data[t].index:
                    current_total += p['qty'] * all_data[t].loc[date, 'Close']
            
            history.append(current_total)

        final_equity = history[-1]
        print(f"📊 RESULTADO FINAL DA CARTEIRA:")
        print(f"🔹 Patrimônio Inicial:  R$ {CAPITAL_PORTFOLIO_INICIAL:,.2f}")
        print(f"🔹 Patrimônio FINAL:    R$ {final_equity:,.2f} (Líquido de tudo)")
        print(f"🚀 Rentabilidade Total: {((final_equity/CAPITAL_PORTFOLIO_INICIAL)-1)*100:.2f}%")
        print(f"📈 Benchmarking (Juros Compostos Ativos): SIM")
        print(f"{'='*100}")

if __name__ == "__main__":
    sim = JarvisPortfolioCompoundingSim()
    sim.run()
