import yfinance as yf
import pandas as pd
import os

# Lista das principais ações da B3 (Exemplo ampliado)
# Podemos expandir essa lista conforme necessário
tickers = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 
    'BBAS3.SA', 'B3SA3.SA', 'ITSA4.SA', 'MGLU3.SA', 'GGBR4.SA',
    'CSNA3.SA', 'JBSS3.SA', 'WEGE3.SA', 'RENT3.SA', 'VVAR3.SA'
]

def collect_all_tickers():
    if not os.path.exists("DATA_INGESTION/RAW_ASSETS"):
        os.makedirs("DATA_INGESTION/RAW_ASSETS")

    for symbol in tickers:
        print(f"Baixando: {symbol}...")
        try:
            df = yf.download(symbol, period="1y", interval="1d") # 1 ano de dados diários
            if not df.empty:
                df.to_csv(f"DATA_INGESTION/RAW_ASSETS/{symbol}_raw.csv")
        except Exception as e:
            print(f"Erro ao baixar {symbol}: {e}")

if __name__ == "__main__":
    collect_all_tickers()
    print("✅ Coleta em massa finalizada!")