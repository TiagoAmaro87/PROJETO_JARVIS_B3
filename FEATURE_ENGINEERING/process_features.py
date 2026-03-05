import pandas as pd
import glob
import os
import numpy as np

def process_all_files():
    path = "DATA_INGESTION/RAW_ASSETS/*.csv"
    files = glob.glob(path)
    full_df = []

    for file in files:
        df = pd.read_csv(file, index_col=0, header=[0,1])
        df.columns = df.columns.get_level_values(0)
        
        # --- 1. TENDÊNCIA & LATERALIDADE ---
        df['SMA_9'] = df['Close'].rolling(window=9).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Width'] = (std * 4) / df['SMA_20']
        
        # --- 2. MOMENTO & REVERSÃO ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # --- 3. SCALPING & VOLATILIDADE (O QUE FALTOU) ---
        # ATR (Average True Range) - Mede a "vibe" do mercado para alvos curtos
        high_low = df['High'] - df['Low']
        high_cp = abs(df['High'] - df['Close'].shift())
        low_cp = abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
        
        # ROC (Rate of Change) - Velocidade do preço (Essencial para Scalper)
        df['ROC'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        
        # --- 4. ROMPIMENTO ---
        df['Vol_Shock'] = df['Volume'] / df['Volume'].rolling(20).mean() # Volume acima da média
        
        # --- ALVO ---
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        full_df.append(df.dropna())

    final_data = pd.concat(full_df)
    final_data.to_csv("FEATURE_ENGINEERING/ALL_STOCKS_processed.csv", index=False)
    print(f"✅ Setup SCALPER/TREND/REVERSAL Concluído!")

if __name__ == "__main__":
    process_all_files()