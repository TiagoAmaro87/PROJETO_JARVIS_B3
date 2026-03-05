import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

# CONFIGURAÇÃO DE CAMINHOS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
SCALER_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")

TICKER_CEGO = "PRIO3.SA" 

print(f"\n🕵️ Jarvis_B3: Teste Cego em {TICKER_CEGO}")

# Carregamento do modelo
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 1. Coleta de dados e limpeza de MultiIndex
df = yf.download(TICKER_CEGO, period="1y", interval="1d", progress=False)

# Se as colunas forem MultiIndex (comum em versões novas do yfinance), achata elas
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Engenharia de Features
df['SMA_9'] = df['Close'].rolling(window=9).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
# BB_Width corrigido para garantir que seja 1D
std_dev = df['Close'].rolling(window=20).std()
df['BB_Width'] = (std_dev * 4) / df['SMA_20']

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + (gain / loss)))

df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
df['ROC'] = df['Close'].pct_change(5) * 100
df['Vol_Shock'] = df['Volume'] / df['Volume'].rolling(20).mean()

df = df.dropna()

# 3. Predição
features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
X = scaler.transform(df[features_cols].values)
df['Probability'] = model.predict(X, verbose=0)

# 4. Backtest Realista (Atraso de Execução T+1)
df['Signal'] = 0
df.loc[df['Probability'] > 0.55, 'Signal'] = 1
df.loc[df['Probability'] < 0.45, 'Signal'] = -1

# O sinal de ontem opera no fechamento de hoje
df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
# Custo operacional (Taxas B3)
df['Net_Returns'] = df['Strategy_Returns'] - (df['Signal'].diff().abs() * 0.0005)

lucro = (df['Net_Returns'].add(1).cumprod().iloc[-1] - 1) * 100

print("\n" + "="*45)
print(f"🚀 RESULTADO NO TESTE CEGO ({TICKER_CEGO})")
print(f"💰 Lucro Líquido Realista: {lucro:.2f}%")
print(f"📊 Trades Realizados: {int(df['Signal'].diff().abs().sum())}")
print("="*45)