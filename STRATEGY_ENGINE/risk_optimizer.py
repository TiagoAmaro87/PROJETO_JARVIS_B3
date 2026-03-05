import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. Setup
model = tf.keras.models.load_model("BRAIN (DL)/modelo_global_b3.h5")
scaler = joblib.load("BRAIN (DL)/scaler_global.pkl")
# IMPORTANTE: Usamos dados que a rede NÃO viu (separe os últimos 20% do CSV)
df = pd.read_csv("FEATURE_ENGINEERING/ALL_STOCKS_processed.csv")
train_size = int(len(df) * 0.8)
df_test = df.iloc[train_size:].copy() 

# 2. Predição
features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
X_test_scaled = scaler.transform(df_test[features_cols].values)
df_test['Probability'] = model.predict(X_test_scaled)

# 3. Execução Rigorosa (Sinal de hoje reflete no trade de AMANHÃ)
df_test['Signal'] = 0
df_test.loc[df_test['Probability'] > 0.55, 'Signal'] = 1
df_test.loc[df_test['Probability'] < 0.45, 'Signal'] = -1

# O PULO DO GATO: O retorno da estratégia é o sinal de ONTEM vezes o retorno de HOJE
df_test['Returns'] = df_test['Close'].pct_change()
df_test['Strategy_Returns'] = df_test['Signal'].shift(1) * df_test['Returns']

# 4. Custos e Stop
custo = 0.0005
df_test['Trades'] = df_test['Signal'].diff().abs()
df_test.loc[df_test['Strategy_Returns'] < -0.02, 'Strategy_Returns'] = -0.02
df_test['Net_Returns'] = df_test['Strategy_Returns'] - (df_test['Trades'] * custo)

# 5. Resultado Real
real_ret = (df_test['Net_Returns'].add(1).cumprod().iloc[-1] - 1) * 100
print(f"📊 RETORNO REAL (Sem olhar o futuro): {real_ret:.2f}%")