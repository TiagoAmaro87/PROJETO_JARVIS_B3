import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. Carregar Modelo e Scaler (O Cérebro e os Olhos)
model = tf.keras.models.load_model("BRAIN (DL)/modelo_global_b3.h5")
scaler = joblib.load("BRAIN (DL)/scaler_global.pkl")

# 2. Carregar os dados processados (Últimos 500 registros)
df = pd.read_csv("FEATURE_ENGINEERING/ALL_STOCKS_processed.csv").tail(500)

# 3. Seleção de Features (Exatamente como no treino!)
features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
X = df[features_cols].values
X_scaled = scaler.transform(X)

# 4. Predição da IA (Probabilidade de Alta)
df['Probability'] = model.predict(X_scaled)

# 5. REGRAS DE EXECUÇÃO (Onde o Scalping e a Tendência se encontram)
# Compra: Probabilidade > 52% | Venda: Probabilidade < 48%
df['Signal'] = 0
df.loc[df['Probability'] > 0.52, 'Signal'] = 1  # BUY
df.loc[df['Probability'] < 0.48, 'Signal'] = -1 # SELL (SHORT)

# 6. Cálculo de Performance
df['Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']

# 7. Relatório Final
print("\n" + "="*40)
print("📊 RELATÓRIO DE PERFORMANCE JARVIS_B3")
print("="*40)
cumulative_ret = (df['Strategy_Returns'].add(1).cumprod().iloc[-1] - 1) * 100
print(f"📈 Retorno Acumulado (Período): {cumulative_ret:.2f}%")
print(f"📉 Pior Drawdown Estimado: {df['Strategy_Returns'].min()*100:.2f}%")
print(f"🎯 Assertividade Média: {len(df[df['Strategy_Returns'] > 0]) / len(df[df['Signal'] != 0]) * 100:.1f}%")
print("="*40)

# Ver os últimos sinais gerados
print("\n👀 ÚLTIMOS 5 SINAIS GERADOS:")
print(df[['Close', 'Probability', 'Signal']].tail(5))