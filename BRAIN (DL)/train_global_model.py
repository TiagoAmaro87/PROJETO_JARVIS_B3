import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Carregar o Dataset Mestre (Versão com Scalping)
print("💾 Carregando dados unificados com setup SCALPER...")
file_path = "FEATURE_ENGINEERING/ALL_STOCKS_processed.csv"

if not os.path.exists(file_path):
    print(f"❌ Erro: O arquivo {file_path} não foi encontrado! Rode o process_features.py primeiro.")
    exit()

df = pd.read_csv(file_path)

# 2. Seleção de Features COMPLETA
# Close, Volume, SMA_9, SMA_20, BB_Width, RSI, ATR, ROC, Vol_Shock
features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
X = df[features_cols].values
y = df['Target'].values

# 3. Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SALVANDO O SCALER
joblib.dump(scaler, "BRAIN (DL)/scaler_global.pkl")
print("✅ Scaler atualizado e salvo!")

# 4. Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# 5. Arquitetura da Rede Neural (Potencializada para mais variáveis)
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'), # Aumentamos para 256 neurônios
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Treinamento
print("\n🚀 Iniciando treinamento do Jarvis_B3 (V3 - Setup Profissional)...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=64, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# 7. Salvar o Modelo Final
model.save("BRAIN (DL)/modelo_global_b3.h5")
print("\n==============================================")
print("✅ TREINAMENTO CONCLUÍDO!")
print(f"📍 Acurácia de Validação Final: {history.history['val_accuracy'][-1]:.4f}")
print("==============================================")