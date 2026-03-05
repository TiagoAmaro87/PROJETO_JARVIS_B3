import os
import sys
import time

print("1. Importando TensorFlow...")
import tensorflow as tf
print("2. TensorFlow importado.")

BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
MODEL_PATH = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")

print(f"3. Tentando carregar o modelo em: {MODEL_PATH}")
start = time.time()
model = tf.keras.models.load_model(MODEL_PATH)
end = time.time()
print(f"4. Modelo carregado com sucesso em {end - start:.2f} segundos!")
