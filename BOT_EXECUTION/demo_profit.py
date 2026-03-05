import sys
import os
import time

# Adicionando path
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
sys.path.append(BASE_DIR)

from BOT_EXECUTION.bdis_executor import JarvisB3Full

def executar_demonstracao_visual():
    print("=============================================================")
    print("🚀 JARVIS B3 - MODO DE DEMONSTRAÇÃO VISUAL (TEMPO REAL)")
    print("=============================================================")
    print("⚠️ ATENÇÃO: Estou assumindo o controle do mouse/teclado.")
    print("Por favor, GARANTA QUE O PROFIT PRO ESTEJA ABERTO (preferencialmente no Simulador).")
    print("\nO Bot tentará:")
    print("1. Encontrar a janela do Profit")
    print("2. Forçar a mudança do ativo para VALE3 (Trava de Segurança)")
    print("3. Executar o atalho de Compra (F5 + Enter)")
    print("=============================================================\n")
    
    for i in range(15, 0, -1):
        print(f"⏳ Iniciando injeção na GUI em {i} segundos...")
        time.sleep(1)

    print("\n[MOCK] Iniciando motor...")
    bot = JarvisB3Full()
    
    print("\n🤖 [MOCK-AI] A Inteligência Artificial (Modelo DL) calculou 98.5% de probabilidade de Alta.")
    print("📊 [MOCK-RISK] O Framework Quantitativo aprovou a volatilidade (VaR OK).")
    print("🎯 [EXECUTION] Enviando sinal para o módulo de Boleta...\n")

    # Injeta um sinal de compra falso na Vale3
    bot.focar_e_boletar(sinal=1, ativo="VALE3.SA")

    print("\n=============================================================")
    print("✅ DEMONSTRAÇÃO VISUAL CONCLUÍDA!")
    print("Verifique no seu Profit se o ativo foi trocado para VALE3 e se a boleta subiu.")
    print("=============================================================")

if __name__ == "__main__":
    executar_demonstracao_visual()
