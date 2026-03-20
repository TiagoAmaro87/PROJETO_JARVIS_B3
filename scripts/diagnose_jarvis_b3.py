import sys
import os
import pandas as pd
import yfinance as yf
from loguru import logger
import importlib.util

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import logic for directory with spaces
spec = importlib.util.spec_from_file_location("brain_engine_v2", os.path.join(BASE_DIR, "BRAIN (DL)", "brain_engine_v2.py"))
brain_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brain_mod)
BrainEngineV2 = brain_mod.BrainEngineV2
compute_features = brain_mod.compute_features

def run_diagnostic():
    print(f"\n{'='*60}")
    print(" 🛸 [JARVIS B3] DIAGNÓSTICO DE INTELIGÊNCIA AGÊNTICA V2.5")
    print(f"{'='*60}\n")
    
    brain = BrainEngineV2(dry_run=True)
    if not brain.load_models():
        print("❌ Erro ao carregar modelos.")
        return

    ticker = "VALE3"
    print(f"🔍 Analisando {ticker} (B3)... Aguarde o processamento cognitivo...")
    
    # Download 60 days of data for simulation
    df = yf.download(f"{ticker}.SA", period="60d", interval="1d", progress=False)
    if df.empty:
        print("❌ Erro ao baixar dados.")
        return
        
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df = compute_features(df).dropna()
    if df.empty:
        print("❌ Dados insuficientes para análise.")
        return
    
    # Run the core logic
    prob = brain.predict(df)
    sentiment = brain.analyze_market_sentiment(df)
    
    print(f"\n🧠 [CAMADA 1: Modelos AI]")
    print(f"   - Confiança do XGBoost/MLP: {prob:.4f}")
    
    print(f"\n🛡️ [CAMADA 2: Sentinela Price Action]")
    print(f"   - Sentimento Curto Prazo: {sentiment['sentiment'].upper()}")
    print(f"   - Detectou Bull Trap? {'SIM ⚠️' if sentiment['trap_detected'] else 'Não'}")
    print(f"   - Volume Spike detectado? {'Sim' if sentiment['high_volume'] else 'Não'}")
    
    print(f"\n🔄 [CAMADA 3: Auto-Reflexão Agentic]")
    should_trade = brain.self_reflect(ticker, prob, sentiment)
    
    if should_trade:
        print(f"   ✅ RESULTADO: IA APROVOU a operação. Prosseguindo para execução.")
    else:
        print(f"   🚫 RESULTADO: IA ABORTOU. O modelo de ML é positivo, mas o contexto de mercado é desfavorável.")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    run_diagnostic()
