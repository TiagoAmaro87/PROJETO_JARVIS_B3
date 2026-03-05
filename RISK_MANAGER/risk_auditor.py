import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Adiciona o diretório base para poder importar os módulos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from RISK_MANAGER.quant_framework import JarvisQuantFramework

class JarvisAuditor:
    def __init__(self):
        print(f"🔎 [AUDITORIA RISK_MANAGER] Iniciando motor de validação quantitativa...")
        
        self.model_path = os.path.join(BASE_DIR, "BRAIN (DL)", "modelo_global_b3.h5")
        self.scaler_path = os.path.join(BASE_DIR, "BRAIN (DL)", "scaler_global.pkl")
        self.data_path = os.path.join(BASE_DIR, "FEATURE_ENGINEERING", "ALL_STOCKS_processed.csv")
        
        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.quant = JarvisQuantFramework()
        
        self.features_cols = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
        
    def run_backtest_with_risk_metrics(self, data_file=None, prob_threshold_buy=0.55, prob_threshold_sell=0.45):
        print("💾 Carregando Dataset para SIMULAÇÃO...")
        df = pd.read_csv(data_file or self.data_path)
        
        if df.empty:
            print("⚠️ Erro: Dataset vazio.")
            return

        # Para fins de simulação de longo prazo, vamos pegar apenas as últimas 2.000 linhas
        # para simular o comportamento de uma carteira
        df = df.tail(2000).copy()
        
        X = df[self.features_cols].values
        X_scaled = self.scaler.transform(X)
        
        print("🧠 Realizando predições históricas pelo modelo de DL...")
        preds = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Estratégia BDIS simulada
        df['Pred_Prob'] = preds
        df['Signal'] = 0
        df.loc[df['Pred_Prob'] > prob_threshold_buy, 'Signal'] = 1
        df.loc[df['Pred_Prob'] < prob_threshold_sell, 'Signal'] = -1
        
        # Calcular Retornos da Estratégia
        # Como o target avaliado no treino era Close_{t+1} > Close_t,
        # O retorno da operação é log_return do dia seguinte * Signal
        df['Return'] = df['Close'].pct_change()
        
        # A estratégia obtém o ganho ou perda baseando-se no sinal do período anterior
        # Porque a decisão de compra só pode ser avaliada pelo mercado do amanhã
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
        df = df.dropna()
        
        # Extrair a série de retornos
        strategy_returns = df['Strategy_Return']
        
        # Não obteve retornos por falta de sinais?
        if len(strategy_returns[strategy_returns != 0]) == 0:
            print("⚠️ Não houve sinais suficientes acionados pelos thresholds informados.")
            return

        print("\n" + "="*50)
        print("📊 RELATÓRIO DE AUDITORIA QUANTITATIVA JARVIS_B3")
        print("="*50)
        
        print("\n🔴 1. ANÁLISE DE RISCOS DE CAUDA (TAIL RISK)")
        var_95 = self.quant.value_at_risk(strategy_returns)
        es_95 = self.quant.expected_shortfall(strategy_returns)
        print(f"Value at Risk (VaR 95%): {var_95:.2%}")
        print(f"Expected Shortfall (ES 95%): {es_95:.2%} (Média das perdas quando o VaR é rompido)")
        
        print("\n📈 2. MÉTRICAS BASE DO PORTFÓLIO")
        sharpe = self.quant.calculate_sharpe_ratio(strategy_returns)
        max_dd = self.quant.maximum_drawdown(strategy_returns)
        
        # Expetativa Matemática simples
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns[strategy_returns != 0])
        avg_win = strategy_returns[strategy_returns > 0].mean()
        avg_loss = strategy_returns[strategy_returns < 0].mean()
        ev = self.quant.mathematical_expectancy(win_rate, avg_win, avg_loss)
        
        print(f"Índice de Sharpe (Anualizado): {sharpe:.2f}")
        print(f"Drawdown Máximo: {max_dd:.2%}")
        print(f"Taxa de Acerto (Win Rate): {win_rate:.2%}")
        print(f"Expectativa Matemática (EV/Trade): {ev:.4f} % no capital")

        print("\n🎲 3. SIMULAÇÃO DE ESTRÉSSE MONTE CARLO (252 DIAS)")
        mc_results = self.quant.monte_carlo_simulation(strategy_returns, num_simulations=500, periods=252)
        print(f"Mediana Esperada (Final 1 Ano): {mc_results['Median_Case']:.2f}x o Capital Investido")
        print(f"Cenário Apocalíptico (Pior 1%): {mc_results['Worst_Case_1%']:.2f}x o Capital Investido")
        
        # Verificando Drift com métrica in_sample teórica de 0.05% log_return diário
        print("\n🚨 4. MONITORAMENTO DE MODEL DRIFT")
        self.quant.detect_model_drift(strategy_returns.tail(30), in_sample_mean=0.0005, threshold=2.0)
        
        print("="*50)

if __name__ == "__main__":
    auditor = JarvisAuditor()
    auditor.run_backtest_with_risk_metrics()
