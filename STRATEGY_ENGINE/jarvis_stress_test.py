import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURAÇÃO ---
BASE_DIR = r"C:\Users\tiago\OneDrive\Área de Trabalho\PROJETO_JARVIS_B3"
CAPITAL_INICIAL = 1000.0
SIMULACOES = 1000  # Número de cenários aleatórios
DIAS_PROJECAO = 252 # 1 ano de pregão

class JarvisStressTester:
    def __init__(self):
        print(f"🎲 [STRESS TEST] Iniciando Simulação de Monte Carlo ({SIMULACOES} cenários)...")

    def get_historical_returns(self, ticker):
        df = yf.download(ticker + ".SA", period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df['Close'].pct_change().dropna()

    def run_monte_carlo(self, ticker):
        returns = self.get_historical_returns(ticker)
        if returns is None: return
        
        mu = returns.mean()
        sigma = returns.std()
        
        # Matriz de simulação: (Dias x Simulações)
        simulated_returns = np.random.normal(mu, sigma, (DIAS_PROJECAO, SIMULACOES))
        
        # Evolução do capital: Capital * cumprod(1 + returns)
        price_paths = CAPITAL_INICIAL * (1 + simulated_returns).cumprod(axis=0)
        
        # Resultados Finais
        final_values = price_paths[-1]
        
        # Métricas de Estresse
        var_95 = np.percentile(final_values, 5) # 5% pior cenário (VaR)
        median_gain = np.percentile(final_values, 50)
        top_gain = np.percentile(final_values, 95)
        
        print(f"\n📊 RESULTADOS DE ESTRESSE: {ticker}")
        print(f"{'-'*40}")
        print(f"💰 Capital Inicial:      R$ {CAPITAL_INICIAL:,.2f}")
        print(f"📉 Pior Cenário (5%):    R$ {var_95:,.2f}")
        print(f"⚖️  Cenário Provável:    R$ {median_gain:,.2f}")
        print(f"🚀 Melhor Cenário (95%): R$ {top_gain:,.2f}")
        
        # Probabilidade de Quebrar (Perder > 50% do capital)
        prob_quebra = (final_values < CAPITAL_INICIAL * 0.5).mean()
        print(f"💀 Prob. Ruína (>50%):  {prob_quebra:.2%}")
        
        return price_paths

    def plot_results(self, price_paths, ticker):
        plt.figure(figsize=(10,6))
        plt.plot(price_paths[:, :100], color='blue', alpha=0.05) # Plota as primeiras 100 trajetórias
        plt.plot(np.percentile(price_paths, 50, axis=1), color='black', linewidth=2, label='Mediana')
        plt.plot(np.percentile(price_paths, 5, axis=1), color='red', linestyle='--', label='Pior Caso (5%)')
        plt.axhline(CAPITAL_INICIAL, color='green', linestyle='-', alpha=0.3)
        plt.title(f"Simulação de Monte Carlo - Futuro de {ticker} (1 ano)")
        plt.ylabel("Patrimônio (R$)")
        plt.xlabel("Dias de Pregão")
        plt.legend()
        
        # Salva o gráfico como imagem para controle
        img_path = os.path.join(BASE_DIR, "STRATEGY_ENGINE", f"monte_carlo_{ticker}.png")
        plt.savefig(img_path)
        print(f"📈 Gráfico salvo em: {img_path}")

if __name__ == "__main__":
    tester = JarvisStressTester()
    # Testando nos 3 principais motores do nosso robô
    ativos = ["PETR4", "MGLU3", "ITUB4"]
    for ativo in ativos:
        paths = tester.run_monte_carlo(ativo)
        if paths is not None:
             tester.plot_results(paths, ativo)
