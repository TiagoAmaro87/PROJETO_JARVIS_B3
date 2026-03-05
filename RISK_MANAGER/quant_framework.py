import numpy as np
import pandas as pd
from scipy.stats import norm

class JarvisQuantFramework:
    """
    Framework Avançado Quantitativo (Padrão JARVIS_CRIPTO portado para B3)
    Responsável por auditoria de risco de cauda, backtesting massivo e monitoramento de Drift.
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    # ==========================================
    # 1. ANÁLISE DE RISCO EXTREMO
    # ==========================================
    def value_at_risk(self, returns_series: pd.Series) -> float:
        """
        Calcula o Value at Risk (VaR) Histórico.
        """
        if returns_series.empty:
            return 0.0
        # Ordena os retornos e pega o percentil correspondente
        alpha = 1 - self.confidence_level
        var = np.percentile(returns_series, alpha * 100)
        return var

    def expected_shortfall(self, returns_series: pd.Series) -> float:
        """
        Calcula o Expected Shortfall (CVaR).
        Média das perdas que ultrapassam o VaR calculado.
        """
        if returns_series.empty:
            return 0.0
        var = self.value_at_risk(returns_series)
        # Filtra os retornos que são piores (menores) que o VaR
        tail_losses = returns_series[returns_series < var]
        if len(tail_losses) == 0:
            return var
        return tail_losses.mean()

    # ==========================================
    # 2. MÉTRICAS BASE DO SISTEMA QUANT
    # ==========================================
    def calculate_sharpe_ratio(self, returns_series: pd.Series, risk_free_rate=0.0) -> float:
        """
        Calcula o Índice de Sharpe Diário 
        Idealmente multiplicar por sqrt(252) para anualizado na B3.
        """
        if returns_series.std() == 0:
            return 0.0
        sharpe = (returns_series.mean() - risk_free_rate) / returns_series.std()
        return sharpe * np.sqrt(252) # Anualizado B3

    def maximum_drawdown(self, returns_series: pd.Series) -> float:
        """
        Calcula o Maximum Drawdown
        """
        cumulative_returns = (1 + returns_series).cumprod()
        peak = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - peak) / peak
        return drawdowns.min()
        
    def mathematical_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calcula a Expectativa Matemática (EV)
        """
        loss_rate = 1 - win_rate
        ev = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        return ev

    # ==========================================
    # 3. BACKTESTING MASSIVO (MONTE CARLO)
    # ==========================================
    def monte_carlo_simulation(self, returns_series: pd.Series, num_simulations=1000, periods=252):
        """
        Realiza testes de stress Monte Carlo para projetar possíveis caminhos do portfólio.
        """
        mean_return = returns_series.mean()
        std_dev = returns_series.std()
        
        simulations = np.zeros((periods, num_simulations))
        
        for i in range(num_simulations):
            # Gera choques aleatórios baseados na distribuição normal (Random Walk)
            random_shocks = np.random.normal(mean_return, std_dev, periods)
            simulations[:, i] = (1 + random_shocks).cumprod()
            
        # Analisa o cenário de Pior Caso (1º percentil das simulações) no final do período
        final_values = simulations[-1, :]
        worst_case = np.percentile(final_values, 1)
        median_case = np.median(final_values)
        
        return {
            "Worst_Case_1%": worst_case,
            "Median_Case": median_case,
            "Simulations_Matrix": simulations
        }

    # ==========================================
    # 4. DETECÇÃO DE MODEL DRIFT (DEGRADAÇÃO)
    # ==========================================
    def detect_model_drift(self, out_of_sample_returns: pd.Series, in_sample_mean: float, threshold=2.0) -> bool:
        """
        Avalia se há degradação da performance OOS (Out-of-Sample) vs Treino.
        Retorna True se o modelo estiver "desviando" consideravelmente da norma.
        """
        if out_of_sample_returns.empty: return False
        
        # Teste T simples para comparar a média recente contra o esperado histórico
        recent_mean = out_of_sample_returns.mean()
        recent_std = out_of_sample_returns.std()
        
        n = len(out_of_sample_returns)
        if n < 10 or recent_std == 0: 
            return False # Dados insuficientes para prova de Drift
            
        t_statistic = (in_sample_mean - recent_mean) / (recent_std / np.sqrt(n))
        
        if t_statistic > threshold:
            print(f"⚠️ Alerta de Drift: O Sistema Quantitativo degradou (T-Stat {t_statistic:.2f} > {threshold})! Retreino de GPU obrigatório.")
            return True
        return False

# Exemplo Rápido de Uso para Auditoria
if __name__ == "__main__":
    # Dados Simulados B3
    np.random.seed(42)
    fake_b3_returns = pd.Series(np.random.normal(0.0005, 0.015, 252)) # Média 0.05% diária, ~1.5% volatilidade
    
    quant_engine = JarvisQuantFramework()
    print("----- RISCO -----")
    print(f"VaR 95%: {quant_engine.value_at_risk(fake_b3_returns):.2%}")
    print(f"Expected Shortfall 95%: {quant_engine.expected_shortfall(fake_b3_returns):.2%}")
    
    print("\n----- MÉTRICAS -----")
    print(f"Sharpe Ratio (Anualizado): {quant_engine.calculate_sharpe_ratio(fake_b3_returns):.2f}")
    print(f"Drawdown Máximo: {quant_engine.maximum_drawdown(fake_b3_returns):.2%}")
    
    print("\n----- MONTE CARLO (252 Dias | 1000 Simulações) -----")
    mc_results = quant_engine.monte_carlo_simulation(fake_b3_returns)
    print(f"Mediana Esperada: {mc_results['Median_Case']:.2f}x do Capital")
    print(f"Pior Cenário (Cauda 1%): {mc_results['Worst_Case_1%']:.2f}x do Capital")
