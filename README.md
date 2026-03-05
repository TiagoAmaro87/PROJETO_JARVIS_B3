# 📈 B3 Deep Intelligence System (JARVIS_B3)

Bem-vindo ao repositório do **JARVIS_B3** - O Sistema Quantitativo Híbrido de Trading Automatizado para o Mercado Brasileiro (B3).

## 🚀 Visão Geral
O JARVIS_B3 é um bot de negociação algorítmica desenhado para operar com inteligência de ponta (Deep Learning, Transformers e LSTMs) focada no comportamento da B3. Com arquitetura modular, ele consegue digerir dados históricos e contextuais (On-Chain/Sentiment) para buscar a melhor performance ajustada ao risco, operando prioritariamente via **Profit Pro**.

## 🏗 Arquitetura Modular

1. **`DATA_INGESTION/`**
   * Responsável por coletar e unificar dados da B3 (APIs como MetaTrader 5, Yahoo Finance, etc.).
2. **`FEATURE_ENGINEERING/`**
   * Transforma os dados brutos em indicadores preditivos (Vol_Shock, BB_Width, ATR, etc).
3. **`BRAIN (DL)/`**
   * Onde residem os modelos de Deep Learning. Treinados com PyTorch/TensorFlow para buscar padrões complexos.
4. **`STRATEGY_ENGINE/`**
   * Motor de lógica das métricas de trading, decidindo os Momentums de entrada (Sentiment, Reversão à média, Breakout).
5. **`RISK_MANAGER/`**
   * Módulo de "Coração" do projeto. Focado em gestão estrita de capital (VaR, Expected Shortfall, Monte Carlo). Nunca permite "setups mágicos" sobreporem um bom risco/retorno.
6. **`BOT_EXECUTION/`**
   * Integração de tela e bot (automação). Onde o robô toma os sinais e executa de fato na tela da B3 (Profit Pro/MT5), garantindo travas de segurança.

## 🔒 Segurança (Trava Anti-Leak de Ativo)
O sistema conta com validação forçada de ticker para o **Profit Pro**. Antes de disparar qualquer ordem via PyAutoGUI, o robô digita automaticamente o ativo correto e confirma o foco do gráfico, blindando sua conta de "vazamento" de ordens para telas erradas.

## 🧪 Validação Quant (Em Desenvolvimento)
O padrão do framework de risco garante a resiliência via:
* **Backtesting Massivo (Monte Carlo):** Resiliência sob caudas longas do mercado.
* **Métricas de Risco:** VaR (Value at Risk) e ES (Expected Shortfall).
* **Detecção de Drift:** Aviso de necessidade de retreino do modelo em caso de degradação.
* **Métricas Base:** Expectativa Matemática e Sharpe Ratio.

## 🛠 Stack Tecnológica
- **Linguagem:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras (Arquivos `.h5` de Brain)
- **Engenharia de Dados:** Pandas, NumPy, Scikit-Learn
- **Execução Automática:** PyAutoGUI, PyGetWindow, YFinance

---

*Desenvolvido sob rígido gerenciamento de risco e auditoria profunda.*
