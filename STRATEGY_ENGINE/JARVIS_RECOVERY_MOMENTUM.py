import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ATIVOS PARA RECUPERAÇÃO (ESTRATÉGIA 2 - VERSÃO MOMENTUM)
ATIVOS_RECOVERY = ["RAIL3.SA", "RADL3.SA", "PRIO3.SA", "SUZB3.SA", "ABEV3.SA"]
CAPITAL_INICIAL = 10000.0

def testar_momentum_breakout(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Filtros de Momentum
        df['Max_10'] = df['High'].rolling(10).max().shift(1)
        df['Min_10'] = df['Low'].rolling(10).min().shift(1)
        df['SMA_50'] = df['Close'].rolling(50).mean() # Filtro de tendência macro

        saldo = CAPITAL_INICIAL
        posicao = 0 
        preco_entrada = 0
        total_trades = 0

        for i in range(10, len(df)):
            preco_atual = df['Close'].iloc[i]
            maxima_10 = df['Max_10'].iloc[i]
            minima_10 = df['Min_10'].iloc[i]
            trend_macro = df['SMA_50'].iloc[i]

            # COMPRA: Preço rompeu a máxima dos últimos 10 dias + está acima da média de 50
            if posicao == 0 and preco_atual > maxima_10 and preco_atual > trend_macro:
                posicao = 1
                preco_entrada = preco_atual
                total_trades += 1
            
            # VENDA: Preço perdeu a mínima dos últimos 10 dias (Trailing Stop)
            elif posicao == 1:
                if preco_atual < minima_10:
                    retorno = (preco_atual / preco_entrada) - 1
                    saldo *= (1 + retorno)
                    posicao = 0

        return {
            "Ativo": ticker,
            "Saldo Final (R$)": round(saldo, 2),
            "Lucro Real %": round(((saldo/CAPITAL_INICIAL)-1)*100, 2),
            "Nº Trades": total_trades
        }
    except: return None

print("\n" + "="*60)
print("⚡ JARVIS RECOVERY: TESTANDO MOMENTUM (BREAKOUT 10 DIAS)")
print("="*60)

resultados = []
for a in ATIVOS_RECOVERY:
    res = testar_momentum_breakout(a)
    if res: resultados.append(res)

df_final = pd.DataFrame(resultados).sort_values(by="Lucro Real %", ascending=False)
print(df_final.to_string(index=False))
print("="*60)