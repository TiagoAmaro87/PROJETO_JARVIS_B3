"""
Walk-Forward Backtest - proper temporal validation for B3.
No data leakage, realistic costs, position management.
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RISK_MANAGER.quant_framework import JarvisQuantFramework

# B3 costs
TAXA_B3 = 0.0005       # 0.05% corretagem + emolumentos
SLIPPAGE = 0.001        # 0.1% slippage estimado
STOP_LOSS_PCT = 0.02    # 2%
TAKE_PROFIT_PCT = 0.04  # 4% (2:1 R:R)
CAPITAL = 100000.0
RISK_PER_TRADE = 0.02   # 2%


def compute_features(df):
    c = df["Close"]
    v = df["Volume"]
    df["SMA_9"] = c.rolling(9).mean()
    df["SMA_20"] = c.rolling(20).mean()
    df["BB_Width"] = (c.rolling(20).std() * 4) / (df["SMA_20"] + 1e-10)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ROC"] = c.pct_change(5) * 100
    df["Vol_Shock"] = v / (v.rolling(20).mean() + 1)
    df["Target"] = (c.shift(-1) > c).astype(int)
    return df.dropna()


def walk_forward_backtest(ticker_data: pd.DataFrame, n_windows: int = 8):
    """Run walk-forward backtest with realistic execution."""
    n = len(ticker_data)
    window_size = n // n_windows
    train_size = int(window_size * 0.8)

    features = ["Close", "Volume", "SMA_9", "SMA_20", "BB_Width", "RSI", "ATR", "ROC", "Vol_Shock"]
    all_trades = []
    equity = CAPITAL
    equity_curve = [equity]

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        from sklearn.tree import DecisionTreeClassifier as GradientBoostingClassifier

    for w in range(n_windows):
        start = w * window_size
        train_end = start + train_size
        test_end = min(start + window_size, n)

        if train_end >= n or test_end > n:
            break

        train = ticker_data.iloc[start:train_end]
        test = ticker_data.iloc[train_end:test_end]

        if len(test) < 5:
            break

        X_train = train[features].values
        y_train = train["Target"].values
        X_test = test[features].values

        # Train simple model per window
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        # Simulate trades
        for i in range(len(test) - 1):
            prob = probs[i]
            price = test["Close"].iloc[i]
            next_price = test["Close"].iloc[i + 1]

            if prob > 0.55:  # Buy signal
                ret = (next_price - price) / price
                ret = max(ret, -STOP_LOSS_PCT)
                ret = min(ret, TAKE_PROFIT_PCT)
                cost = TAXA_B3 + SLIPPAGE
                net_ret = ret - cost

                trade_size = equity * RISK_PER_TRADE / STOP_LOSS_PCT
                pnl = trade_size * net_ret
                equity += pnl

                all_trades.append({
                    "window": w, "side": "buy", "prob": prob,
                    "return": ret, "net_return": net_ret, "pnl": pnl,
                })

            elif prob < 0.45:  # Sell signal
                ret = (price - next_price) / price
                ret = max(ret, -STOP_LOSS_PCT)
                ret = min(ret, TAKE_PROFIT_PCT)
                cost = TAXA_B3 + SLIPPAGE
                net_ret = ret - cost

                trade_size = equity * RISK_PER_TRADE / STOP_LOSS_PCT
                pnl = trade_size * net_ret
                equity += pnl

                all_trades.append({
                    "window": w, "side": "sell", "prob": prob,
                    "return": ret, "net_return": net_ret, "pnl": pnl,
                })

            equity_curve.append(equity)

    return all_trades, equity_curve


def run_full_backtest():
    print("=" * 60)
    print("  JARVIS_B3 - Walk-Forward Backtest")
    print("=" * 60)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(base, "DATA_INGESTION", "RAW_ASSETS", "*.csv"))

    quant = JarvisQuantFramework()
    all_results = {}

    for f in files:
        ticker = os.path.basename(f).replace("_raw.csv", "").replace(".SA", "")
        try:
            df = pd.read_csv(f, index_col=0, header=[0, 1])
            df.columns = df.columns.get_level_values(0)
            df = compute_features(df)

            if len(df) < 100:
                continue

            trades, equity_curve = walk_forward_backtest(df, n_windows=6)

            if not trades:
                continue

            returns = pd.Series([t["net_return"] for t in trades])
            wins = [t for t in trades if t["net_return"] > 0]
            losses = [t for t in trades if t["net_return"] <= 0]

            sharpe = quant.calculate_sharpe_ratio(returns)
            max_dd = quant.maximum_drawdown(returns)
            win_rate = len(wins) / len(trades) if trades else 0
            avg_win = np.mean([t["net_return"] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t["net_return"]) for t in losses]) if losses else 0
            pf = (sum(t["net_return"] for t in wins) / abs(sum(t["net_return"] for t in losses))) if losses else 0

            final_equity = equity_curve[-1]
            total_return = (final_equity - CAPITAL) / CAPITAL * 100

            print(f"\n  {ticker}:")
            print(f"    Trades: {len(trades)} | WR: {win_rate:.1%} | PF: {pf:.2f}")
            print(f"    Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1%}")
            print(f"    Return: {total_return:.1f}% | Final: R${final_equity:,.2f}")

            all_results[ticker] = {
                "trades": len(trades), "win_rate": win_rate, "sharpe": sharpe,
                "max_dd": max_dd, "return_pct": total_return, "profit_factor": pf,
            }

        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    if all_results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {len(all_results)} tickers backtested")
        avg_wr = np.mean([r["win_rate"] for r in all_results.values()])
        avg_sharpe = np.mean([r["sharpe"] for r in all_results.values()])
        print(f"  Avg Win Rate: {avg_wr:.1%}")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    run_full_backtest()
