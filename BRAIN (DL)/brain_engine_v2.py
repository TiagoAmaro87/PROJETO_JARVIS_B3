"""
JARVIS_B3 - Brain Engine V2
Uses MT5 API instead of PyAutoGUI. Supports XGBoost + MLP ensemble.
"""
import os
import time
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

import MetaTrader5 as mt5
sys.path.insert(0, r"E:\JARVIS_SYSTEM")
from core.nexus_unified import Nexus
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# Strat Mapping (EDGE Mode)
STRAT_MAP = {
    'VALE3': 'BREAKOUT', 'PETR4': 'BREAKOUT', 'ITUB4': 'TREND', 'BBDC4': 'TREND', 'BBAS3': 'BREAKOUT', 
    'ABEV3': 'BREAKOUT', 'B3SA3': 'BREAKOUT', 'ITSA4': 'TREND', 'SUZB3': 'BREAKOUT', 'RENT3': 'REVERSION', 
    'MGLU3': 'REVERSION', 'WEGE3': 'TREND', 'RADL3': 'TREND', 'RAIL3': 'REVERSION', 'GGBR4': 'REVERSION', 
    'EQTL3': 'REVERSION', 'LREN3': 'REVERSION', 'VIVT3': 'TREND', 'CSNA3': 'REVERSION', 'VBBR3': 'TREND', 
    'HYPE3': 'BREAKOUT', 'RDOR3': 'BREAKOUT', 'SBSP3': 'TREND', 'BRAP4': 'BREAKOUT', 'TOTS3': 'BREAKOUT', 
    'USIM5': 'REVERSION', 'GOAU4': 'REVERSION', 'ENGI11': 'REVERSION', 'BPAC11': 'REVERSION', 'BEEF3': 'TREND', 
    'CYRE3': 'REVERSION', 'MULT3': 'TREND',
    'WIN$': 'TREND', 'WDO$': 'REVERSION'
}

from BOT_EXECUTION.mt5_executor import B3Executor
from RISK_MANAGER.quant_framework import JarvisQuantFramework

try:
    from check_xp_config import MT5_LOGIN, MT5_PASS, MT5_SERVER
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "BRAIN (DL)")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v2.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "modelo_xgb_v2.json")
MLP_PATH = os.path.join(MODEL_DIR, "modelo_mlp_v2.h5")

WATCHLIST = list(STRAT_MAP.keys())

FEATURES_BASE = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock']
FEATURES_ADV = ['OFI', 'Volume_Delta', 'VWAP_Dist', 'Momentum_5', 'Momentum_20']

# Risk params
CAPITAL_INICIAL = 100000.0
RISCO_POR_TRADE = 0.02  # 2%
STOP_LOSS_PCT = 0.02    # 2%
MAX_POSICOES = 5
PROB_COMPRA = 0.60      # lowered from 0.85 - more realistic
PROB_VENDA = 0.40       # raised from 0.15


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for prediction."""
    c = df["Close"]
    v = df["Volume"]
    h = df.get("High", c)
    l = df.get("Low", c)

    df["SMA_9"] = c.rolling(9).mean()
    df["SMA_20"] = c.rolling(20).mean()
    df["BB_Width"] = (c.rolling(20).std() * 4) / (df["SMA_20"] + 1e-10)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["ATR"] = pd.concat([h - l, abs(h - c.shift()), abs(l - c.shift())], axis=1).max(axis=1).rolling(14).mean()
    df["ROC"] = c.pct_change(5) * 100
    df["Vol_Shock"] = v / (v.rolling(20).mean() + 1)

    # Advanced
    df["OFI"] = np.where(c > c.shift(), v, -v).cumsum()
    df["OFI"] = df["OFI"].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 1e-10) if len(x) > 1 else 0)
    df["Volume_Delta"] = v * np.sign(c - c.shift()).fillna(0)
    df["Volume_Delta"] = df["Volume_Delta"].rolling(10).sum() / (v.rolling(10).sum() + 1)
    typical_price = (h + l + c) / 3
    vwap = (typical_price * v).rolling(20).sum() / (v.rolling(20).sum() + 1)
    df["VWAP_Dist"] = (c - vwap) / (vwap + 1e-10)
    df["Momentum_5"] = c.pct_change(5)
    df["Momentum_20"] = c.pct_change(20)

    return df


class BrainEngineV2:
    """JARVIS_B3 Brain - ensemble prediction with MT5 execution."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.executor = B3Executor(dry_run=dry_run)
        self.risk = JarvisQuantFramework()
        self.scaler = None
        self.xgb_model = None
        self.mlp_model = None
        self.features = FEATURES_BASE
        self._trade_history = []
        self.network_log_path = os.path.join(BASE_DIR, "logs", "agent_network.json")
        
        # Ensure logs dir exists
        os.makedirs(os.path.dirname(self.network_log_path), exist_ok=True)

    def load_models(self) -> bool:
        """Load trained models."""
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            self.features = FEATURES_BASE + FEATURES_ADV
            logger.info("[BRAIN] Loaded V2 scaler (advanced features)")
        else:
            old_scaler = os.path.join(MODEL_DIR, "scaler_global.pkl")
            if os.path.exists(old_scaler):
                self.scaler = joblib.load(old_scaler)
                self.features = FEATURES_BASE
                logger.info("[BRAIN] Loaded V1 scaler (basic features)")
            else:
                logger.error("[BRAIN] No scaler found!")
                return False

        if HAS_XGB and os.path.exists(XGB_PATH):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(XGB_PATH)
            logger.info("[BRAIN] Loaded XGBoost model")

        if HAS_TF and os.path.exists(MLP_PATH):
            self.mlp_model = tf.keras.models.load_model(MLP_PATH)
            logger.info("[BRAIN] Loaded MLP model")

        if self.xgb_model is None and self.mlp_model is None:
            # Try legacy model
            legacy = os.path.join(MODEL_DIR, "modelo_global_b3.h5")
            if HAS_TF and os.path.exists(legacy):
                self.mlp_model = tf.keras.models.load_model(legacy)
                self.features = FEATURES_BASE
                logger.info("[BRAIN] Loaded legacy MLP model")
            else:
                logger.error("[BRAIN] No models found!")
                return False

        return True

    def analyze_market_sentiment(self, df: pd.DataFrame) -> dict:
        """Agentic Multi-Step Analysis (Inspired by AgentKit 2.1)."""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Price Action Filter (Trap Detection)
        closing_range = (last['High'] - last['Close']) / (last['High'] - last['Low'] + 1e-10)
        is_bull_trap = last['Close'] < prev['High'] and last['High'] > prev['High'] and closing_range > 0.6
        
        # 2. Volume Profile Check
        vol_avg = df['Volume'].tail(20).mean()
        vol_spike = last['Volume'] > vol_avg * 1.5
        
        # 3. Decision Logic
        sentiment = "neutral"
        if last['Close'] > last['SMA_9'] and not is_bull_trap:
            sentiment = "bullish"
        elif last['Close'] < last['SMA_9']:
            sentiment = "bearish"
        
        return {
            "sentiment": sentiment,
            "trap_detected": is_bull_trap,
            "high_volume": vol_spike,
            "range_score": 1.0 - closing_range
        }

    def self_reflect(self, ticker: str, prob: float, sentiment: dict) -> bool:
        """Self-Reflection Layer: Should I trust my model's probability?"""
        if sentiment['trap_detected']:
            logger.warning(f"[{ticker}] REFLEXÃO: Modelo deu prob={prob:.2f}, mas detectei Bull Trap! Operação ABORTADA.")
            return False
            
        if prob > PROB_COMPRA and sentiment['sentiment'] != 'bullish':
            logger.warning(f"[{ticker}] REFLEXÃO: Prob Alta ({prob:.2f}), mas tendência curta é Bearish. Aguardando confirmação.")
            return False
            
        return True

    def predict(self, df: pd.DataFrame) -> float:
        """Ensemble prediction. Returns probability of up move."""
        available = [f for f in self.features if f in df.columns]
        last = df[available].tail(1)
        if last.empty or last.isna().any(axis=1).iloc[0]:
            return 0.5

        X = self.scaler.transform(last.values)
        probs = []

        if self.xgb_model:
            p = self.xgb_model.predict_proba(X)[0][1]
            probs.append(p * 0.6)  # 60% weight to XGBoost

        if self.mlp_model:
            p = float(self.mlp_model.predict(X, verbose=0)[0][0])
            weight = 0.4 if self.xgb_model else 1.0
            probs.append(p * weight)

        return sum(probs) if probs else 0.5

    def get_strategy_signal(self, ticker: str, df: pd.DataFrame) -> str:
        """Strategy Multiplexer for the EDGE."""
        strat = STRAT_MAP.get(ticker, "TREND")
        last = df.iloc[-1]
        
        # 1. TREND RIDER (EMA 200 + EMA 20)
        if strat == "TREND":
            ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
            ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
            if last['Close'] > ema200 and last['Close'] > ema20: return "BUY"
            if last['Close'] < ema200 and last['Close'] < ema20: return "SELL"
            
        # 2. MEAN REVERSION (Bollinger + RSI)
        elif strat == "REVERSION":
            std = df['Close'].rolling(20).std().iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            if last['Close'] < ma20 - 2*std: return "BUY"
            if last['Close'] > ma20 + 2*std: return "SELL"
            
        # 3. BREAKOUT HUNTER (Volatility Pivot)
        elif strat == "BREAKOUT":
            high_range = df['High'].tail(10).max()
            low_range = df['Low'].tail(10).min()
            if last['Close'] > high_range: return "BUY"
            if last['Close'] < low_range: return "SELL"
            
        return "NEUTRAL"

    def ask_oracle(self, query: str):
        """Ask the Oracle Agent for intelligence via JARVIS_NETWORK."""
        logger.info(f"[NETWORK] Brain -> Oracle: {query}")
        # Log to the JSON hub
        msg = {"from": "Brain", "to": "Oracle", "query": query, "time": datetime.now().isoformat()}
        try:
            with open(self.network_log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        except: pass
        
        # Mirror to Obsidian (Scribe work)
        obsidian_path = r"C:\Users\tiago\Documents\Obsidian_Brain\00_Sistemas\Sala_de_Guerra_Agentes.md"
        if os.path.exists(obsidian_path):
            with open(obsidian_path, "a", encoding="utf-8") as f:
                f.write(f"| **BrainEngine** | **Oracle** | \"{query}\" | \"Oráculo analisando e varrendo fontes...\" |\n")

    def detect_smc_zones(self, df: pd.DataFrame) -> dict:
        """Detect Fair Value Gaps (FVG)."""
        if len(df) < 5: return {"bullish_fvg": False, "bearish_fvg": False}
        
        # Bullish FVG: Low(3) > High(1)
        bullish = df.iloc[-1]['Low'] > df.iloc[-3]['High']
        # Bearish FVG: High(3) < Low(1)
        bearish = df.iloc[-1]['High'] < df.iloc[-3]['Low']
        
        return {"bullish_fvg": bullish, "bearish_fvg": bearish}

    def _manage_trailing_stop(self):
        """Move stop to lock profit at 1%, 2.5% and 3.5% levels."""
        if self.dry_run: return
        
        positions = self.executor.get_posicoes_abertas()
        for pos in positions:
            ticker = pos['symbol']
            entry = pos['price_open']
            current = pos['price_current']
            ticket = pos['ticket']
            sl = pos['sl']
            
            # Profit %
            profit_pct = (current - entry) / entry if pos['type'] == "buy" else (entry - current) / entry
            
            new_sl = None
            if pos['type'] == "buy":
                # BUY Trail
                if profit_pct >= 0.035 and sl < entry * 1.02:
                    new_sl = entry * 1.02 # Lock 2%
                elif profit_pct >= 0.025 and sl < entry * 1.01:
                    new_sl = entry * 1.01 # Lock 1%
                elif profit_pct >= 0.010 and sl < entry:
                    new_sl = entry # Breakeven
            else:
                # SELL Trail
                if profit_pct >= 0.035 and sl > entry * 0.98:
                    new_sl = entry * 0.98 # Lock 2%
                elif profit_pct >= 0.025 and sl > entry * 0.99:
                    new_sl = entry * 0.99 # Lock 1%
                elif profit_pct >= 0.010 and sl > entry:
                    new_sl = entry # Breakeven
            
            if new_sl:
                self.executor.modificar_stop(ticket, new_sl)

    def _autonomous_scribe(self, summary: str):
        """Scribe Agent: Writes to Obsidian daily log and war room automatically."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = f"C:\\Users\\tiago\\Documents\\Obsidian_Brain\\04_Diário\\{today}.md"
        war_room_path = "C:\\Users\\tiago\\Documents\\Obsidian_Brain\\00_Sistemas\\Sala_de_Guerra_Agentes.md"
        
        # Append to Daily Log
        if os.path.exists(log_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n- **[AUTO-SCRIBE]** {datetime.now().strftime('%H:%M:%S')}: {summary}\n")
        
        # Mirror to War Room
        if os.path.exists(war_room_path):
            with open(war_room_path, "a", encoding="utf-8") as f:
                f.write(f"| **BrainEngine** | **Scribe** | \"Status: {summary}\" | \"Registrado no Diário de Bordo.\" |\n")

    def _autonomous_guardian(self):
        """Guardian Agent: Commits and pushes to GitHub every hour."""
        # Check if it's been 1 hour since last push
        # (Simplified: Runs every 60 scans if not on a timer)
        if not hasattr(self, '_scan_count'): self._scan_count = 0
        self._scan_count += 1
        
        if self._scan_count % 60 == 0:
            logger.info("[GUARDIAN] Hourly Cloud Backup initiating...")
            os.system(f"cd /d E:\\PROJETO_JARVIS_B3 && git add . && git commit -m \"AUTO-GUARDIAN: System Sync {today}\" && git push origin main")

    def scan_market(self):
        """Scan all watchlist tickers."""
        now = datetime.now()
        hora = now.hour

        # B3 market hours: 10:00 - 17:00 BRT
        if hora < 10 or hora >= 17:
            if now.minute == 0:
                logger.info(f"[BRAIN] Fora do horario B3 ({hora}h). Aguardando...")
            return

        self._manage_trailing_stop() # New: Manage open trades first

        posicoes = self.executor.get_posicoes_abertas() if not self.dry_run else []
        summary = f"Varredura Realizada. Posicoes: {len(posicoes)}/{MAX_POSICOES}"
        self._autonomous_scribe(summary)
        self._autonomous_guardian()

        logger.info(f"\n[BRAIN] Varredura {now.strftime('%H:%M:%S')} | "
                    f"Posicoes: {len(posicoes)}/{MAX_POSICOES}")

        for ticker in WATCHLIST:
            try:
                self._analyze_ticker(ticker)
            except Exception as e:
                logger.error(f"[{ticker}] Erro: {e}")

    def _analyze_ticker(self, ticker: str):
        """Analyze single ticker."""
        yf_ticker = f"{ticker}.SA"
        df = pd.DataFrame()

        # SPECIAL HANDLING: Mini Contracts (WIN$/WDO$)
        if "$" in ticker:
            if self.executor._connected:
                rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_M5, 0, 500)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            else:
                return # Can't trade mini without MT5
        else:
            if not HAS_YF: return
            df = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
            if df.empty: return
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if df.empty:
            return

        df = compute_features(df)
        df = df.dropna()
        if len(df) < 5:
            return

        prob = self.predict(df)
        preco = df["Close"].iloc[-1]
        # SMC Filter
        smc = self.detect_smc_zones(df)
        
        # EDGE: Multi-Strategy Signal
        strat_signal = self.get_strategy_signal(ticker, df)
        
        # NEW: Agentic Layer (Reflection)
        sentiment = self.analyze_market_sentiment(df)
        if not self.self_reflect(ticker, prob, sentiment):
            return

        # LOGIC: Hybrid (ML Prob + Strat Alignment + SMC)
        if prob > PROB_COMPRA and strat_signal == "BUY" and smc['bullish_fvg']:
            direction = "BUY"
        elif prob < PROB_VENDA and strat_signal == "SELL" and smc['bearish_fvg']:
            direction = "SELL"
        else:
            return # No institutional alignment today
        posicoes = self.executor.get_posicoes_abertas() if not self.dry_run else []
        if len(posicoes) >= MAX_POSICOES:
            return

        if prob > PROB_COMPRA:
            lote = self._calcular_lote(ticker)
            sl = preco * (1 - STOP_LOSS_PCT)
            tp = preco * (1 + STOP_LOSS_PCT * 2)  # 2:1 R:R

            logger.info(f"[BRAIN] {ticker} COMPRA | prob={prob:.4f} | preco={preco:.2f}")
            result = self.executor.comprar(ticker, lote, sl, tp)
            if "error" in result:
                logger.error(f"[BRAIN] {ticker} REJEITADA: {result['error']}")
                self.ask_oracle(f"Por que a ordem de {ticker} foi rejeitada? Verifique a corretora XP.")
            else:
                logger.success(f"[BRAIN] {ticker} ORDEM DISPARADA: {result['status']}")
                self.ask_oracle(f"Ordem de COMPRA disparada para {ticker}. Monitorar notícias do setor.")

            self._trade_history.append({
                "ticker": ticker, "side": "buy", "prob": prob,
                "price": preco, "time": datetime.now().isoformat(),
            })

        elif prob < PROB_VENDA:
            lote = self._calcular_lote(ticker)
            sl = preco * (1 + STOP_LOSS_PCT)
            tp = preco * (1 - STOP_LOSS_PCT * 2)

            logger.info(f"[BRAIN] {ticker} VENDA | prob={prob:.4f} | preco={preco:.2f}")
            result = self.executor.vender(ticker, lote, sl, tp)
            if "error" in result:
                logger.error(f"[BRAIN] {ticker} REJEITADA: {result['error']}")
            else:
                logger.success(f"[BRAIN] {ticker} ORDEM DISPARADA: {result['status']}")

            self._trade_history.append({
                "ticker": ticker, "side": "sell", "prob": prob,
                "price": preco, "time": datetime.now().isoformat(),
            })

    def _calcular_lote(self, ticker: str) -> float:
        """Calculate lot size. 1 for Mini, 100 for Stocks."""
        if "WIN" in ticker or "WDO" in ticker:
            return 1.0
        return 100.0

    def run(self):
        """Main loop."""
        logger.info("=" * 60)
        logger.info("  JARVIS_B3 - Brain Engine V2")
        logger.info(f"  Mode: {'DRY_RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"  Watchlist: {WATCHLIST}")
        logger.info("=" * 60)

        if not self.load_models():
            logger.error("Failed to load models. Exiting.")
            return

        # In dry_run: use yfinance for data (no MT5 needed)
        # In live: connect to MT5 (requires MT5 B3 terminal open and logged in)
        if not self.dry_run:
            conn_args = {"path": r"E:\MetaTrader5_B3\terminal64.exe"}
            if HAS_CONFIG:
                conn_args.update({
                    "login": MT5_LOGIN,
                    "password": MT5_PASS,
                    "server": MT5_SERVER
                })
            
            if not self.executor.connect(**conn_args):
                logger.error("MT5 B3 not connected. Cannot run live.")
                return
        logger.info(f"[BRAIN] Data source: {'MT5' if not self.dry_run and self.executor._connected else 'yfinance'}")

        try:
            while True:
                self.scan_market()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.executor.disconnect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", default=False)
    args = parser.parse_args()

    brain = BrainEngineV2(dry_run=not args.live)
    brain.run()
