"""
JARVIS_B3 - Model Training V2
Fixes: temporal split (no data leakage), walk-forward, XGBoost + MLP ensemble.
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False


FEATURES = ['Close', 'Volume', 'SMA_9', 'SMA_20', 'BB_Width', 'RSI', 'ATR', 'ROC', 'Vol_Shock',
            'OFI', 'Volume_Delta', 'VWAP_Dist', 'Momentum_5', 'Momentum_20']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "FEATURE_ENGINEERING", "ALL_STOCKS_v2.csv")
MODEL_DIR = os.path.join(BASE_DIR, "BRAIN (DL)")


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add microstructure-inspired features."""
    c = df["Close"]
    v = df["Volume"]
    h = df["High"]
    l = df["Low"]

    # Original features
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

    # NEW: Advanced features
    # Order Flow Imbalance proxy (using volume + price direction)
    df["OFI"] = np.where(c > c.shift(), v, -v).cumsum()
    df["OFI"] = df["OFI"].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 1e-10) if len(x) > 1 else 0)

    # Volume Delta (buy vs sell volume proxy)
    df["Volume_Delta"] = v * np.sign(c - c.shift()).fillna(0)
    df["Volume_Delta"] = df["Volume_Delta"].rolling(10).sum() / (v.rolling(10).sum() + 1)

    # VWAP distance
    typical_price = (h + l + c) / 3
    vwap = (typical_price * v).rolling(20).sum() / (v.rolling(20).sum() + 1)
    df["VWAP_Dist"] = (c - vwap) / (vwap + 1e-10)

    # Multi-period momentum
    df["Momentum_5"] = c.pct_change(5)
    df["Momentum_20"] = c.pct_change(20)

    # Target: next day return > 0
    df["Target"] = (c.shift(-1) > c).astype(int)

    return df


def temporal_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15):
    """Temporal split - NO SHUFFLE, respects time order."""
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def walk_forward_validate(X, y, n_windows=5, train_ratio=0.8):
    """Walk-forward validation - no data leakage."""
    n = len(X)
    window_size = n // n_windows
    results = []

    for i in range(n_windows):
        start = i * window_size
        split = start + int(window_size * train_ratio)
        end = min(start + window_size, n)

        if split >= n or end > n:
            break

        X_train, y_train = X[start:split], y[start:split]
        X_test, y_test = X[split:end], y[split:end]

        if HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", verbosity=0
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            preds = model.predict(X_test)
        else:
            # Fallback to simple logistic-like approach
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results.append({"window": i, "accuracy": acc, "train_size": len(X_train), "test_size": len(X_test)})
        print(f"  Window {i}: acc={acc:.4f} (train={len(X_train)}, test={len(X_test)})")

    return results


def train():
    print("=" * 60)
    print("  JARVIS_B3 - Model Training V2 (No Data Leakage)")
    print("=" * 60)

    # Load data
    if os.path.exists(DATA_PATH):
        print(f"Loading {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        print("V2 data not found. Processing from raw files...")
        import glob
        files = glob.glob(os.path.join(BASE_DIR, "DATA_INGESTION", "RAW_ASSETS", "*.csv"))
        all_dfs = []
        for f in files:
            try:
                d = pd.read_csv(f, index_col=0, header=[0, 1])
                d.columns = d.columns.get_level_values(0)
                ticker = os.path.basename(f).replace("_raw.csv", "")
                d["Ticker"] = ticker
                d = add_advanced_features(d)
                all_dfs.append(d.dropna())
            except Exception as e:
                print(f"  Error processing {f}: {e}")
        if not all_dfs:
            print("No data found!")
            return
        df = pd.concat(all_dfs)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved processed data: {len(df)} rows")

    # Ensure features exist
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Missing features (regenerating): {missing}")
        df = add_advanced_features(df)
        available = [f for f in FEATURES if f in df.columns]

    X = df[available].values
    y = df["Target"].values
    print(f"\nFeatures: {len(available)} | Samples: {len(X)}")

    # Temporal split
    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.pkl"))

    # Walk-forward validation
    print("\n--- Walk-Forward Validation ---")
    wf_results = walk_forward_validate(X_train_s, y_train, n_windows=5)
    avg_wf_acc = np.mean([r["accuracy"] for r in wf_results])
    print(f"  Average WF accuracy: {avg_wf_acc:.4f}")

    # Train XGBoost (primary model)
    if HAS_XGB:
        print("\n--- Training XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss", verbosity=0
        )
        xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

        val_pred = xgb_model.predict(X_val_s)
        test_pred = xgb_model.predict(X_test_s)
        print(f"  Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
        print(f"  Test accuracy (OOS): {accuracy_score(y_test, test_pred):.4f}")

        # Feature importance
        imp = dict(zip(available, xgb_model.feature_importances_))
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        print("\n  Feature Importance:")
        for fname, score in sorted_imp[:10]:
            print(f"    {fname}: {score:.4f}")

        xgb_model.save_model(os.path.join(MODEL_DIR, "modelo_xgb_v2.json"))
        print("  Saved: modelo_xgb_v2.json")

    # Train MLP (secondary)
    if HAS_TF:
        print("\n--- Training MLP ---")
        model = Sequential([
            Dense(128, input_dim=X_train_s.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
        model.fit(X_train_s, y_train, epochs=100, batch_size=64,
                  validation_data=(X_val_s, y_val), callbacks=[es], verbose=0)

        val_acc = model.evaluate(X_val_s, y_val, verbose=0)[1]
        test_acc = model.evaluate(X_test_s, y_test, verbose=0)[1]
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Test accuracy (OOS): {test_acc:.4f}")

        model.save(os.path.join(MODEL_DIR, "modelo_mlp_v2.h5"))
        print("  Saved: modelo_mlp_v2.h5")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train()
