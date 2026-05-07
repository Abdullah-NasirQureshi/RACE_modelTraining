"""Single-command reproducible training pipeline (spec section 10.1).

Usage
-----
    python -m src.train_all \
        --train_path data/raw/train.csv \
        --val_path   data/raw/val.csv  \
        --test_path  data/raw/test.csv \
        --max_train_rows 20000        \
        --max_val_rows   5000         \
        [--use_tfidf]                 \
        [--demo]

Side effects
------------
Writes pickled artifacts to:
    models/model_a/traditional/...
    models/model_b/traditional/...
And summary metrics to:
    models/manifest.json   (run metadata + final metrics)
    models/metrics.json    (UI-consumed metrics; written by evaluate.py)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from typing import Optional

import joblib
import pandas as pd
import sklearn

from .demo_data import load_demo_dataframe
from .evaluate import evaluate
from .model_a_train import train_model_a
from .model_b_train import train_model_b
from .preprocessing import load_race_csv


MODELS_ROOT = "models"


def _maybe_load(path: Optional[str], max_rows: Optional[int]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return load_race_csv(path, max_rows=max_rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Train Model A + Model B end-to-end.")
    p.add_argument("--train_path", type=str, default="data/raw/train.csv")
    p.add_argument("--val_path", type=str, default="data/raw/val.csv")
    p.add_argument("--test_path", type=str, default=None,
                   help="Optional test CSV; if given, evaluate.py will run on it.")
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_val_rows", type=int, default=None)
    p.add_argument("--max_test_rows", type=int, default=None)
    p.add_argument("--n_clusters", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_tfidf", action="store_true",
                   help="Use TF-IDF instead of One-Hot Encoding (ablation).")
    p.add_argument("--demo", action="store_true",
                   help="Train on the synthetic demo dataset (smoke test).")
    args = p.parse_args()

    os.makedirs(MODELS_ROOT, exist_ok=True)
    t0 = time.time()

    if args.demo:
        print("[train_all] DEMO MODE: using synthetic dataset from src.demo_data")
        train_df = load_demo_dataframe()
        # Repeat to give the verifier some signal.
        train_df = pd.concat([train_df] * 4, ignore_index=True)
        val_df = load_demo_dataframe()
        test_df = load_demo_dataframe()
    else:
        if not os.path.exists(args.train_path):
            print(
                f"[train_all] train CSV not found at {args.train_path}. "
                "Tip: pass --demo to train on the synthetic dataset, or place the "
                "RACE Kaggle CSVs at data/raw/ and re-run.",
                file=sys.stderr,
            )
            return 1
        if not os.path.exists(args.val_path):
            print(
                f"[train_all] val CSV not found at {args.val_path}.",
                file=sys.stderr,
            )
            return 1

        train_df = load_race_csv(args.train_path, max_rows=args.max_train_rows, seed=args.seed)
        val_df = load_race_csv(args.val_path, max_rows=args.max_val_rows, seed=args.seed)
        test_df = (
            load_race_csv(args.test_path, max_rows=args.max_test_rows, seed=args.seed)
            if args.test_path else None
        )

    print(f"[train_all] train={len(train_df)}  val={len(val_df)}  "
          f"test={len(test_df) if test_df is not None else 'N/A'}")

    # ----------------------------- Model A --------------------------------
    metrics_a = train_model_a(
        train_df=train_df,
        val_df=val_df,
        use_tfidf=args.use_tfidf,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    # Re-use Model A's vectorizer for Model B if compatible (One-Hot only).
    aux_vec = None
    if not args.use_tfidf:
        try:
            aux_vec = joblib.load(
                os.path.join("models", "model_a", "traditional", "vectorizer.pkl")
            )
        except Exception:
            aux_vec = None

    # ----------------------------- Model B --------------------------------
    metrics_b = train_model_b(
        train_df=train_df,
        val_df=val_df,
        seed=args.seed,
        max_rows=args.max_train_rows,
        onehot_vectorizer=aux_vec,
    )

    # ----------------------------- Final eval -----------------------------
    print("[train_all] running unified evaluation ...")
    eval_df = test_df if test_df is not None else val_df
    report = evaluate(eval_df, max_rows=args.max_val_rows or args.max_test_rows)
    print(f"[train_all] eval done. accuracy={report.verifier_accuracy:.3f}")

    # ----------------------------- Manifest -------------------------------
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "feature_representation": "tfidf" if args.use_tfidf else "onehot",
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)) if test_df is not None else 0,
        "demo_mode": bool(args.demo),
        "seed": args.seed,
        "metrics_model_a": metrics_a.to_dict(),
        "metrics_model_b": metrics_b.to_dict(),
        "metrics_eval": report.to_dict(),
    }
    manifest_path = os.path.join(MODELS_ROOT, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[train_all] manifest written to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
