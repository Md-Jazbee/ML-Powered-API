#!/usr/bin/env python3
"""
Synthetic Data Generator
------------------------
Generate synthetic datasets tailored to the algorithm you want to learn/test.

Supported algorithm names (aliases included):
  Regression:
    - linear_regression, ridge, lasso, elasticnet
    - random_forest_regressor, rf_regressor, xgboost_regressor (pattern only)
    - svr_regressor, knn_regressor
  Classification:
    - logistic_regression, logreg
    - random_forest_classifier, rf_classifier
    - svm_classifier, svc, knn_classifier, naive_bayes
    - xgboost_classifier (pattern only)
  Clustering:
    - kmeans, dbscan, gmm
  Time series (forecasting):
    - arima_like, prophet_like, time_series, forecasting, sales_forecast

Output:
  - CSV by default, or Parquet if --format parquet (requires pyarrow or fastparquet)

Examples:
  python synthetic_data_generator.py --algo linear_regression --n-samples 500 --out data/linreg.csv
  python synthetic_data_generator.py --algo random_forest_classifier --n-samples 2000 --classes 3 --out data/rf_cls.csv
  python synthetic_data_generator.py --algo kmeans --n-samples 1500 --clusters 4 --out data/kmeans.csv
  python synthetic_data_generator.py --algo time_series --days 365 --stores 10 --out data/ts_sales.csv
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Optional sklearn imports for tabular generators
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons

ALG_ALIASES: Dict[str, str] = {
    # Regression
    "linear_regression": "regression",
    "ridge": "regression",
    "lasso": "regression",
    "elasticnet": "regression",
    "random_forest_regressor": "regression",
    "rf_regressor": "regression",
    "xgboost_regressor": "regression",  # pattern only
    "svr_regressor": "regression",
    "knn_regressor": "regression",
    # Classification
    "logistic_regression": "classification",
    "logreg": "classification",
    "random_forest_classifier": "classification",
    "rf_classifier": "classification",
    "svm_classifier": "classification",
    "svc": "classification",
    "knn_classifier": "classification",
    "naive_bayes": "classification",
    "xgboost_classifier": "classification",  # pattern only
    # Clustering
    "kmeans": "clustering",
    "dbscan": "clustering",
    "gmm": "clustering",
    # Time Series
    "arima_like": "timeseries",
    "prophet_like": "timeseries",
    "time_series": "timeseries",
    "forecasting": "timeseries",
    "sales_forecast": "timeseries",
}


def infer_problem(algo: str) -> str:
    key = algo.strip().lower()
    if key in ALG_ALIASES:
        return ALG_ALIASES[key]
    # Fallback heuristics
    if "regress" in key:
        return "regression"
    if any(k in key for k in ["classif", "logistic", "svm", "svc", "nb", "naive_bayes"]):
        return "classification"
    if any(k in key for k in ["kmeans", "cluster", "dbscan", "gmm"]):
        return "clustering"
    if any(k in key for k in ["time", "forecast", "arima", "prophet"]):
        return "timeseries"
    raise ValueError(f"Unknown algorithm name: {algo}")


def gen_regression(
    n_samples: int,
    n_features: int,
    noise: float,
    random_state: int,
) -> pd.DataFrame:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features, n_features - 1)),
        noise=noise,
        random_state=random_state,
        bias=5.0,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


def gen_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    class_sep: float,
    weights: Optional[str],
    random_state: int,
) -> pd.DataFrame:
    if n_classes == 2:
        weights_list = None
        if weights:
            parts = [float(p) for p in weights.split(",")]
            if abs(sum(parts) - 1.0) > 1e-6:
                raise ValueError("--weights must sum to 1.0")
            weights_list = parts
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, min(n_features, 10)),
            n_redundant=max(0, min(2, n_features - 2)),
            n_repeated=0,
            n_classes=2,
            weights=weights_list,
            class_sep=class_sep,
            flip_y=0.01,
            random_state=random_state,
        )
    else:
        # Multi-class via blobs + a tiny nonlinearity
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            n_features=n_features,
            cluster_std=1.5 / max(1.0, class_sep),
            random_state=random_state,
        )
        # Add moons pattern to two features if possible
        if n_features >= 2:
            Xm, ym = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
            X[:, 0] = (X[:, 0] * 0.7 + Xm[:, 0] * 0.3)
            X[:, 1] = (X[:, 1] * 0.7 + Xm[:, 1] * 0.3)

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(int)
    return df


def gen_clustering(
    n_samples: int,
    n_features: int,
    clusters: int,
    cluster_std: float,
    random_state: int,
) -> pd.DataFrame:
    X, y = make_blobs(
        n_samples=n_samples,
        centers=clusters,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["cluster_label"] = y.astype(int)  # ground truth provided for evaluation
    return df


def gen_time_series(
    days: int,
    stores: int,
    items: int,
    promo_prob: float,
    random_state: int,
    price_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Generates a realistic retail-like panel time series:
      columns: date, store_id, item_id, promo_flag, price, sales
    Sales = base + weekly seasonality + yearly seasonality + promo lift - price elasticity + noise
    """
    rng = np.random.default_rng(random_state)
    date_index = pd.date_range("2018-01-01", periods=days, freq="D")

    rows = []
    for store in range(1, stores + 1):
        for item in range(1, items + 1):
            base = rng.uniform(10, 80)
            weekly = 8 * np.sin(2 * np.pi * (date_index.dayofweek) / 7.0)
            yearly = 10 * np.sin(2 * np.pi * (date_index.dayofyear) / 365.25)
            promo = (rng.random(len(date_index)) < promo_prob).astype(int)
            price = rng.uniform(price_range[0], price_range[1]) + rng.normal(0, 0.15, len(date_index))
            noise = rng.normal(0, 4.0, len(date_index))
            elasticity = rng.uniform(0.3, 0.9)
            promo_lift = rng.uniform(6, 15)

            sales = base + weekly + yearly + promo_lift * promo - elasticity * price + noise
            sales = np.clip(np.round(sales), a_min=0, a_max=None)

            df = pd.DataFrame({
                "date": date_index,
                "store_id": store,
                "item_id": item,
                "promo_flag": promo,
                "price": price.round(2),
                "sales": sales.astype(int),
            })
            rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    return out


def save_frame(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "parquet":
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            raise RuntimeError(
                "Writing Parquet failed. Install pyarrow or fastparquet, e.g., "
                "`pip install pyarrow`."
            ) from e
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main():
    p = argparse.ArgumentParser(description="Generate synthetic datasets by algorithm name.")
    p.add_argument("--algo", required=True, help="Algorithm name (e.g., linear_regression, random_forest_classifier, kmeans, time_series)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")
    p.add_argument("--out", required=True, help="Output path (e.g., data/train.csv)")

    # Shared tabular args
    p.add_argument("--n-samples", type=int, default=1000, help="Samples for tabular tasks")
    p.add_argument("--n-features", type=int, default=10, help="Features for tabular tasks")

    # Regression
    p.add_argument("--noise", type=float, default=5.0, help="Noise level for regression")

    # Classification
    p.add_argument("--classes", type=int, default=2, help="Number of classes")
    p.add_argument("--class-sep", type=float, default=1.0, help="Class separation")
    p.add_argument("--weights", type=str, default=None, help="Comma-separated class weights summing to 1.0 (binary only)")

    # Clustering
    p.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    p.add_argument("--cluster-std", type=float, default=1.0, help="Cluster std deviation")

    # Time series
    p.add_argument("--days", type=int, default=365, help="Days to simulate for time series")
    p.add_argument("--stores", type=int, default=5, help="Number of stores")
    p.add_argument("--items", type=int, default=1, help="Items per store")
    p.add_argument("--promo-prob", type=float, default=0.2, help="Daily promo probability")
    p.add_argument("--price-min", type=float, default=2.5, help="Min price")
    p.add_argument("--price-max", type=float, default=7.5, help="Max price")

    args = p.parse_args()

    try:
        problem = infer_problem(args.algo)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        print("Supported names include:", ", ".join(sorted(ALG_ALIASES.keys())), file=sys.stderr)
        sys.exit(2)

    rng = np.random.default_rng(args.seed)

    if problem == "regression":
        df = gen_regression(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            random_state=args.seed,
        )

    elif problem == "classification":
        # Ensure sane class count
        n_classes = max(2, int(args.classes))
        if n_classes > 20:
            raise ValueError("--classes too large; keep it <= 20 for practicality.")
        df = gen_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_classes=n_classes,
            class_sep=args.class_sep,
            weights=args.weights,
            random_state=args.seed,
        )

    elif problem == "clustering":
        df = gen_clustering(
            n_samples=args.n_samples,
            n_features=args.n_features,
            clusters=max(2, int(args.clusters)),
            cluster_std=float(args.cluster_std),
            random_state=args.seed,
        )

    elif problem == "timeseries":
        if args.price_min >= args.price_max:
            raise ValueError("--price-min must be < --price-max")
        df = gen_time_series(
            days=args.days,
            stores=max(1, int(args.stores)),
            items=max(1, int(args.items)),
            promo_prob=float(args.promo_prob),
            random_state=args.seed,
            price_range=(args.price_min, args.price_max),
        )

    else:
        raise AssertionError("Unexpected problem type")

    save_frame(df, args.out, args.format)
    print(f"Wrote {args.out} with shape {df.shape} and columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
