import os
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_GRANGER = True
except Exception:
    HAS_GRANGER = False

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CURRENT_DATA = UPLOAD_DIR / "Agriculture_price_dataset.csv"

_DATASET_CACHE = {"path": None, "mtime": None, "df": None}


def resolve_dataset_path():
    if CURRENT_DATA.exists():
        return CURRENT_DATA
    candidates = [
        Path(r"D:\downloads\Agriculture_price_dataset.csv"),
        Path(__file__).resolve().parent.parent / "Agriculture_price_dataset.csv",
        Path(tempfile.gettempdir()) / "mandi_forecast_uploads" / "Agriculture_price_dataset.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def get_current_dataframe():
    path = resolve_dataset_path()
    if path is None or not path.exists():
        return None
    mtime = path.stat().st_mtime
    if (
        _DATASET_CACHE["df"] is not None
        and _DATASET_CACHE["path"] == str(path)
        and _DATASET_CACHE["mtime"] == mtime
    ):
        return _DATASET_CACHE["df"]
    df = load_dataset(path)
    _DATASET_CACHE["path"] = str(path)
    _DATASET_CACHE["mtime"] = mtime
    _DATASET_CACHE["df"] = df
    return df


def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_dataset(path):
    df = clean_columns(pd.read_csv(path))
    required = ["Price Date", "Market Name", "Commodity", "Modal_Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing) +
            ". Required columns are Price Date, Market Name, Commodity, Modal_Price."
        )

    # Match the notebook's date parsing and numeric conversion.
    df["Price Date"] = pd.to_datetime(
        df["Price Date"], format="%m/%d/%Y", errors="coerce"
    )
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
    df["Market Name"] = df["Market Name"].astype(str).str.strip()
    df["Commodity"] = df["Commodity"].astype(str).str.strip()
    df = df.dropna(subset=["Price Date", "Market Name", "Commodity", "Modal_Price"])
    return df


def get_metadata(df):
    commodities = sorted(df["Commodity"].dropna().unique().tolist())
    market_counts = (
        df.groupby("Commodity")["Market Name"]
        .nunique()
        .sort_values(ascending=False)
    )
    default_commodity = "Onion" if "Onion" in commodities else (commodities[0] if commodities else "")
    default_markets = sorted(
        df.loc[df["Commodity"].eq(default_commodity), "Market Name"].dropna().unique().tolist()
    ) if default_commodity else []
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "date_min": df["Price Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Price Date"].max().strftime("%Y-%m-%d"),
        "commodities": commodities,
        "markets_by_commodity": {
            str(k): int(v) for k, v in market_counts.items()
        },
        "markets": default_markets,
        "xgboost_available": HAS_XGB,
        "granger_available": HAS_GRANGER,
    }


def top(series, k=5, ascending=False):
    return list(
        series.dropna().sort_values(ascending=ascending).head(k).index
    )


def build_model_set(random_state=42):
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )
    else:
        models["HistGradientBoosting"] = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state
        )
    return models


def calc_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAPE": float(
            np.mean(np.abs((y - pred) / np.maximum(np.abs(y), 1e-8))) * 100
        ),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else None,
    }


def run_experiment(df, target_market, target_commodity, top_k=5, min_overlap=60):
    commodity_df = df[df["Commodity"].eq(target_commodity)].copy()
    if commodity_df.empty:
        raise ValueError(f"No rows found for commodity: {target_commodity}")
    if target_market not in set(commodity_df["Market Name"]):
        raise ValueError(
            f"{target_market!r} is not available for {target_commodity!r}."
        )

    daily = (
        commodity_df.groupby(["Price Date", "Market Name"], as_index=False)["Modal_Price"]
        .mean()
        .rename(columns={"Price Date": "date", "Market Name": "market", "Modal_Price": "price"})
    )
    observed = daily.pivot(index="date", columns="market", values="price").sort_index()
    calendar = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    wide = observed.reindex(calendar)
    wide.index.name = "date"
    wide_ffill = wide.ffill(limit=3)

    target_observed = wide[target_market].dropna()
    dates = target_observed.index.sort_values()
    n = len(dates)
    if n < 100:
        raise ValueError(
            f"Only {n} target observations found. The notebook needs enough history "
            "for lag features and chronological splits."
        )

    train_frac, valid_frac = 0.60, 0.20
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))
    train_dates = dates[:i_train]
    valid_dates = dates[i_train:i_valid]
    test_dates = dates[i_valid:]
    target_train = target_observed.reindex(train_dates)

    candidates = [
        market for market in wide_ffill.columns
        if market != target_market
        and wide_ffill[market].reindex(train_dates).notna().sum() >= min_overlap
    ]

    corr_scores = {}
    granger_p = {}

    for market in candidates:
        correlations = []
        for lag in [1, 2, 3]:
            external_lag = wide_ffill[market].shift(lag).reindex(train_dates)
            paired = pd.concat(
                [target_train.rename("target"), external_lag.rename("external")],
                axis=1
            ).dropna()
            if (
                len(paired) >= min_overlap
                and paired["target"].nunique() > 1
                and paired["external"].nunique() > 1
            ):
                corr = paired["target"].corr(paired["external"])
                if np.isfinite(corr):
                    correlations.append(abs(float(corr)))
        if correlations:
            corr_scores[market] = max(correlations)

        if HAS_GRANGER:
            paired = pd.concat(
                [
                    target_train.rename("target"),
                    wide_ffill[market].reindex(train_dates).rename("candidate")
                ],
                axis=1
            ).dropna()
            if (
                len(paired) >= min_overlap
                and paired["target"].nunique() > 1
                and paired["candidate"].nunique() > 1
            ):
                try:
                    results = grangercausalitytests(
                        paired[["target", "candidate"]],
                        maxlag=3,
                        verbose=False
                    )
                    p_values = [
                        results[lag][0]["ssr_ftest"][1]
                        for lag in range(1, 4)
                    ]
                    granger_p[market] = min(p_values)
                except Exception:
                    pass

    corr_scores = pd.Series(corr_scores, dtype=float).sort_values(ascending=False)
    granger_p = pd.Series(granger_p, dtype=float).sort_values(ascending=True)

    selection = {
        "target_only": [],
        "correlation": top(corr_scores, k=top_k, ascending=False),
        "granger": top(granger_p, k=top_k, ascending=True) if len(granger_p) else [],
        "all_markets": candidates
    }

    rank_parts = []
    if len(corr_scores):
        rank_parts.append(corr_scores.rank(pct=True).rename("corr"))
    if len(granger_p):
        rank_parts.append(
            (-np.log10(granger_p.clip(lower=1e-12))).rank(pct=True).rename("granger")
        )
    if rank_parts:
        hybrid = (
            pd.concat(rank_parts, axis=1)
            .mean(axis=1)
            .dropna()
            .sort_values(ascending=False)
        )
        selection["hybrid"] = top(hybrid, k=top_k, ascending=False)
    else:
        selection["hybrid"] = selection["correlation"]

    def build_data(selected_markets):
        target = wide[target_market].dropna().sort_index()
        z = pd.DataFrame(index=target.index)
        z["target"] = target

        for lag in [1, 2, 3, 7, 14, 30]:
            z[f"target_lag_{lag}"] = target.shift(lag)
        for win in [7, 14, 30]:
            z[f"target_roll_{win}"] = (
                target.shift(1).rolling(win, min_periods=win).mean()
            )
        z["month"] = z.index.month
        z["day_of_week"] = z.index.dayofweek
        z["day_of_year"] = z.index.dayofyear

        for market in selected_markets:
            ext = wide_ffill[market].reindex(z.index)
            name = str(market).replace(" ", "_").replace("/", "_")
            for lag in [1, 2, 3, 7]:
                z[f"{name}_lag_{lag}"] = ext.shift(lag)

        core = [
            c for c in z.columns
            if c.startswith("target_lag_") or c.startswith("target_roll_")
        ]
        return z.dropna(subset=core)

    supervised = {method: build_data(markets) for method, markets in selection.items()}

    def split_rows(data, split_dates):
        return data[data.index.isin(split_dates)]

    val_results = []
    for strategy, data in supervised.items():
        tr = split_rows(data, train_dates)
        va = split_rows(data, valid_dates)
        if len(tr) < 30 or len(va) < 10:
            continue
        Xtr, ytr = tr.drop(columns="target"), tr["target"]
        Xva, yva = va.drop(columns="target"), va["target"]
        for model_name, base_model in build_model_set().items():
            model = make_pipeline(SimpleImputer(strategy="median"), base_model)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            val_results.append({
                "strategy": strategy,
                "model": model_name,
                "external_markets": len(selection[strategy]),
                **calc_metrics(yva, pred)
            })

    if not val_results:
        raise ValueError("No validation results were produced; check data coverage.")
    validation_results = sorted(val_results, key=lambda x: x["RMSE"])
    best = validation_results[0]
    best_strategy, best_model = best["strategy"], best["model"]

    test_results = []
    test_predictions = {}
    for strategy, data in supervised.items():
        tr = data[data.index.isin(dates[:i_valid])]
        te = data[data.index.isin(test_dates)]
        if len(tr) < 30 or len(te) < 10:
            continue
        Xtr, ytr = tr.drop(columns="target"), tr["target"]
        Xte, yte = te.drop(columns="target"), te["target"]
        for model_name, base_model in build_model_set().items():
            model = make_pipeline(SimpleImputer(strategy="median"), base_model)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            metrics = calc_metrics(yte, pred)
            test_results.append({
                "strategy": strategy,
                "model": model_name,
                "external_markets": len(selection[strategy]),
                **metrics
            })
            if strategy == best_strategy and model_name == best_model:
                test_predictions = {
                    "dates": [d.strftime("%Y-%m-%d") for d in yte.index],
                    "actual": [float(v) for v in yte.values],
                    "predicted": [float(v) for v in pred]
                }

    test_results.sort(key=lambda x: x["RMSE"])

    # Price-regime summary for transparent reporting.
    split_stats = {}
    for name, split_dates in [
        ("train", train_dates), ("validation", valid_dates), ("test", test_dates)
    ]:
        vals = target_observed.reindex(split_dates).dropna()
        split_stats[name] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max())
        }

    top_corr = [
        {"market": str(k), "score": float(v)}
        for k, v in corr_scores.head(max(top_k, 10)).items()
    ]
    top_granger = [
        {"market": str(k), "p_value": float(v)}
        for k, v in granger_p.head(max(top_k, 10)).items()
    ]

    return {
        "config": {
            "target_market": target_market,
            "commodity": target_commodity,
            "top_k": top_k,
            "min_overlap": min_overlap,
            "target_observations": int(n),
            "train_observations": int(len(train_dates)),
            "validation_observations": int(len(valid_dates)),
            "test_observations": int(len(test_dates)),
            "eligible_markets": int(len(candidates)),
            "train_rows_after_features": int(len(split_rows(supervised["target_only"], train_dates))),
            "validation_rows_after_features": int(len(split_rows(supervised["target_only"], valid_dates))),
            "test_rows_after_features": int(len(split_rows(supervised["target_only"], test_dates))),
            "geographic_selection": "Not enabled (no coordinates file supplied)"
        },
        "selection": {
            key: [{"market": m} for m in value]
            for key, value in selection.items()
        },
        "top_correlation": top_corr,
        "top_granger": top_granger,
        "validation_results": validation_results,
        "test_results": test_results,
        "selected_on_validation": {
            "strategy": best_strategy,
            "model": best_model,
            "validation_rmse": float(best["RMSE"])
        },
        "selected_model_test_predictions": test_predictions,
        "price_distribution": split_stats,
        "notes": {
            "xgboost_available": HAS_XGB,
            "granger_available": HAS_GRANGER,
            "external_forward_fill_days": 3,
            "selection_uses_training_period_only": True
        }
    }


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "xgboost_available": HAS_XGB, "granger_available": HAS_GRANGER})


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Please attach the Agriculture_price_dataset.csv file."}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400
    file.save(CURRENT_DATA)
    _DATASET_CACHE["df"] = None
    try:
        df = load_dataset(CURRENT_DATA)
        _DATASET_CACHE["path"] = str(CURRENT_DATA)
        _DATASET_CACHE["mtime"] = CURRENT_DATA.stat().st_mtime
        _DATASET_CACHE["df"] = df
        return jsonify(get_metadata(df))
    except Exception as exc:
        CURRENT_DATA.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400


@app.get("/api/metadata")
def metadata():
    df = get_current_dataframe()
    if df is None:
        return jsonify({"error": "Upload the CSV dataset first."}), 404
    try:
        return jsonify(get_metadata(df))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/markets")
def markets():
    df = get_current_dataframe()
    if df is None:
        return jsonify({"error": "Upload the CSV dataset first."}), 404
    commodity = str(request.args.get("commodity", "Onion")).strip()
    try:
        names = sorted(
            df.loc[df["Commodity"].eq(commodity), "Market Name"].dropna().unique().tolist()
        )
        return jsonify({"commodity": commodity, "markets": names})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/run")
def run():
    df = get_current_dataframe()
    if df is None:
        return jsonify({"error": "Upload the CSV dataset first."}), 400
    payload = request.get_json(silent=True) or {}
    target_market = str(payload.get("target_market", "Pratapgarh")).strip()
    target_commodity = str(payload.get("commodity", "Onion")).strip()
    try:
        top_k = max(1, min(int(payload.get("top_k", 5)), 20))
        min_overlap = max(20, int(payload.get("min_overlap", 60)))
        result = run_experiment(
            df, target_market, target_commodity, top_k=top_k, min_overlap=min_overlap
        )
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
