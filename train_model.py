import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from supabase import create_client
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY env vars")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def fetch_calls(limit: int = 200000) -> pd.DataFrame:
    res = sb.table("calls").select("*").order("detected_time", desc=True).limit(limit).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # normalisera tider
    for c in ["call_time", "detected_time", "expiry_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

def main():
    df = fetch_calls()
    print(f"rows={len(df)}")

    if df.empty:
        print("No data. Exiting.")
        return

    # Vi tränar bara på stängda trades (TP/SL). EXPIRED kan du ta med senare.
    df = df[df["status"].isin(["TP", "SL"])].copy()
    print(f"closed(TP/SL)={len(df)}")

    if len(df) < 200:
        print("Not enough closed trades yet (<200). Exiting.")
        return

    # label: TP=1, SL=0
    df["y"] = (df["status"] == "TP").astype(int)

    # Features: håll det enkelt
    feature_cols_num = ["dump_pct", "vol_z", "vol_ratio", "liq_ratio"]
    feature_cols_cat = ["coin"]

    # säkerställ kolumner
    for c in feature_cols_num:
        if c not in df.columns:
            df[c] = np.nan
    for c in feature_cols_cat:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    # droppa NaNs i numeriska (du kan istället fylla med median)
    df = df.dropna(subset=feature_cols_num).copy()

    X = df[feature_cols_num + feature_cols_cat]
    y = df["y"].values

    # Pipeline: one-hot coin + logistic regression
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # hanterar obalans TP/SL
        solver="lbfgs",
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    # plocka ut learned params för att spara i DB
    ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = [f"coin={c}" for c in ohe.categories_[0].tolist()]
    feature_names = feature_cols_num + cat_feature_names

    # LogisticRegression coef_
    coef = pipe.named_steps["clf"].coef_[0].tolist()
    intercept = float(pipe.named_steps["clf"].intercept_[0])

    payload = {
        "version": "chance_model_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "features": feature_names,
        "intercept": intercept,
        "coef": coef,
        "base_rate": float(df["y"].mean()),
    }

    sb.table("model_params").upsert(
        {"id": "chance_model_v1", "payload": payload},
        on_conflict="id"
    ).execute()

    print("Saved model_params chance_model_v1")
    print(f"base_rate={payload['base_rate']:.3f}, n={payload['n_samples']}")

if __name__ == "__main__":
    main()
