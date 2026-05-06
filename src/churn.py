"""Churn modeling utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_loader import load_and_clean_data
from src.rfm import ensure_rfm_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
PREDICTIONS_PATH = OUTPUTS_DIR / "churn_predictions.csv"


def _cache_data_fallback(*_args, **_kwargs) -> Callable:
    """Return a no-op decorator when Streamlit is unavailable."""

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    cache_data = _cache_data_fallback


@cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create customer-level churn features and a synthetic churn target."""

    try:
        if df is None:
            df = load_and_clean_data()

        rfm_df = ensure_rfm_output()
        analysis_date = df["InvoiceDate"].max()
        recent_90_cutoff = analysis_date - pd.Timedelta(days=90)
        recent_180_cutoff = analysis_date - pd.Timedelta(days=180)

        invoice_level = (
            df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate", "Country"], as_index=False)
            .agg(OrderRevenue=("Revenue", "sum"), Items=("Quantity", "sum"))
        )
        invoice_level["IsRecent90D"] = (invoice_level["InvoiceDate"] >= recent_90_cutoff).astype(int)
        invoice_level["IsRecent180D"] = (invoice_level["InvoiceDate"] >= recent_180_cutoff).astype(int)
        invoice_level["Revenue90DComponent"] = invoice_level["OrderRevenue"] * invoice_level["IsRecent90D"]
        invoice_level["Revenue180DComponent"] = invoice_level["OrderRevenue"] * invoice_level["IsRecent180D"]

        customer_features = (
            invoice_level.groupby("CustomerID")
            .agg(
                TotalOrders=("InvoiceNo", "nunique"),
                TotalRevenue=("OrderRevenue", "sum"),
                AvgOrderValue=("OrderRevenue", "mean"),
                AvgItemsPerOrder=("Items", "mean"),
                Revenue90D=("Revenue90DComponent", "sum"),
                Revenue180D=("Revenue180DComponent", "sum"),
                Orders90D=("IsRecent90D", "sum"),
                Orders180D=("IsRecent180D", "sum"),
                FirstPurchaseDate=("InvoiceDate", "min"),
                LastPurchaseDate=("InvoiceDate", "max"),
                Country=("Country", lambda x: x.mode().iloc[0]),
            )
            .reset_index()
        )

        customer_features["CustomerLifetimeDays"] = (
            customer_features["LastPurchaseDate"] - customer_features["FirstPurchaseDate"]
        ).dt.days.clip(lower=1)
        customer_features["PurchaseVelocity"] = (
            customer_features["TotalOrders"] / customer_features["CustomerLifetimeDays"]
        ).round(4)

        product_diversity = (
            df.groupby("CustomerID")
            .agg(
                DistinctProducts=("Description", "nunique"),
                AvgUnitPrice=("UnitPrice", "mean"),
            )
            .reset_index()
        )

        features = customer_features.merge(
            rfm_df[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]],
            on="CustomerID",
            how="left",
        ).merge(product_diversity, on="CustomerID", how="left")

        features["RevenueMomentum"] = (features["Revenue90D"] / features["Revenue180D"].replace(0, np.nan)).fillna(0)
        features["MonetaryPerOrder"] = (features["Monetary"] / features["Frequency"].replace(0, np.nan)).fillna(0)
        features["DaysSinceLastPurchase"] = features["Recency"]

        risk_score = (
            0.04 * features["Recency"]
            - 0.12 * features["Orders90D"]
            - 0.015 * features["Revenue90D"]
            - 0.20 * features["PurchaseVelocity"]
            + 0.10 * (features["Segment"].isin(["At-Risk", "Lost"]).astype(int))
        )
        churn_probability = 1 / (1 + np.exp(-(risk_score - risk_score.median()) / (risk_score.std() + 1e-6)))
        features["Churned"] = (churn_probability > np.quantile(churn_probability, 0.70)).astype(int)
        features["SyntheticChurnProbability"] = churn_probability.round(4)

        features["FirstPurchaseDate"] = pd.to_datetime(features["FirstPurchaseDate"])
        features["LastPurchaseDate"] = pd.to_datetime(features["LastPurchaseDate"])
        return features
    except Exception as exc:
        LOGGER.exception("Failed to engineer churn features.")
        raise RuntimeError("Feature engineering failed.") from exc


def _prepare_training_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare the feature matrix for XGBoost training."""

    try:
        model_df = features.copy()
        country_dummies = pd.get_dummies(model_df["Country"], prefix="Country")
        segment_dummies = pd.get_dummies(model_df["Segment"], prefix="Segment")
        X = pd.concat(
            [
                model_df[
                    [
                        "Recency",
                        "Frequency",
                        "Monetary",
                        "AvgOrderValue",
                        "AvgItemsPerOrder",
                        "Revenue90D",
                        "Revenue180D",
                        "Orders90D",
                        "Orders180D",
                        "CustomerLifetimeDays",
                        "PurchaseVelocity",
                        "DistinctProducts",
                        "AvgUnitPrice",
                        "RevenueMomentum",
                        "MonetaryPerOrder",
                        "DaysSinceLastPurchase",
                    ]
                ],
                country_dummies,
                segment_dummies,
            ],
            axis=1,
        ).astype(float)
        y = model_df["Churned"].astype(int)
        return X, y, country_dummies.columns.tolist(), segment_dummies.columns.tolist()
    except Exception as exc:
        LOGGER.exception("Failed to prepare training matrix.")
        raise RuntimeError("Training matrix preparation failed.") from exc


def train_model() -> dict:
    """Train the XGBoost churn model, save artifacts, and write predictions."""

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        features = engineer_features()
        X, y, country_columns, segment_columns = _prepare_training_matrix(features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        pos = max(int(y_train.sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        scale_pos_weight = neg / pos

        model = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)

        test_prob = model.predict_proba(X_test)[:, 1]
        LOGGER.info(
            "Churn model ROC-AUC: %.4f | Report: %s",
            roc_auc_score(y_test, test_prob),
            classification_report(y_test, (test_prob >= 0.5).astype(int), zero_division=0),
        )

        full_prob = model.predict_proba(X)[:, 1]
        predictions = features[
            ["CustomerID", "Country", "Segment", "Recency", "Frequency", "Monetary", "LastPurchaseDate"]
        ].copy()
        predictions["ChurnProbability"] = np.round(full_prob, 4)
        predictions["RiskLabel"] = np.where(predictions["ChurnProbability"] >= 0.5, "High Risk", "Low Risk")
        predictions = predictions.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)
        predictions.to_csv(PREDICTIONS_PATH, index=False)

        bundle = {
            "model": model,
            "feature_names": X.columns.tolist(),
            "country_columns": country_columns,
            "segment_columns": segment_columns,
            "default_country": "United Kingdom",
            "default_segment": "Loyal",
            "training_columns": X.columns.tolist(),
        }
        joblib.dump(bundle, MODEL_PATH)
        LOGGER.info("Churn model saved to %s", MODEL_PATH)
        return bundle
    except Exception as exc:
        LOGGER.exception("Failed to train churn model.")
        raise RuntimeError("Churn model training failed.") from exc


def load_model_bundle() -> dict:
    """Load the saved churn model bundle, training it first if needed."""

    try:
        if not MODEL_PATH.exists():
            return train_model()
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        LOGGER.exception("Failed to load model bundle.")
        raise RuntimeError("Model loading failed.") from exc


@cache_data(show_spinner=False)
def score_customers() -> pd.DataFrame:
    """Return the latest churn predictions for all customers."""

    try:
        if not PREDICTIONS_PATH.exists():
            train_model()
        return pd.read_csv(PREDICTIONS_PATH, parse_dates=["LastPurchaseDate"])
    except Exception as exc:
        LOGGER.exception("Failed to score customers.")
        raise RuntimeError("Customer scoring failed.") from exc


def build_manual_feature_row(inputs: dict) -> pd.DataFrame:
    """Build a single-row feature frame for manual churn prediction."""

    try:
        bundle = load_model_bundle()
        feature_names = bundle["feature_names"]
        row = {column: 0.0 for column in feature_names}

        numeric_mapping = {
            "Recency": inputs["recency"],
            "Frequency": inputs["frequency"],
            "Monetary": inputs["monetary"],
            "AvgOrderValue": inputs["avg_order_value"],
            "AvgItemsPerOrder": inputs["avg_items_per_order"],
            "Revenue90D": inputs["revenue_90d"],
            "Revenue180D": inputs["revenue_180d"],
            "Orders90D": inputs["orders_90d"],
            "Orders180D": inputs["orders_180d"],
            "CustomerLifetimeDays": inputs["customer_lifetime_days"],
            "PurchaseVelocity": inputs["frequency"] / max(inputs["customer_lifetime_days"], 1),
            "DistinctProducts": inputs["distinct_products"],
            "AvgUnitPrice": inputs["avg_unit_price"],
            "RevenueMomentum": inputs["revenue_90d"] / max(inputs["revenue_180d"], 1),
            "MonetaryPerOrder": inputs["monetary"] / max(inputs["frequency"], 1),
            "DaysSinceLastPurchase": inputs["recency"],
        }
        row.update(numeric_mapping)

        country_column = f"Country_{inputs['country']}"
        segment_column = f"Segment_{inputs['segment']}"
        if country_column in row:
            row[country_column] = 1.0
        if segment_column in row:
            row[segment_column] = 1.0

        return pd.DataFrame([row], columns=feature_names)
    except Exception as exc:
        LOGGER.exception("Failed to build manual feature row.")
        raise RuntimeError("Manual feature row construction failed.") from exc


if not MODEL_PATH.exists():
    train_model()
