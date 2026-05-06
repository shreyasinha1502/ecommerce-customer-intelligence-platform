"""SHAP explainability helpers for the churn model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap

from src.churn import engineer_features, load_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


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
def get_shap_values() -> dict:
    """Compute SHAP values using TreeExplainer for the saved churn model."""

    try:
        bundle = load_model_bundle()
        features = engineer_features()
        model = bundle["model"]

        modeling_frame = pd.get_dummies(features[["Country", "Segment"]], prefix=["Country", "Segment"])
        numeric_frame = features[
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
        ]
        X = pd.concat([numeric_frame, modeling_frame], axis=1)
        X = X.reindex(columns=bundle["feature_names"], fill_value=0.0)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        return {
            "values": shap_values.values,
            "base_values": shap_values.base_values,
            "X": X,
            "customer_ids": features["CustomerID"].tolist(),
        }
    except Exception as exc:
        LOGGER.exception("Failed to compute SHAP values.")
        raise RuntimeError("SHAP value generation failed.") from exc


def plot_global_importance():
    """Return a Plotly chart for global feature importance."""

    try:
        shap_bundle = get_shap_values()
        shap_values = shap_bundle["values"]
        X = shap_bundle["X"]
        importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "MeanAbsSHAP": abs(shap_values).mean(axis=0),
            }
        ).sort_values("MeanAbsSHAP", ascending=False).head(12)
        fig = px.bar(
            importance,
            x="MeanAbsSHAP",
            y="Feature",
            orientation="h",
            title="Global Feature Importance",
            color="MeanAbsSHAP",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig
    except Exception as exc:
        LOGGER.exception("Failed to build global SHAP importance plot.")
        raise RuntimeError("Global SHAP plotting failed.") from exc


def plot_individual_explanation(customer_id: int):
    """Return a matplotlib waterfall figure for a specific customer."""

    try:
        shap_bundle = get_shap_values()
        customer_ids = shap_bundle["customer_ids"]
        if customer_id not in customer_ids:
            raise ValueError(f"CustomerID {customer_id} was not found.")

        index = customer_ids.index(customer_id)
        plt.close("all")
        explanation = shap.Explanation(
            values=shap_bundle["values"][index],
            base_values=shap_bundle["base_values"][index],
            data=shap_bundle["X"].iloc[index].values,
            feature_names=shap_bundle["X"].columns.tolist(),
        )
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        return fig
    except Exception as exc:
        LOGGER.exception("Failed to build individual SHAP explanation.")
        raise RuntimeError("Individual SHAP plotting failed.") from exc
