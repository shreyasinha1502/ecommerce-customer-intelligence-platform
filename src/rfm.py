"""RFM segmentation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RFM_OUTPUT_PATH = OUTPUTS_DIR / "rfm_segments.csv"


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
def calculate_rfm(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calculate recency, frequency, and monetary values per customer."""

    try:
        if df is None:
            df = load_and_clean_data()

        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = (
            df.groupby("CustomerID")
            .agg(
                Recency=("InvoiceDate", lambda x: int((snapshot_date - x.max()).days)),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("Revenue", "sum"),
                AvgOrderValue=("Revenue", "mean"),
                Country=("Country", lambda x: x.mode().iloc[0]),
                LastPurchaseDate=("InvoiceDate", "max"),
            )
            .reset_index()
        )
        rfm["Monetary"] = rfm["Monetary"].round(2)
        rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)
        return rfm
    except Exception as exc:
        LOGGER.exception("Failed to calculate RFM metrics.")
        raise RuntimeError("RFM calculation failed.") from exc


@cache_data(show_spinner=False)
def segment_customers(rfm_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cluster customers into 4 segments and save the output."""

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        if rfm_df is None:
            rfm_df = calculate_rfm()

        feature_df = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_df)

        model = KMeans(n_clusters=4, random_state=42, n_init=20)
        rfm_df = rfm_df.copy()
        rfm_df["Cluster"] = model.fit_predict(scaled)

        cluster_summary = (
            rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .assign(
                Score=lambda frame: frame["Monetary"].rank(pct=True)
                + frame["Frequency"].rank(pct=True)
                + (1 - frame["Recency"].rank(pct=True))
            )
            .sort_values("Score", ascending=False)
        )
        ordered_clusters = cluster_summary.index.tolist()
        segment_labels = {
            ordered_clusters[0]: "Champions",
            ordered_clusters[1]: "Loyal",
            ordered_clusters[2]: "At-Risk",
            ordered_clusters[3]: "Lost",
        }
        rfm_df["Segment"] = rfm_df["Cluster"].map(segment_labels)
        rfm_df = rfm_df.sort_values(["Segment", "Monetary"], ascending=[True, False]).reset_index(drop=True)
        rfm_df.to_csv(RFM_OUTPUT_PATH, index=False)
        LOGGER.info("RFM segments saved to %s", RFM_OUTPUT_PATH)
        return rfm_df
    except Exception as exc:
        LOGGER.exception("Failed to segment customers.")
        raise RuntimeError("Customer segmentation failed.") from exc


def ensure_rfm_output() -> pd.DataFrame:
    """Ensure the RFM output exists on disk."""

    try:
        if RFM_OUTPUT_PATH.exists():
            return pd.read_csv(RFM_OUTPUT_PATH, parse_dates=["LastPurchaseDate"])
        return segment_customers()
    except Exception as exc:
        LOGGER.exception("Failed to ensure RFM output.")
        raise RuntimeError("RFM output creation failed.") from exc


if not RFM_OUTPUT_PATH.exists():
    ensure_rfm_output()
