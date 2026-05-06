"""Streamlit dashboard for the E-Commerce Customer Intelligence Platform."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.churn import build_manual_feature_row, engineer_features, load_model_bundle, score_customers
from src.data_loader import load_and_clean_data
from src.rfm import ensure_rfm_output
from src.shap_explain import get_shap_values, plot_global_importance, plot_individual_explanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="E-Commerce Customer Intelligence Platform",
    layout="wide",
    page_icon="🛍️",
)


@st.cache_data(show_spinner=False)
def load_platform_data() -> dict:
    """Load all core datasets used by the dashboard."""

    try:
        transactions = load_and_clean_data()
        rfm = ensure_rfm_output()
        churn_predictions = score_customers()
        churn_features = engineer_features()
        return {
            "transactions": transactions,
            "rfm": rfm,
            "churn_predictions": churn_predictions,
            "churn_features": churn_features,
        }
    except Exception as exc:
        LOGGER.exception("Failed to load platform data.")
        raise RuntimeError("Platform data loading failed.") from exc


@st.cache_data(show_spinner=False)
def filter_transactions(
    transactions: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp],
    countries: list[str],
) -> pd.DataFrame:
    """Filter transaction data using sidebar controls."""

    try:
        filtered = transactions.copy()
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[
            (filtered["InvoiceDate"] >= start_date)
            & (filtered["InvoiceDate"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        ]
        if countries:
            filtered = filtered[filtered["Country"].isin(countries)]
        return filtered
    except Exception as exc:
        LOGGER.exception("Failed to filter transactions.")
        raise RuntimeError("Transaction filtering failed.") from exc


@st.cache_data(show_spinner=False)
def filter_customer_views(
    rfm: pd.DataFrame,
    churn_predictions: pd.DataFrame,
    selected_countries: list[str],
    selected_segments: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter customer-level datasets using sidebar controls."""

    try:
        filtered_rfm = rfm.copy()
        filtered_churn = churn_predictions.copy()
        if selected_countries:
            filtered_rfm = filtered_rfm[filtered_rfm["Country"].isin(selected_countries)]
            filtered_churn = filtered_churn[filtered_churn["Country"].isin(selected_countries)]
        if selected_segments:
            filtered_rfm = filtered_rfm[filtered_rfm["Segment"].isin(selected_segments)]
            filtered_churn = filtered_churn[filtered_churn["Segment"].isin(selected_segments)]
        return filtered_rfm, filtered_churn
    except Exception as exc:
        LOGGER.exception("Failed to filter customer views.")
        raise RuntimeError("Customer view filtering failed.") from exc


def render_segmentation_page(filtered_rfm: pd.DataFrame) -> None:
    """Render the customer segmentation page."""

    try:
        st.subheader("Customer Segmentation")
        if filtered_rfm.empty:
            st.warning("No customers match the current filters.")
            return
        segment_colors = {
            "Champions": "#1b9e77",
            "Loyal": "#377eb8",
            "At-Risk": "#ff7f00",
            "Lost": "#d62728",
        }

        styled = filtered_rfm.style.map(
            lambda value: (
                f"background-color: {segment_colors.get(value, '#ffffff')}; color: white; font-weight: 600;"
                if value in segment_colors
                else ""
            ),
            subset=["Segment"],
        ).format({"Monetary": "{:,.2f}", "AvgOrderValue": "{:,.2f}"})
        st.dataframe(styled, use_container_width=True, height=420)

        col1, col2 = st.columns(2)
        with col1:
            pie_fig = px.pie(
                filtered_rfm,
                names="Segment",
                title="Segment Distribution",
                color="Segment",
                color_discrete_map=segment_colors,
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        with col2:
            scatter_fig = px.scatter(
                filtered_rfm,
                x="Recency",
                y="Monetary",
                color="Segment",
                size="Frequency",
                hover_data=["CustomerID", "Country"],
                title="Recency vs Monetary by Segment",
                color_discrete_map=segment_colors,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        st.download_button(
            label="Download RFM Segments CSV",
            data=filtered_rfm.to_csv(index=False).encode("utf-8"),
            file_name="rfm_segments.csv",
            mime="text/csv",
        )
    except Exception as exc:
        LOGGER.exception("Failed to render segmentation page.")
        st.error(f"Unable to render segmentation page: {exc}")


def render_churn_page(filtered_churn: pd.DataFrame) -> None:
    """Render the churn prediction page."""

    try:
        st.subheader("Churn Predictor")
        if filtered_churn.empty:
            st.warning("No churn records match the current filters.")
            return
        bundle = load_model_bundle()
        model = bundle["model"]

        with st.form("manual_churn_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                recency = st.number_input("Recency (days)", min_value=0, value=45)
                frequency = st.number_input("Frequency", min_value=1, value=6)
                monetary = st.number_input("Monetary", min_value=0.0, value=420.0, step=10.0)
                avg_order_value = st.number_input("Avg Order Value", min_value=0.0, value=70.0, step=5.0)
                avg_items_per_order = st.number_input("Avg Items / Order", min_value=1.0, value=3.0, step=1.0)
            with col2:
                revenue_90d = st.number_input("Revenue Last 90 Days", min_value=0.0, value=150.0, step=10.0)
                revenue_180d = st.number_input("Revenue Last 180 Days", min_value=1.0, value=320.0, step=10.0)
                orders_90d = st.number_input("Orders Last 90 Days", min_value=0, value=2)
                orders_180d = st.number_input("Orders Last 180 Days", min_value=0, value=4)
                lifetime_days = st.number_input("Customer Lifetime (days)", min_value=1, value=240)
            with col3:
                distinct_products = st.number_input("Distinct Products", min_value=1, value=8)
                avg_unit_price = st.number_input("Avg Unit Price", min_value=0.5, value=24.0, step=1.0)
                country = st.selectbox("Country", sorted(filtered_churn["Country"].dropna().unique().tolist()))
                segment = st.selectbox("Segment", ["Champions", "Loyal", "At-Risk", "Lost"], index=1)

            submitted = st.form_submit_button("Predict Churn Risk")

        if submitted:
            manual_row = build_manual_feature_row(
                {
                    "recency": recency,
                    "frequency": frequency,
                    "monetary": monetary,
                    "avg_order_value": avg_order_value,
                    "avg_items_per_order": avg_items_per_order,
                    "revenue_90d": revenue_90d,
                    "revenue_180d": revenue_180d,
                    "orders_90d": orders_90d,
                    "orders_180d": orders_180d,
                    "customer_lifetime_days": lifetime_days,
                    "distinct_products": distinct_products,
                    "avg_unit_price": avg_unit_price,
                    "country": country,
                    "segment": segment,
                }
            )
            churn_prob = float(model.predict_proba(manual_row)[:, 1][0])
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Manual Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#d62728" if churn_prob >= 0.5 else "#1b9e77"},
                        "steps": [
                            {"range": [0, 50], "color": "#d9f0d3"},
                            {"range": [50, 100], "color": "#fcbba1"},
                        ],
                    },
                )
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(
                f"### {'High Risk 🚨' if churn_prob >= 0.5 else 'Low Risk ✅'}"
            )

        top_risk = filtered_churn.nlargest(10, "ChurnProbability")
        st.markdown("### Top 10 Churn Risk Customers")
        st.dataframe(
            top_risk.style.format({"Monetary": "{:,.2f}", "ChurnProbability": "{:.2%}"}),
            use_container_width=True,
            height=380,
        )
        st.download_button(
            label="Download Churn Predictions CSV",
            data=filtered_churn.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        LOGGER.exception("Failed to render churn page.")
        st.error(f"Unable to render churn page: {exc}")


def render_shap_page() -> None:
    """Render the SHAP explainability page."""

    try:
        st.subheader("SHAP Insights")
        st.plotly_chart(plot_global_importance(), use_container_width=True)

        shap_bundle = get_shap_values()
        customer_ids = shap_bundle["customer_ids"]
        default_customer = int(customer_ids[0])
        selected_customer = st.number_input("Enter CustomerID", min_value=int(min(customer_ids)), value=default_customer)

        if int(selected_customer) in customer_ids:
            waterfall_fig = plot_individual_explanation(int(selected_customer))
            st.pyplot(waterfall_fig, clear_figure=True)
            st.markdown(
                "This explanation shows which customer behaviors pushed the churn prediction up or down. "
                "Positive SHAP contributions increase churn risk, while negative contributions indicate stronger retention signals."
            )
        else:
            st.warning("CustomerID not found in the current model dataset.")
    except Exception as exc:
        LOGGER.exception("Failed to render SHAP page.")
        st.error(f"Unable to render SHAP page: {exc}")


def render_summary_page(
    filtered_transactions: pd.DataFrame,
    filtered_rfm: pd.DataFrame,
    filtered_churn: pd.DataFrame,
) -> None:
    """Render the business summary page."""

    try:
        st.subheader("Business Summary")
        if filtered_transactions.empty:
            st.warning("No transactions match the current filters.")
            return
        total_customers = int(filtered_rfm["CustomerID"].nunique())
        churn_pct = float((filtered_churn["RiskLabel"] == "High Risk").mean() * 100) if len(filtered_churn) else 0.0
        champions = int((filtered_rfm["Segment"] == "Champions").sum())
        avg_order_value = float(filtered_transactions["Revenue"].mean()) if len(filtered_transactions) else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churn %", f"{churn_pct:.1f}%")
        col3.metric("Champions", f"{champions:,}")
        col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

        monthly = (
            filtered_transactions.assign(Month=lambda df: df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month", as_index=False)["Revenue"]
            .sum()
        )
        trend_fig = px.line(monthly, x="Month", y="Revenue", title="Monthly Revenue Trend", markers=True)
        st.plotly_chart(trend_fig, use_container_width=True)

        country_revenue = (
            filtered_transactions.groupby("Country", as_index=False)["Revenue"].sum().nlargest(5, "Revenue")
        )
        country_fig = px.bar(
            country_revenue,
            x="Country",
            y="Revenue",
            title="Top 5 Countries by Revenue",
            color="Revenue",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(country_fig, use_container_width=True)
    except Exception as exc:
        LOGGER.exception("Failed to render summary page.")
        st.error(f"Unable to render summary page: {exc}")


def main() -> None:
    """Run the Streamlit application."""

    try:
        st.title("E-Commerce Customer Intelligence Platform")

        with st.spinner("Loading retail data, segments, model, and explanations..."):
            data_bundle = load_platform_data()

        transactions = data_bundle["transactions"]
        rfm = data_bundle["rfm"]
        churn_predictions = data_bundle["churn_predictions"]

        st.sidebar.header("Filters")
        min_date = transactions["InvoiceDate"].min().date()
        max_date = transactions["InvoiceDate"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            selected_dates = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        else:
            selected_dates = (pd.Timestamp(min_date), pd.Timestamp(max_date))

        all_countries = sorted(transactions["Country"].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect("Country", options=all_countries, default=all_countries)
        all_segments = ["Champions", "Loyal", "At-Risk", "Lost"]
        selected_segments = st.sidebar.multiselect("Segment", options=all_segments, default=all_segments)

        filtered_transactions = filter_transactions(transactions, selected_dates, selected_countries)
        filtered_rfm, filtered_churn = filter_customer_views(
            rfm,
            churn_predictions,
            selected_countries,
            selected_segments,
        )

        page = st.sidebar.radio(
            "Navigate",
            [
                "Customer Segmentation",
                "Churn Predictor",
                "SHAP Insights",
                "Business Summary",
            ],
        )

        if page == "Customer Segmentation":
            render_segmentation_page(filtered_rfm)
        elif page == "Churn Predictor":
            render_churn_page(filtered_churn)
        elif page == "SHAP Insights":
            render_shap_page()
        else:
            render_summary_page(filtered_transactions, filtered_rfm, filtered_churn)
    except Exception as exc:
        LOGGER.exception("Application error.")
        st.error(f"Application failed to start: {exc}")


if __name__ == "__main__":
    main()
