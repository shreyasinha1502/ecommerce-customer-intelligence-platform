"""Data generation and cleaning utilities for the e-commerce intelligence app."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "online_retail.csv"


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
def generate_synthetic_data(n_rows: int = 80_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic e-commerce transactions dataset and save it."""

    try:
        LOGGER.info("Generating synthetic retail data with %s rows.", n_rows)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        customer_ids = np.arange(10_000, 16_500)
        countries = {
            "United Kingdom": 1.00,
            "Germany": 1.15,
            "France": 1.10,
            "Netherlands": 1.18,
            "Spain": 1.05,
            "Ireland": 1.08,
            "Belgium": 1.12,
            "Sweden": 1.20,
            "Australia": 1.35,
            "United States": 1.28,
        }
        products = [
            ("Wireless Mouse", 24.99),
            ("Mechanical Keyboard", 78.50),
            ("Laptop Stand", 39.95),
            ("USB-C Hub", 34.20),
            ("Noise Cancelling Headphones", 129.99),
            ("Smart Water Bottle", 31.40),
            ("Office Chair Cushion", 26.75),
            ("Notebook Set", 12.80),
            ("Ceramic Coffee Mug", 14.50),
            ("Portable SSD 1TB", 119.00),
            ("LED Desk Lamp", 28.60),
            ("Fitness Tracker", 66.90),
            ("Yoga Mat", 25.30),
            ("Travel Backpack", 58.25),
            ("Bluetooth Speaker", 48.99),
            ("Air Purifier Filter", 22.40),
            ("Phone Case", 16.95),
            ("Skincare Gift Box", 44.70),
            ("Organic Tea Collection", 18.25),
            ("Gaming Controller", 54.80),
            ("Kitchen Storage Jar", 10.90),
            ("Smart Plug", 21.60),
            ("Baby Blanket", 29.10),
            ("Pet Toy Bundle", 17.45),
            ("Scented Candle", 15.20),
        ]

        country_names = np.array(list(countries.keys()))
        country_probs = np.array([0.34, 0.10, 0.09, 0.06, 0.07, 0.05, 0.05, 0.04, 0.10, 0.10])
        product_names = np.array([name for name, _ in products])
        product_prices = np.array([price for _, price in products])

        invoices_needed = int(n_rows / 3.5) + 2_000
        invoice_customer = rng.choice(customer_ids, size=invoices_needed, replace=True)
        invoice_country = rng.choice(country_names, size=invoices_needed, p=country_probs)
        line_counts = rng.choice(np.arange(1, 8), size=invoices_needed, p=[0.16, 0.22, 0.21, 0.16, 0.11, 0.08, 0.06])

        start_date = np.datetime64("2023-01-01")
        end_date = np.datetime64("2024-12-31")
        total_days = int((end_date - start_date).astype("timedelta64[D]").astype(int))
        invoice_dates = start_date + rng.integers(0, total_days + 1, size=invoices_needed).astype("timedelta64[D]")
        invoice_minutes = rng.integers(8 * 60, 21 * 60, size=invoices_needed)
        invoice_datetimes = pd.to_datetime(invoice_dates) + pd.to_timedelta(invoice_minutes, unit="m")

        rows = []
        invoice_counter = 100000
        for idx in range(invoices_needed):
            invoice_no = f"INV{invoice_counter + idx}"
            customer_id = int(invoice_customer[idx])
            country = invoice_country[idx]
            country_multiplier = countries[country]
            for _ in range(int(line_counts[idx])):
                product_idx = int(rng.integers(0, len(product_names)))
                base_price = product_prices[product_idx]
                quantity = int(rng.choice([1, 2, 3, 4, 5, 6, 8, 10], p=[0.22, 0.25, 0.18, 0.12, 0.10, 0.06, 0.04, 0.03]))
                unit_price = round(base_price * country_multiplier * rng.uniform(0.88, 1.18), 2)
                rows.append(
                    {
                        "CustomerID": customer_id,
                        "InvoiceDate": invoice_datetimes[idx],
                        "InvoiceNo": invoice_no,
                        "Quantity": quantity,
                        "UnitPrice": max(unit_price, 2.50),
                        "Description": product_names[product_idx],
                        "Country": country,
                    }
                )
                if len(rows) >= n_rows:
                    break
            if len(rows) >= n_rows:
                break

        df = pd.DataFrame(rows)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df.to_csv(DATA_PATH, index=False)
        LOGGER.info("Synthetic retail data saved to %s", DATA_PATH)
        return df
    except Exception as exc:
        LOGGER.exception("Failed to generate synthetic data.")
        raise RuntimeError("Synthetic data generation failed.") from exc


@cache_data(show_spinner=False)
def load_and_clean_data() -> pd.DataFrame:
    """Load and clean the retail transactions dataset."""

    try:
        if not DATA_PATH.exists():
            generate_synthetic_data()

        LOGGER.info("Loading retail data from %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])
        df.columns = [col.strip() for col in df.columns]
        df["CustomerID"] = df["CustomerID"].astype(int)
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
        df["Description"] = df["Description"].astype(str).str.strip()
        df["Country"] = df["Country"].astype(str).str.strip()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).clip(lower=1).astype(int)
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0).clip(lower=0.5)
        df = df.dropna(subset=["InvoiceDate", "CustomerID", "InvoiceNo", "Description", "Country"])
        df = df.drop_duplicates()
        df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
        LOGGER.info("Retail data loaded with %s rows after cleaning.", len(df))
        return df
    except Exception as exc:
        LOGGER.exception("Failed to load and clean data.")
        raise RuntimeError("Data loading and cleaning failed.") from exc


if not DATA_PATH.exists():
    generate_synthetic_data()
