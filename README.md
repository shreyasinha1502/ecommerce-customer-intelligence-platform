# E-Commerce Customer Intelligence Platform

This project is a production-ready Streamlit application that generates a realistic synthetic e-commerce dataset, performs RFM segmentation, trains an XGBoost churn model, and explains churn predictions with SHAP.

## What it does

- Auto-generates `data/online_retail.csv` with 80,000 realistic retail transaction rows.
- Cleans data and computes customer-level RFM metrics.
- Segments customers into `Champions`, `Loyal`, `At-Risk`, and `Lost`.
- Engineers churn features and trains an XGBoost classifier with class imbalance handling.
- Saves model and prediction artifacts automatically.
- Provides a multi-page Streamlit dashboard with segmentation, churn prediction, SHAP insights, and business KPIs.

## Project Structure

```text
.
├── app.py
├── src/
│   ├── data_loader.py
│   ├── rfm.py
│   ├── churn.py
│   └── shap_explain.py
├── models/
│   └── churn_model.pkl
├── data/
│   └── online_retail.csv
├── outputs/
│   ├── rfm_segments.csv
│   └── churn_predictions.csv
├── requirements.txt
└── README.md
```
## Deployment
### Application Link :
https://ecommerce-customer-intelligence-platform-hduzpx9dvqsw27pbpracx.streamlit.app/

## Application Preview
![Project Snapshot1](https://github.com/shreyasinha1502/ecommerce-customer-intelligence-platform/blob/main/Capture1.PNG)
![Project Snapshot2](https://github.com/shreyasinha1502/ecommerce-customer-intelligence-platform/blob/main/Capture2.PNG)
![Project Snapshot3](https://github.com/shreyasinha1502/ecommerce-customer-intelligence-platform/blob/main/Capture3.PNG)


