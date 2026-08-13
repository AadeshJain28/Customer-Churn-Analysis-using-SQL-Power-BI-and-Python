"""Streamlit churn dashboard.

    streamlit run app/streamlit_app.py

Four tabs: score a customer, the retention economics, how the model compares with the
one-line rule, and the data audit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Customer Churn", layout="wide")


@st.cache_resource
def load_model():
    p = ROOT / "models" / "churn_model.joblib"
    if not p.exists():
        return None
    import joblib

    return joblib.load(p)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "Customer_Data.csv")
    return df[df.Customer_Status.isin(["Churned", "Stayed"])].copy()


def read_json(name: str):
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(name: str):
    p = ROOT / "reports" / name
    return pd.read_csv(p) if p.exists() else None


df = load_data()
bundle = load_model()

st.title("Customer churn — prediction and retention targeting")
churn_rate = (df.Customer_Status == "Churned").mean()
c1, c2, c3 = st.columns(3)
c1.metric("Customers with a known outcome", f"{len(df):,}")
c2.metric("Churn rate", f"{churn_rate:.1%}")
c3.metric("Accuracy of predicting 'nobody churns'", f"{1 - churn_rate:.1%}")

tab_score, tab_econ, tab_rule, tab_audit = st.tabs(
    ["Score a customer", "Retention economics", "Model vs the one-line rule", "Data audit"]
)

with tab_score:
    if bundle is None:
        st.warning("No trained model. Run `make train`.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            age = st.number_input("Age", 18, 100, 40)
        with c2:
            monthly = st.number_input("Monthly charge", 0.0, 500.0, 75.0)
            internet = st.selectbox("Internet type", sorted(df.Internet_Type.dropna().unique()))
            payment = st.selectbox("Payment method", sorted(df.Payment_Method.dropna().unique()))
        with c3:
            married = st.selectbox("Married", ["Yes", "No"])
            referrals = st.number_input("Referrals", 0, 20, 0)
            state = st.selectbox("State", sorted(df.State.dropna().unique()))

        if st.button("Score", type="primary"):
            from customer_churn.config import Config
            from customer_churn.features import prepare_inference_frame

            cfg = Config.load()
            row = {c: df[c].mode()[0] for c in df.columns if c != "Customer_Status"}
            row.update({
                "Contract": contract, "Tenure_in_Months": tenure, "Age": age,
                "Monthly_Charge": monthly, "Internet_Type": internet,
                "Payment_Method": payment, "Married": married,
                "Number_of_Referrals": referrals, "State": state,
            })
            # Routed through the same coercion the training pipeline uses. Building the
            # frame by hand here is what caused the "could not convert string to float:
            # 'Yes'" failure: the CSV and the DuckDB view disagree on dtypes.
            frame = prepare_inference_frame(pd.DataFrame([row]), cfg)
            proba = float(bundle["pipeline"].predict_proba(frame)[0, 1])
            t = bundle["threshold"]
            a, b = st.columns(2)
            a.metric("Churn probability", f"{proba:.1%}")
            b.metric(f"Flag at threshold {t:.2f}?", "YES" if proba >= t else "no")
            if (proba >= t) != (contract == "Month-to-Month"):
                st.info(
                    "The model and the month-to-month rule disagree here — these are the "
                    "only customers where the model changes the decision."
                )

with tab_econ:
    curve = read_csv("threshold_curve.csv")
    if curve is None:
        st.info("Run `make train` to generate the cost curve.")
    else:
        st.subheader("Expected cost by decision threshold")
        st.line_chart(curve.set_index("threshold")["expected_cost"])
        st.caption(
            "Cost = (customers contacted x intervention cost) + (churners missed x lost value). "
            "The minimum, not 0.5, is the operating point."
        )
        st.subheader("Recall and reach by threshold")
        st.line_chart(curve.set_index("threshold")[["recall", "flagged_share"]])
        sens = read_csv("cost_sensitivity.csv")
        if sens is not None:
            st.subheader("How much does the threshold depend on the assumed cost ratio?")
            st.dataframe(sens, use_container_width=True)

with tab_rule:
    base = read_csv("baselines.csv")
    lb = read_csv("leaderboard.csv")
    if base is None:
        st.info("Run `make audit` or `make train`.")
    else:
        st.subheader("Baselines — no model involved")
        st.dataframe(base, use_container_width=True)
        st.caption(
            "`rule_month_to_month` flags every month-to-month contract. It is the bar a "
            "trained model has to clear to justify existing."
        )
    if lb is not None:
        st.subheader("Model leaderboard")
        st.dataframe(lb, use_container_width=True)

with tab_audit:
    audit = read_json("data_audit.json")
    if audit:
        st.json(audit)
    summary = read_json("summary.json")
    if summary:
        st.subheader("Training summary")
        st.json(summary)
    st.subheader("Power BI report")
    st.markdown(
        "The descriptive dashboard lives in `powerbi/Churn Analysis Project.pbix` and reads "
        "the same views defined in `sql/`. See `powerbi/README.md`."
    )
