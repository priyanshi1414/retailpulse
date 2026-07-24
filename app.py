import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="RetailPulse", layout="wide")

st.sidebar.title("RetailPulse")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Demand Forecasting", "Customer Segments", "Churn Risk", "Inventory Optimizer", "Model Monitoring"]
)

# ------------------------------------------------------
# data + model loading
# ------------------------------------------------------
data = pd.read_csv("online_retail_cleaned.csv")
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

summary = pd.read_csv("model_performance_summary.csv")

with open("prophet_model.pkl", "rb") as f:
    prophet_model = pickle.load(f)

with open("churn_model_tuned.pkl", "rb") as f:
    churn_model = pickle.load(f)

rfm = pd.read_csv("rfm_segments.csv")

inventory = pd.read_csv("inventory_recommendations.csv")
inventory = inventory.rename(columns={inventory.columns[0]: "StockCode"})
# ------------------------------------------------------
# OVERVIEW (day 15 skeleton, still placeholder)
# ------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.write("Quick snapshot of overall sales performance.")

    total_revenue = data["Revenue"].sum()
    total_orders = data["Invoice"].nunique()
    total_customers = data["Customer ID"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_customers:,}")

    st.subheader("Daily Revenue Trend")
    daily = data.groupby("InvoiceDate")["Revenue"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

    st.subheader("Top 10 Products")
    top_products = data.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    top_products.plot(kind="bar", ax=ax2)
    st.pyplot(fig2)


# ------------------------------------------------------
# DEMAND FORECASTING (day 16, real prophet model)
# ------------------------------------------------------
if page == "Demand Forecasting":
    st.title("Demand Forecasting")
    st.caption("Prophet based forecasting, using the real trained model from earlier notebooks.")

    prophet_row = summary[summary["Model"] == "Prophet"]
    hybrid_row = summary[summary["Model"] == "Hybrid"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Prophet MAPE", f"{prophet_row['Score'].values[0]:.2f} %")
    col2.metric("Forecast Horizon", "30 Days")
    col3.metric("Hybrid MAPE", f"{hybrid_row['Score'].values[0]:.2f} %")

    st.write("---")

    horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
    promo_lift = st.slider("Promo Lift (%)", min_value=0, max_value=50, value=0)

    st.subheader("Forecast Visualization")

    future = prophet_model.make_future_dataframe(periods=horizon)
    forecast = prophet_model.predict(future)
    forecast["yhat_adjusted"] = forecast["yhat"] * (1 + promo_lift / 100)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(forecast["ds"], forecast["yhat_adjusted"], label="forecast")
    ax3.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
    ax3.set_title("Demand Forecast")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Forecast Table (last rows)")
    st.dataframe(forecast[["ds", "yhat_adjusted"]].tail(horizon))


# ------------------------------------------------------
# CUSTOMER SEGMENTS (day 17, real rfm clusters)
# ------------------------------------------------------
if page == "Customer Segments":
    st.title("Customer Segmentation")
    st.caption("RFM segments from KMeans clustering, computed in the notebook.")

    st.subheader("Segment Summary")
    st.dataframe(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1))

    st.subheader("Segment Visualization")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="tab10", ax=ax4)
    st.pyplot(fig4)

    st.subheader("Customer Table (sample)")
    st.dataframe(rfm.head(20))


# ------------------------------------------------------
# CHURN RISK (day 17, real tuned xgboost model)
# ------------------------------------------------------
if page == "Churn Risk":
    st.title("Churn Risk")
    st.caption("Predictions from the tuned XGBoost churn model.")

    features = ["Frequency", "Monetary"]
    rfm_scored = rfm.copy()
    rfm_scored["churn_probability"] = churn_model.predict_proba(rfm_scored[features])[:, 1]
    rfm_scored["at_risk"] = rfm_scored["churn_probability"] > 0.5

    col1, col2 = st.columns(2)
    col1.metric("Customers At Risk", int(rfm_scored["at_risk"].sum()))
    col2.metric("Percent At Risk", f"{rfm_scored['at_risk'].mean()*100:.1f} %")

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    rfm_scored["at_risk"].value_counts().plot(kind="bar", ax=ax5)
    ax5.set_xticklabels(["Active", "At Risk"], rotation=0)
    st.pyplot(fig5)

    st.subheader("Highest Risk Customers")
    st.dataframe(
        rfm_scored[rfm_scored["at_risk"]].sort_values("churn_probability", ascending=False).head(20)
    )

    st.write("---")
    st.subheader("Export")

    churn_csv = rfm_scored[rfm_scored["at_risk"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download At-Risk Customers (CSV)", churn_csv, "at_risk_customers.csv", "text/csv")

# ------------------------------------------------------
# INVENTORY OPTIMIZER (day 15 skeleton, still placeholder - day 18 next)
# ------------------------------------------------------
if page == "Inventory Optimizer":
    st.title("Inventory Optimizer")
    st.caption("Reorder point recommendations from the Day 10 inventory logic.")

    st.dataframe(inventory.head(20))

    inventory_clean = inventory[inventory["StockCode"] != "M"]

    top_10 = inventory_clean.sort_values("reorder_point", ascending=False).head(10)
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.bar(top_10["StockCode"].astype(str), top_10["reorder_point"])
    ax6.set_yscale("log")
    plt.xticks(rotation=45)
    ax6.set_title("Top 10 products by reorder point")
    st.pyplot(fig6)

    
    st.write("---")
    st.subheader("Alerts")

    threshold = inventory_clean["risk_score"].quantile(0.9)
    high_risk_products = inventory_clean[inventory_clean["risk_score"] > threshold]

    if len(high_risk_products) > 0:
        st.warning(f"{len(high_risk_products)} products flagged as high stockout risk")
        st.dataframe(high_risk_products[["StockCode", "avg_daily_demand", "demand_std", "risk_score"]].head(10))
    else:
        st.success("No high-risk stockout alerts right now")

    st.write("---")
    st.subheader("Export")

    inv_csv = inventory_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download Inventory Report (CSV)", inv_csv, "inventory_report.csv", "text/csv")


# ------------------------------------------------------
# MODEL MONITORING (day 15 skeleton, still placeholder)
# ------------------------------------------------------
if page == "Model Monitoring":
    st.title("Model Monitoring")
    st.caption("Model performance summary and live drift check.")

    st.subheader("Model Performance Summary")
    st.dataframe(summary)

    st.write("---")
    st.subheader("Data Drift Check")

    from scipy.stats import ks_2samp

    cutoff = data["InvoiceDate"].max() - pd.Timedelta(days=60)
    old_data = data[data["InvoiceDate"] < cutoff]
    new_data = data[data["InvoiceDate"] >= cutoff]

    st.caption(f"Old: before {cutoff.date()} ({len(old_data):,} rows) vs. "
               f"New: last 60 days ({len(new_data):,} rows)")

    stat_q, p_q = ks_2samp(old_data["Quantity"], new_data["Quantity"])
    stat_p, p_p = ks_2samp(old_data["Price"], new_data["Price"])

    drift_df = pd.DataFrame({
        "Feature": ["Quantity", "Price"],
        "KS Statistic": [round(stat_q, 4), round(stat_p, 4)],
        "p-value": [round(p_q, 4), round(p_p, 4)],
        "Drift Detected (p<0.05)": [p_q < 0.05, p_p < 0.05]
    })
    st.dataframe(drift_df)

    if drift_df["Drift Detected (p<0.05)"].any():
        st.warning("Drift detected — consider retraining.")
    else:
        st.success("No significant drift detected.")

    fig_d, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(old_data["Quantity"].clip(upper=50), bins=30, alpha=0.5, label="old")
    ax1.hist(new_data["Quantity"].clip(upper=50), bins=30, alpha=0.5, label="new")
    ax1.set_title("Quantity: old vs new"); ax1.legend()
    ax2.hist(old_data["Price"].clip(upper=50), bins=30, alpha=0.5, label="old")
    ax2.hist(new_data["Price"].clip(upper=50), bins=30, alpha=0.5, label="new")
    ax2.set_title("Price: old vs new"); ax2.legend()
    st.pyplot(fig_d)

    st.write("---")
    st.write(
        "Automated retraining was designed as an Airflow DAG but was not deployed "
        "live due to Windows environment constraints."
    )