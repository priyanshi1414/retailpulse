import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

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
    st.write("---")
    st.subheader("System Health (Day 26)")
    LOG_FILE = "events_log.csv"
    st.caption("Lightweight monitoring in place of Prometheus/Grafana — logged locally since we don't have a server for a full metrics stack.")

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events Logged", len(logs))
        col2.metric("Page Views", (logs["event"] == "page_view").sum())

        pred_logs = logs[logs["event"] == "prediction"]
        if len(pred_logs) > 0:
            avg_latency = pred_logs["duration_sec"].mean()
            col3.metric("Avg Forecast Latency", f"{avg_latency:.2f}s")

        st.write("**Page views over time**")
        views_over_time = logs[logs["event"] == "page_view"].groupby(logs["timestamp"].dt.date).size()
        st.line_chart(views_over_time)

        st.write("**Recent log entries**")
        st.dataframe(logs.tail(15))
    else:
        st.info("No monitoring data logged yet — interact with the dashboard to generate logs.")

    # ------------------------------------------------------
    # load testing (day 27)
    # simulates repeated rapid requests since we don't have
    # a server to run locust/jmeter against
    # ------------------------------------------------------
    st.write("---")
    st.subheader("Load Test (Day 27)")
    st.caption("Simulated repeated forecast requests, timed locally in place of Locust/JMeter.")

    if st.button("Run Load Test (20 requests)"):
        load_times = []
        progress = st.progress(0)
        for i in range(20):
            t0 = time.time()
            future_lt = prophet_model.make_future_dataframe(periods=30)
            _ = prophet_model.predict(future_lt)
            t1 = time.time()
            load_times.append(t1 - t0)
            progress.progress((i + 1) / 20)

        load_df = pd.DataFrame({"request": range(1, 21), "latency_sec": load_times})

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Latency", f"{load_df['latency_sec'].mean():.3f}s")
        col2.metric("Max Latency", f"{load_df['latency_sec'].max():.3f}s")
        col3.metric("Min Latency", f"{load_df['latency_sec'].min():.3f}s")

        st.line_chart(load_df.set_index("request")["latency_sec"])

        # log the load test as its own event
        log_event("Model Monitoring", "load_test", duration=round(load_df["latency_sec"].mean(), 3),
                   extra=f"{len(load_df)} requests")

    # ------------------------------------------------------
    # final accuracy validation against zidio targets (day 27)
    # ------------------------------------------------------
    st.write("---")
    st.subheader("Final Accuracy Validation (Day 27)")
    st.caption("Checking model performance against the targets set in the Zidio project brief.")

    hybrid_mape_val = summary[summary["Model"] == "Hybrid"]["Test_MAPE"].values[0]
    prophet_mape_val = summary[summary["Model"] == "Prophet"]["Test_MAPE"].values[0]

    mape_target = 12.0
    mape_pass = hybrid_mape_val <= mape_target

    val_col1, val_col2 = st.columns(2)
    with val_col1:
        st.metric("Hybrid MAPE (target ≤ 12%)", f"{hybrid_mape_val:.2f}%",
                   delta=f"{hybrid_mape_val - mape_target:.2f} pts", delta_color="inverse")
    with val_col2:
        st.write(" Target met" if mape_pass else " Target not met")

    st.write("")

    # churn model validation
    features = ["Frequency", "Monetary"]
    from sklearn.metrics import roc_auc_score

    # NOTE: this assumes rfm has a ground-truth churn label column.
    # if it doesn't, this section needs an actual labeled holdout set instead.
    if "Churn" in rfm.columns:
        churn_probs = churn_model.predict_proba(rfm[features])[:, 1]
        auc = roc_auc_score(rfm["Churn"], churn_probs)
        auc_target = 0.88
        auc_pass = auc >= auc_target

        val_col3, val_col4 = st.columns(2)
        with val_col3:
            st.metric("Churn AUC-ROC (target ≥ 0.88)", f"{auc:.3f}",
                       delta=f"{auc - auc_target:.3f}")
        with val_col4:
            st.write(" Target met" if auc_pass else " Target not met")
    else:
        st.info("Churn AUC-ROC check needs a ground-truth 'Churn' column in rfm_segments.csv — not present, so skipping this check on the live dashboard. Report this metric from your notebook's train/test split instead.")
