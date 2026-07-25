## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running with Docker

```bash
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

## Model Details

**Churn Prediction:** Trained on `Frequency` and `Monetary` only — `Recency` is deliberately
excluded because it directly defines the churn label (`Churn = Recency > 90`), and including
it would cause label leakage. This results in a realistic AUC-ROC of 0.779 rather than an
inflated score.

**Customer Segmentation:** 4 clusters chosen via the elbow method (tested k=1–10), with a
silhouette score of 0.591.

**Demand Forecasting:** Prophet and LSTM (30-day lookback window) combined via simple
averaging into a hybrid ensemble; MAPE tracked and logged for all three approaches.

**Experiment Tracking:** All training runs (churn, segmentation, forecasting) are logged to
MLflow with parameters and metrics. Run `mlflow ui` to view them.

## Known Limitations / Next Steps

- Automated retraining was designed as an Airflow DAG but not deployed live (local
  environment constraints)
- Kubernetes manifests are included under `/k8s` (validated, not deployed to a live cluster)
- Prometheus/Grafana monitoring is documented in the architecture but not deployed
- Customer segmentation currently uses 4 clusters; could be extended to 6–8 for finer-
  grained segments per original scope

## Author
Priyanshi Gaur