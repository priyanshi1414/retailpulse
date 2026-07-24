\# RetailPulse — AI-Powered Customer Analytics \& Demand Forecasting



An end-to-end retail analytics platform combining customer segmentation, churn prediction,

demand forecasting, and inventory optimization in an interactive Streamlit dashboard.



\## Live Demo

\[Add your Streamlit Cloud URL here]



\## Features



| Page | What it does |

|---|---|

| Overview | Revenue, orders, and customer KPIs with daily trend and top-product charts |

| Demand Forecasting | Prophet-based forecast with adjustable horizon and promo-lift slider; live Prophet vs. LSTM vs. Hybrid comparison against a 30-day holdout |

| Customer Segments | RFM-based K-Means clustering (4 segments) with recency/monetary visualization |

| Churn Risk | Tuned XGBoost churn model (AUC-ROC 0.779), at-risk customer export |

| Inventory Optimizer | Reorder point recommendations with high-risk stockout alerts |

| Model Monitoring | Live data drift detection (Kolmogorov-Smirnov test) comparing recent vs. historical transaction data |



\## Tech Stack



\- \*\*Language:\*\* Python 3.11

\- \*\*Dashboard:\*\* Streamlit

\- \*\*ML/Forecasting:\*\* scikit-learn, XGBoost, Prophet, TensorFlow (LSTM)

\- \*\*Experiment Tracking:\*\* MLflow

\- \*\*Explainability:\*\* SHAP

\- \*\*Containerization:\*\* Docker



\## Project Structure

retailpulse-app/

├── app.py # Streamlit dashboard

├── Dockerfile # Multi-stage container build

├── requirements.txt

├── prophet\_model.pkl

├── lstm\_model.h5

├── lstm\_scaler.pkl

├── churn\_model\_tuned.pkl

├── rfm\_segments.csv

├── inventory\_recommendations.csv

├── online\_retail\_cleaned.csv

├── model\_performance\_summary.csv

├── daily\_sales\_prepped.csv

└── Untitled.ipynb # Full training/EDA notebook

\## Running Locally



```bash

pip install -r requirements.txt

streamlit run app.py

```



\## Running with Docker



```bash

docker build -t retailpulse .

docker run -p 8501:8501 retailpulse

```



\## Model Details



\*\*Churn Prediction:\*\* Trained on `Frequency` and `Monetary` only — `Recency` is deliberately

excluded because it directly defines the churn label (`Churn = Recency > 90`), and including

it would cause label leakage. This results in a realistic AUC-ROC of 0.779 rather than an

inflated score.



\*\*Customer Segmentation:\*\* 4 clusters chosen via the elbow method (tested k=1–10), with a

silhouette score of 0.591.



\*\*Demand Forecasting:\*\* Prophet and LSTM (30-day lookback window) combined via simple

averaging into a hybrid ensemble; MAPE tracked and logged for all three approaches.



\*\*Experiment Tracking:\*\* All training runs (churn, segmentation, forecasting) are logged to

MLflow with parameters and metrics. Run `mlflow ui` to view them.



\## Known Limitations / Next Steps



\- Automated retraining was designed as an Airflow DAG but not deployed live (local

&#x20; environment constraints)

\- Kubernetes/Prometheus/Grafana are documented in the architecture but not deployed for

&#x20; this submission; current deployment uses Streamlit Cloud + Docker

\- Customer segmentation currently uses 4 clusters; could be extended to 6–8 for finer-

&#x20; grained segments per original scope



\## Author

\[Your Name]

