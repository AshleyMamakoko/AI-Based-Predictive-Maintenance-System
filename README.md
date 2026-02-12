Ashley — this README is structured for:

Professional GitHub presentation

Recruiter visibility

Academic submission clarity

Clean technical credibility (no inflated claims)

You can paste this directly into README.md.

⚡ Smart Grid Predictive Maintenance System

AI-powered energy load forecasting and risk assessment system for smart grid operations.

📌 Overview

This project transforms an LSTM energy forecasting model into a deployable predictive maintenance system.

The platform:

Forecasts next-hour energy load using historical data

Detects anomalous load behaviour

Calculates multi-factor equipment risk scores

Generates maintenance recommendations

Monitors model performance in production

Supports model iteration and A/B testing

The system demonstrates how a machine learning prototype can be engineered into a production-ready application.

🏗 System Architecture
Client (Dashboard / API Consumer)
            ↓
Flask REST API
            ↓
Model Manager (LSTM Inference + Risk Engine)
            ↓
PostgreSQL Database
            ↓
Monitoring & Alerts


The application is modular, container-ready, and designed for horizontal scalability.

🧠 AI Model
Model Type

LSTM (Long Short-Term Memory) Neural Network

Input

24-hour historical load values (MW)

Output

Predicted next-hour load

Risk classification

Maintenance recommendation

Risk Calculation (Weighted Model)
Total Risk =
0.3 × Forecast Error
+ 0.2 × Load Magnitude
+ 0.3 × Anomaly Score
+ 0.2 × Volatility


Risk levels:

Minimal

Low

Medium

High

Critical

🚀 Features

RESTful prediction API

Real-time dashboard

Statistical anomaly detection (z-score)

Multi-factor risk assessment

Automated alert generation

Model versioning system

Retraining pipeline

A/B testing framework

Monitoring endpoints

Docker deployment support

📂 Project Structure
smart_grid_maintenance/
│
├── api/                    # Flask application & routes
├── models/                 # LSTM model and risk engine
├── database/               # ORM models and DB configuration
├── monitoring/             # Metrics & alert tracking
├── pipeline/               # Data preprocessing & retraining
├── web/                    # Dashboard frontend
├── deployment/             # Docker configuration
├── tests/                  # Unit & integration tests
├── scripts/                # Utility scripts
├── saved_models/           # Versioned trained models
├── requirements.txt
└── run.py

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/smart-grid-maintenance.git
cd smart-grid-maintenance

2️⃣ Create Virtual Environment

Using Conda:

conda create -n ml_env python=3.13
conda activate ml_env


Or venv:

python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Environment Variables (Windows PowerShell)
$env:FLASK_ENV="development"
$env:SECRET_KEY="dev-secret"
$env:DATABASE_URL="sqlite:///smartgrid.db"


For production, use PostgreSQL.

5️⃣ Initialize Database
python -c "from api.app import create_app; from database.models import db; app = create_app(); app.app_context().push(); db.create_all()"

6️⃣ Run Application
python run.py


Open:

http://localhost:5000/dashboard

🐳 Docker Deployment
docker-compose up -d


Scale API instances:

docker-compose up -d --scale api=3

📡 API Endpoints
Endpoint	Method	Description
/api/v1/predict	POST	Single prediction
/api/v1/predict/batch	POST	Batch predictions
/api/v1/monitoring/performance	GET	Performance metrics
/api/v1/monitoring/alerts	GET	Active alerts
/health	GET	System health
🧪 Testing

Run unit tests:

pytest tests/ -v


Run integration test:

python tests/test_api.py

📊 Monitoring

The system tracks:

MAE

RMSE

MAPE

API latency

Risk distribution

Active alerts

Model versions

Monitoring endpoints provide real-time operational visibility.

🔁 Model Iteration Pipeline

The retraining script:

scripts/retrain_robust_model.py


Supports:

Performance drift detection

Versioned model saving

Metadata tracking

Automatic deployment of latest model

Saved model structure:

saved_models/
├── model_timestamp.h5
├── scaler.pkl
├── metadata.json

🛡 Security & Production Considerations

Environment-based configuration

Input validation

ORM-based database interaction

Health checks

Structured logging

Stateless API design

Horizontal scaling ready

🎯 Learning Objectives Demonstrated

ML model deployment

API engineering

Monitoring & observability

Model iteration workflows

Risk-based decision systems

Scalable architecture design

Production containerization

📌 Future Enhancements

Real-time SCADA integration

Weather data enrichment

Multi-asset correlation modelling

Advanced drift detection

Mobile interface for technicians

👨🏽‍💻 Author

Ashley Mamakoko
AI / Machine Learning & Systems Engineering
