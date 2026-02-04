# EcoPackAI - AI-Powered Sustainable Packaging Recommendation System

EcoPackAI is a full-stack web platform designed to help businesses switch to sustainable packaging. It uses Machine Learning to recommend packaging materials that balance **Cost Efficiency**, **Durability**, and **Environmental Impact (CO₂)**.

## 🚀 Features
- **AI Recommendations**: Random Forest & XGBoost models predict Cost and CO₂ impact.
- **Dynamic Scoring**: Balances sustainability scores with operational metrics (weight, price, protection needs).
- **Category-Aware**: Recommends suitable materials based on product type (e.g., Food-grade vs. Electronics).
- **Interactive UI**: View predictions, compare variations, and see visual charts.
- **Database Logging**: All requests and predictions are stored in PostgreSQL for analytics.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, Bootstrap 5, Chart.js
- **Backend**: Python (Flask), Flask-CORS
- **Database**: PostgreSQL
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy

## 📂 Project Structure
```
EcoPackAI/
├── EcoPackAI_Backend-main/
│   ├── app.py                # Main Flask Application
│   ├── model/                # Trained ML Models (.pkl)
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # Product Input Form
│   ├── results.html          # Recommendation Dashboard
│   └── css/                  # Styles
├── scripts/
│   ├── train_model.py        # ML Training Script (Run this to retrain models)
│   └── setup_db.py           # Database Initialization Script
└── data/                     # Datasets (Materials & Products)
```

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- PostgreSQL installed and running locally

### 2. Install Dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib psycopg2
```

### 3. Database Setup
Ensure PostgreSQL is running. Then run the setup script to create the `ecopackai` database and tables:
```bash
python scripts/setup_db.py
```
*(Note: Default credentials are set to `postgres`/`Jaga@123`. Update `setup_db.py` and `app.py` if yours differ.)*

### 4. Train Models (Optional)
If you need to regenerate the models:
```bash
python scripts/train_model.py
```

## ▶️ Running the Application

### 1. Start the Backend
```bash
python EcoPackAI_Backend-main/app.py
```
Server will start at `http://localhost:5000`.

### 2. Launch the Frontend
Simply open **`frontend/index.html`** in your web browser.

## 📊 Evaluation Metrics
- **Cost Prediction**: Random Forest Regressor
- **CO2 Prediction**: XGBoost Regressor
- **Sustainability Score**: Custom weighted formula considering Biodegradability, Recyclability, Cost, and Emissions.

## 👥 Authors
- **Frontend**: [Your Name/Team]
- **Backend & ML**: [Teammate Names]
