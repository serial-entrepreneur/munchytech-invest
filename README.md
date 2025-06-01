# Quantitative Trading Engine

A modular, scalable, and AI-driven algorithmic trading system leveraging Kite Connect for real-time market data and order execution, along with advanced ML pipelines for research, signal generation, and model deployment.

## 📌 Features

### Data Sources
- Market Data & Order Book: [Kite Connect](https://kite.trade/)
- Fundamentals: Screener.in API

### Storage
- Time Series: Parquet (via PySpark)
- Alternative: PostgreSQL & MongoDB

### Data Pipeline
- Custom validation & cleaning
- Missing data handling
- Versioned schema control

### Feature Engineering
- Technical indicators (`pandas-ta`, `finta`, `ta-lib`)
- Statistical & PCA-based features
- Order book depth metrics
- Fundamental ratios

### Targets
- Price direction (classification)
- Price regression (future returns)
- Volatility classes
- Asset ranking (based on predictive signals)

### Modeling
- Pipelines with `scikit-learn`, `xgboost`, `lightgbm`, `lstm`
- Hyperparameter tuning via `optuna`
- Experiment tracking with `mlflow`

### Testing & Standards
- Follows **SOLID** principles
- Linting: `flake8`, formatting: `black`
- Unit, integration & E2E testing with `pytest`
- Type safety with `mypy`

### DevOps
- Dockerized services
- CI/CD with GitHub Actions
- Logging via `loguru`

## 📁 Project Structure

```
quant-trading-engine/
│
├── data_ingestion/     # Kite Connect, Screener ingestion
├── data_validation/    # Schema & data checks
├── feature_engineering/# All features logic
├── target_generation/  # Target labeling and configs
├── models/            # Training, pipelines, evaluations
├── notebooks/         # EDA & prototyping
├── tests/            # Unit and integration tests
├── config/           # .env, schema files, configs
├── utils/            # Logging, constants, helpers
├── docker/           # Dockerfiles, docker-compose
└── README.md
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- Docker (recommended)
- Kite API Key & Screener credentials

### Installation

```bash
git clone https://github.com/yourusername/quant-trading-engine.git
cd quant-trading-engine

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp config/.env.example config/.env

# Run linting & tests
flake8 .
pytest
```

## 🧪 Testing

- Unit tests: `pytest tests/unit/`
- Integration tests: `pytest tests/integration/`
- E2E tests: `pytest tests/e2e/`

## 🧠 ML & AI Models

- Classical ML: RandomForest, XGBoost, LightGBM
- Deep Learning: LSTM (coming soon)
- Ensemble strategies: Voting & Stacking
- Reinforcement learning: Planned in v1.0

## 🚀 Roadmap

| Version | Features |
|---------|----------|
| v0.1    | Data ingestion, validation & storage |
| v0.2    | Feature engineering |
| v0.3    | Target generation |
| v0.4    | ML training pipelines |
| v0.5    | Evaluation & full E2E testing |
| v1.0    | Real-time trading loop, RL agents, dashboard |

## 🤝 Contributing

PRs are welcome! Please read the CONTRIBUTING.md and ensure all new code is tested and linted.

## 📄 License

MIT License

## 📬 Contact

- Maintainer – Your Name
- Project Homepage – GitHub