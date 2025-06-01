# AI/ML-Driven Algorithmic Trading Framework PRD

## Overview

This Product Requirements Document (PRD) outlines the design and development roadmap for an AI/ML-driven algorithmic trading framework. The system will:

- Ingest market, order book, and fundamental data
- Store and validate data
- Generate comprehensive features
- Support multiple targets for diverse analyses
- Evolve through version-wise increments

### Key Components

- **Market & Order Book Data**: Via Zerodha's Kite Connect
- **Fundamental Data**: Via Screener API
- **Time-Series Storage**: Parquet (PySpark)
- **Alternative Data Storage**: PostgreSQL & MongoDB
- **Data Handling**: Manual missing-data handling and validation
- **Feature Engineering**: Using all major Python libraries
- **Multiple Targets**: Support for regression, classification, clustering, and RL

## Objectives

1. Establish a robust data pipeline that ingests, stores, and validates raw market, order book, and fundamental data.
2. Implement end-to-end feature engineering using industry-standard libraries to derive technical, statistical, and alternative features.
3. Support multiple analytical targets (e.g., next-period return prediction, directional classification, regime clustering, RL-based position sizing).
4. Versioned delivery roadmap allowing iterative enhancements: start with basic ingestion & storage, then add validation, feature generation, and multi-target capabilities.

## Stakeholders

- **Quantitative Researchers & Data Scientists**: Define feature sets, targets, and algorithms
- **Backend Engineers (Data Platform Team)**: Build ingestion, storage, and validation pipelines
- **DevOps/Infrastructure**: Provision and maintain cloud resources, scheduling, containerization
- **Product Manager**: Prioritize feature roadmap, coordinate between teams
- **QA & Compliance**: Ensure data quality, auditability, and regulatory compliance

## Technical Architecture & Components

### 1. Data Ingestion Layer

#### Sources & APIs

**Market Data (OHLCV)**
- Source: Kite Connect REST/WebSocket endpoints
- Frequency: Tick-level (optional), 1 min, 5 min bars

**Order Book Data (Level II)**
- Source: Kite Connect WebSocket (depth-of-book updates)
- Frequency: Real-time (as pushed by Kite)

**Fundamental Data**
- Source: Screener API (e.g., quarterly financials, ratios, corporate actions)
- Frequency: Batch (daily/weekly updates)

#### Ingestion Infrastructure

**Batch Framework**
- Apache Airflow or Prefect to schedule daily/nightly pulls of historical OHLCV and fundamental snapshots
- PySpark jobs (Spark SQL) to transform raw JSON/CSV into standardized Parquet partitions

**Real-Time/Streaming**
- A lightweight Python service (e.g., using kiteconnect library) subscribes to WebSocket streams for order book & tick data
- Data is pushed into Kafka or AWS Kinesis for immediate downstream processing (if real-time features are required)

#### Requirements

- Authenticate with Kite Connect (API key/secret, access token refresh)
- Gracefully handle API rate limits & reconnect logic (exponential backoff)
- Log ingestion success/failure, batch run times, and record counts

### 2. Data Storage Layer

#### Time-Series Storage (Parquet + PySpark)

**Raw Parquet**
- Landing zone: Raw JSON/CSV to HDFS/S3-like object store
- "Bronze" tables: Unprocessed OHLCV and order book data in Parquet (partitioned by date/instrument)

**Cleaned Parquet (Silver/Gold)**
- Silver: After missing-value imputation & basic validation, store cleaned Parquet tables
- Gold: Enriched time-series (merged OHLCV + order book + adjusted for splits/dividends)

**Schema Considerations**
- instrument_token, timestamp, open, high, low, close, volume, plus depth fields (bid_price_1, bid_qty_1, ask_price_1, ask_qty_1, …)
- Parquet partitioning by year, month, day, and optionally instrument_token for efficient queries

#### Alternative Data Storage

**PostgreSQL (Relational)**
- Fundamental tables (quarterly/annual financials, ratios, corporate actions)
- Index on symbol or instrument_token, and report_date

**MongoDB (NoSQL / Document)**
- Semi-structured alternative data (news sentiment JSON, social media feeds, macroeconomic releases)
- Collections: news_sentiment, social_media_sentiment, macroeconomic_indicators
- Document schema allows for flexible fields (e.g., nested sentiment scores, source metadata)

### 3. Data Validation & Missing Data Handling

#### Manual Missing-Data Handling

**Detection**
- For each time-series table, run a PySpark validation job to detect:
  - Missing timestamps (gaps in continuous series)
  - Null or zero volumes at unusual times
  - Outliers (e.g., price spikes/drops > Xσ from rolling mean)
- Log anomalies in a separate "Data Quality Report" (Parquet or relational table)

**Imputation Strategies (Manual)**
- For short gaps (< n minutes): Forward fill (ffill) and backward fill (bfill) in PySpark DataFrame
- For longer gaps or missing days: Mark as "market holiday" if consistent; otherwise, fill with nearest available closing price or drop if outside tolerance
- Document assumptions and keep a record of imputations (audit trail)

#### Preliminary Validation Checks

**Schema Compliance**
- Ensure each data source ingested matches the expected schema (data types, column count)

**Basic Sanity Checks**
- OHLC: high ≥ max(open, close), low ≤ min(open, close)
- Volume: Non-negative, non-zero for trading hours
- Order Book: Bid price ≤ ask price

**Automated Validation Pipelines**
- Use Great Expectations (or a custom PySpark validation library) for rule-based checks
- Schedule validation jobs immediately after ingestion; send alerts (email/Slack) on failures

### 4. Feature Engineering Layer

#### Primary Libraries & Tools

- **pandas & PySpark DataFrame**: Core transformations, rolling windows
- **NumPy**: Efficient numeric operations
- **TA-Lib**: Standard technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- **pandas_ta**: Additional or customized technical indicators
- **tsfresh**: Automated extraction of time-series features
- **Featuretools**: Automated "Deep Feature Synthesis" for relational data
- **scikit-learn**: Standard scaling, PCA, clustering, feature selection
- **PyOD**: Outlier detection features for regime identification
- **NLTK / spaCy**: For text cleaning if news/sentiment data is pulled
- **Transformers (Hugging Face)**: Fine-tune BERT or FinBERT models
- **NetworkX (optional)**: Compute graph-based features

#### Feature Categories & Examples

**Technical Features (Bar-Level)**
- Moving Averages: SMA_5, SMA_10, EMA_12, EMA_26
- Momentum: RSI_14, MACD (12,26,9), Stochastic Oscillator %K/%D
- Volatility: Bollinger_Band_Width, ATR_14
- Volume: OBV, VWAP, volume_zscore

**Order-Book Microstructure Features**
- Bid‐ask spread: ask_1_price − bid_1_price
- Order pressure: (sum(bid_qty_1..n) − sum(ask_qty_1..n)) / (sum(bid_qty_1..n) + sum(ask_qty_1..n))
- Depth imbalance over rolling windows
- Book slope: Linear regression of price vs. cumulative quantity

**Statistical/Time-Series Features**
- Rolling returns: log(close_t / close_{t−n}) for n ∈ {1,5,15,30,60 min}
- Rolling volatility: rolling_std(log returns) over 15/30/60/240 min
- Correlation/Covariance: rolling correlation between instruments
- PCA features: Principal component scores on multivariate price series
- tsfresh features: time-series entropy, sample entropy, autocorrelation

**Fundamental Features**
- Valuation Ratios: P/E, P/B, EV/EBITDA
- Profitability: ROE, ROA, Net Profit Margin
- Growth: YoY revenue growth, QoQ EBITDA growth
- Liquidity: Current ratio, quick ratio
- Leverage: Debt/Equity, Interest Coverage

**Alternative/NLP-Based Features**
- Sentiment Score: FinBERT or custom Transformer output
- Volume of Mentions: Count of tweets/news articles per symbol
- Macroeconomic Surprise Index: (Actual − Consensus)/Consensus

**Portfolio/Risk Features**
- Rolling VaR (5% or 1%) of returns
- Rolling portfolio beta
- Drawdown metrics: max drawdown over 1-day, 5-day, 20-day

#### Implementation Considerations

- Implement feature extraction in PySpark UDFs or vectorized pandas calls
- Cache intermediate features to avoid recomputation
- Maintain a centralized "Feature Registry"
- Version features alongside code (Git tags)

### 5. Target Generation Layer

#### Target Types

**Regression Targets**
- Next-Period Return: $\text{ret}{t+1} = \ln\bigl(\frac{\text{close}{t+1}}{\text{close}_t}\bigr)$
- Log-Scaled Movement: $\ln(\text{high}{t+1}/\text{low}{t+1})$

**Classification Targets**
- Direction: Binary (Up/Down) if $\text{close}_{t+δ} − \text{close}_t > θ$
- Volatility Regime: Low/Medium/High
- Spread Explosion: Binary indicator if bid-ask spread widens beyond μ + 2σ

**Clustering/Unsupervised Labels**
- Regime Clusters: HMM or KMeans on rolling returns & volatility
- Asset Grouping: Cluster instruments by correlation profile

**Reinforcement Learning Environment & Rewards**
- State: Vector of features at time t
- Action Space: {-1, 0, +1} or continuous [-1,1]
- Reward: Realized PnL minus transaction cost
- Episode Definition: Standard trading day

#### Storage of Targets

Save targets alongside features in "Gold" Parquet tables with unified schema:
```
instrument_token | timestamp | feature_vector_id | reg_ret_1min | dir_label_1min | vol_regime_1hour | cluster_state | rl_reward_t+1
```

### 6. Backtesting & Performance Analytics

#### Backtesting Engine
- Lightweight Python backtester consuming Parquet feature/target tables
- Simple strategies simulation
- Trading cost model (fixed bps slippage + commission)

#### Performance Metrics
- Gross & Net PnL, Sharpe ratio, max drawdown, win rate, profit factor
- Rolling performance (monthly, quarterly)
- Confusion matrix and classification report

#### Walk-Forward Analysis
- Split historical data into train/valid/test windows
- Automate rolling forward the window
- Record out-of-sample metrics

#### Reporting
- Generate PDF/HTML reports with Matplotlib charts
- Store reports in S3 bucket or shared network drive

### 7. Monitoring, Deployment & CI/CD

#### Git Repository Structure
```
├── ingestion/         # PySpark jobs & streaming scripts  
├── validation/        # Great Expectations suites or custom PySpark checks  
├── feature_engineering/  # Python modules defining each feature group  
├── target_generation/   # Scripts to compute regression/classification/cluster/RL labels  
├── backtesting/         # Backtesting harness & reporting templates  
├── config/              # YAML/JSON configuration for environments, API keys (not checked in)  
├── tests/               # PyTest suites for unit tests (schema checks, feature correctness)  
├── Dockerfile           # Container specification for reproducible environments  
└── .github/workflows/   # CI pipelines (linting, unit tests, integration tests)  
```

#### Unit & Integration Tests
- Validate Parquet schemas, column data types, expected row counts
- Test feature generation logic on synthetic data
- Validate target generation logic

#### Deployment Environments

**Dev**
- Small dataset (1–2 instruments, 1 month history)
- Local PySpark cluster (or single-node)

**Staging/QA**
- Full dataset for 3–6 months
- AWS EMR / Databricks for PySpark
- PostgreSQL/MongoDB replicated

**Prod**
- Full historical data (5+ years)
- Nightly ingestion on EMR/Spark cluster
- Real-time streaming on Kubernetes pods
- PostgreSQL & MongoDB clustered instances

#### Secrets Management
- Use HashiCorp Vault or AWS Secrets Manager
- Access secrets at runtime via environment variables or IAM roles

#### Scheduler & Orchestration
- Apache Airflow or Prefect for batch jobs
- Kubernetes CronJobs for scheduled tasks in prod

## Versioned Roadmap & Incremental Deliverables

### Version 1.0 — Core Ingestion & Storage

**Release Criteria:**
- Market Data Ingestion
- Order Book Ingestion
- Fundamental Data Ingestion
- Storage Infrastructure
- Data Catalog & Metadata

**Major Deliverables:**
- Documentation: "How to configure credentials and run ingestion scripts"
- Sample Parquet files for OHLCV and order book for at least 30 days
- PostgreSQL schema for fundamentals table
- MongoDB collections created for alternative data (empty)

### Version 1.1 — Validation & Missing-Data Handling

**Release Criteria:**
- Preliminary Data Validation Pipelines
- Missing Data Handling
- Order Book Validation

**Major Deliverables:**
- Data validation framework code
- Imputation scripts applied to OHLCV & order book
- Updated documentation: "Data validation rules & missing-data handling procedures"

### Version 1.2 — Base Feature Engineering

**Release Criteria:**
- Technical Feature Modules
- Order Book Features
- Fundamental Feature Join
- Feature Registry & Documentation

**Major Deliverables:**
- Feature engineering modules
- Parquet table "features_v1"
- Documentation: "Feature definitions and usage"

### Version 1.3 — Statistical & Alternative Feature Expansion

**Release Criteria:**
- tsfresh Integration
- NLP Sentiment Prototype
- Featuretools Integration

**Major Deliverables:**
- Parquet table "features_v1_3"
- Scripts for tsfresh extraction
- Scripts for sentiment score generation
- Featuretools workflow notebook

### Version 2.0 — Multi-Target Generation & Basic Modeling

**Release Criteria:**
- Regression & Classification Targets
- Clustering/Unsupervised Labels
- RL Environment Skeleton
- Basic Modeling & Validation

**Major Deliverables:**
- Parquet table "targets_v2"
- Gym environment code
- Baseline model training notebooks
- Documentation: "Target definitions, metrics, and baseline modeling results"

### Version 2.1 — Extended Backtesting & Risk Analysis

**Release Criteria:**
- Enhanced Backtesting Engine
- Walk-Forward & Stress Testing
- Performance Attribution & Reporting
- CI/CD for Backtesting

**Major Deliverables:**
- Backtesting engine code
- Sample backtest reports
- Risk analysis scripts
- Documentation: "Backtesting methodology, assumptions, and caveats"

### Version 3.0 — Automated Model Retraining & Deployment

**Release Criteria:**
- Automated Retraining Pipeline
- Model Serving & Inference
- Real-Time Feature Pipeline
- RL Policy Deployment (Proof of Concept)

**Major Deliverables:**
- Airflow DAG: retrain_models_monthly
- MLflow registry with versioned models
- Docker image for inference service
- Spark Structured Streaming job

### Version 3.1 — Live Paper-Trading Integration & Monitoring

**Release Criteria:**
- Broker Integration & Order Management
- Monitoring & Alerting Dashboard
- Security & Compliance Checks

**Major Deliverables:**
- Paper-trading integration code
- Grafana dashboards definitions
- Alert rules
- Documentation: "Paper-trade setup, reconciliation, and monitoring guidelines"

