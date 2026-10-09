# 🚀 DriftWatch: Real-Time Distributed Concept Drift Detection Engine

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C.svg?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-7.4-231F20.svg?logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop%20HDFS-3.2-66CCFF.svg?logo=apachehadoop&logoColor=black)](https://hadoop.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX%20Runtime-1.17-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)

> **A production-grade, distributed streaming MLOps platform featuring dual-confirmation statistical drift detection (PSI + KS-Test), SHAP feature attribution drift tracking, sub-millisecond ONNX serving with zero-downtime hot-reloading, and an automated closed-loop retraining pipeline.**

---

## 📌 Executive Summary

Machine learning models deployed in production inevitably suffer from **silent performance degradation** caused by feature and concept drift. Traditional batch-monitoring solutions fail because they inspect model health hours or days after degradation has occurred.

**DriftWatch** solves this by establishing an end-to-end distributed streaming observability engine. It monitors real-time inference traffic emitted through Apache Kafka, performs dual-confirmation statistical drift analysis in PySpark Structured Streaming, computes explainable AI (SHAP) feature attribution shifts, and triggers **autonomous closed-loop retraining** when statistical degradation thresholds are breached.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │   Live Traffic Generator (03_kafka_producer.py)        │
       │   (Simulates: Normal Traffic -> Mild -> Severe Drift)  │
       └───────────────────────────┬────────────────────────────┘
                                   │  10-50 events/sec (JSON)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Apache Kafka Broker (Topic: raw-events)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       PySpark Structured Streaming Engine (Day 4/5)    │
       │                 (04_spark_psi_ks_shap.py)              │
       ├────────────────────────────────────────────────────────┤
       │  • Parquet Sink: Hadoop HDFS (/lake/raw_inferences/)   │
       │  • foreachBatch Statistical Engine:                    │
       │      ├─ Population Stability Index (PSI >= 0.25)       │
       │      ├─ 2-Sample Kolmogorov-Smirnov Test (p < 0.05)    │
       │      ├─ SHAP Attribution Drift (Top-5 features)        │
       │      └─ Real-Time F1 & Accuracy Ground-Truth Tracking  │
       └───────────────────────────┬────────────────────────────┘
                                   │
             Alert condition: Sustained PSI >= 0.25 & KS p < 0.05
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Automated Closed-Loop Retrainer (05_retrain_job.py)│
       ├────────────────────────────────────────────────────────┤
       │  1. Ingests drifted samples from HDFS Data Lake        │
       │  2. Retrains model with balanced class weighting       │
       │  3. Validates F1 recovery on drifted distribution      │
       │  4. Promotes v2 artifact & updates model_registry.json │
       │  5. Hot-reloads active ONNX runtime microservice       │
       └────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Engineering Highlights

### 1. Dual-Confirmation Drift Detection
To eliminate false alarms without missing true statistical decay, DriftWatch enforces a **dual-gate confirmation mechanism**:
- **Population Stability Index (PSI):** Measures macro shift in continuous feature distributions across 10 quantile bins (deciles). Threshold: $\text{PSI} \ge 0.25$.
- **Two-Sample Kolmogorov-Smirnov (KS) Test:** Non-parametric hypothesis test comparing the empirical cumulative distribution function (eCDF) of streaming batches against a baseline reference. Threshold: $p < 0.05$.
- **Alert Condition:** Requires both conditions to hold simultaneously across consecutive micro-batches.

### 2. Explainable AI (SHAP) Attribution Drift
Distribution shifts alone do not guarantee model failure. DriftWatch incorporates `TreeExplainer` attribution tracking:
- Profiles baseline feature importance ($|\text{SHAP}|$ values) on the top 5 predictive features.
- Computes real-time streaming SHAP attribution shifts to pinpoint which specific feature drives model decision divergence.

### 3. High-Performance ONNX Serving with Hot-Reloading
- Replaces vulnerable, slow Python `.pkl` deserialization with a **C++ ONNX Runtime engine** exposed via FastAPI.
- Achieves sub-millisecond inference latencies.
- Supports zero-downtime dynamic model swapping (`POST /reload`) upon automated worker promotion.

### 4. Closed-Loop Automated Retraining & Recovery
- When sustained drift is detected, the worker automatically ingests recent drifted inference records from the HDFS Parquet lake alongside historical baseline data.
- Retrains a candidate model (`v2`), verifies the **Closed-Loop V-Curve recovery** ($F_1$ recovery on drifted traffic), updates `models/model_registry.json`, and promotes the new artifact.

### 5. Multi-Dataset & Categorical Metric Support
- **Numerical Pipeline:** Credit Card Fraud Detection (28 continuous PCA features + Amount).
- **Categorical / Mixed Pipeline:** UCI Bank Marketing dataset with discrete PSI, Chi-Square Contingency ($\chi^2$), and Cramér's V tests.

---

## 📁 Repository Structure

```text
DriftWatch/
├── dataset/                           # Raw datasets (creditcard.csv, bank_marketing.csv)
├── models/
│   ├── baseline_model.onnx            # Serialized baseline ONNX model
│   ├── bank_model.onnx                # Bank marketing ONNX model
│   └── model_registry.json            # Versioned model lineage & metadata
├── profiles/
│   ├── baseline_profile.json          # Decile bin edges & expected distributions
│   ├── shap_baseline.json             # Baseline top-5 SHAP feature attributions
│   └── bank_marketing_baseline.json   # Categorical baseline profiles
├── scripts/
│   ├── drift_metrics_categorical.py   # Chi-square, Cramér's V, categorical PSI
│   └── export_credit_to_onnx.py       # ONNX export utility
├── logs/                              # Generated run logs & metrics
├── 01_train_model.py                  # Baseline training & benchmarking (Credit Card)
├── 01_train_categorical_model.py      # Baseline training (Bank Marketing)
├── 02_baseline_profiler.py            # SHAP & quantile profiling engine
├── 02_model_service.py                # FastAPI + ONNX Runtime serving microservice
├── 03_kafka_producer.py               # Streaming traffic generator with 3-phase drift
├── 04_spark_psi_ks_shap.py            # PySpark streaming drift detection engine
├── 05_retrain_job.py                  # Closed-loop automated retraining worker
├── compare_numerical_vs_categorical.py# Comparative benchmark suite
├── verify_closed_loop_recovery.py     # Quantitative F1 recovery validator
├── verify_drift_correlation.py        # Empirical drift vs accuracy analyzer
├── docker-compose.yml                 # Kafka, Zookeeper, HDFS cluster specification
└── requirements.txt                   # Project Python dependencies
```

---

## ⚡ Quickstart & Execution Guide

### 1. Prerequisites
- **OS:** Windows 10/11, Linux, or macOS
- **Python:** Python 3.11 (Recommended for PySpark & ONNX compatibility)
- **Docker Desktop:** Running (for Kafka & Hadoop HDFS containers)
- **Java:** OpenJDK 8, 11, or 17 (for Apache Spark)

---

### 2. Environment Setup

Open PowerShell or terminal in the project root:

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\activate

# 3. Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Launch Distributed Infrastructure

Start the Kafka, Zookeeper, and Hadoop HDFS cluster via Docker Compose:

```powershell
docker-compose up -d
```

Verify that all 4 containers are healthy:
- **Zookeeper:** `localhost:2181`
- **Kafka Broker:** `localhost:9092`
- **HDFS NameNode Web UI:** `http://localhost:9870`
- **HDFS IPC:** `hdfs://localhost:9000`

---

### 4. Step-by-Step Pipeline Execution

#### Step 4.1: Train Baseline Model & Initialize Registry
Trains the benchmark baseline model on `creditcard.csv`, evaluates test performance ($F_1 \approx 0.846$), and registers version `v1` in `models/model_registry.json`:

```powershell
python 01_train_model.py
```

#### Step 4.2: Generate Baseline Profiles & SHAP Rankings
Extracts the top 5 influential features via SHAP TreeExplainer, calculates quantile bin edges (deciles), and captures sample distributions for KS tests:

```powershell
python 02_baseline_profiler.py
```

*(Artifacts saved to `profiles/baseline_profile.json`, `profiles/shap_baseline.json`, and `profiles/baseline_samples.pkl`)*

---

#### Step 4.3: Start Real-Time ONNX Model Serving Service *(Terminal 1)*
Launches the high-performance inference microservice:

```powershell
python 02_model_service.py
```
*Health check available at: `http://localhost:8000/health`*

---

#### Step 4.4: Launch PySpark Streaming Drift Engine *(Terminal 2)*
Starts the distributed streaming engine consuming from Kafka, computing real-time PSI, KS-test, SHAP attribution drift, and micro-batch $F_1$ tracking:

```powershell
python 04_spark_psi_ks_shap.py
```

---

#### Step 4.5: Start Kafka Traffic Producer with Drift Simulation *(Terminal 3)*
Streams inference events at 10 events/sec through three progressive phases (**Normal** $\rightarrow$ **Mild Drift** $\rightarrow$ **Severe Drift**):

```powershell
python 03_kafka_producer.py --dataset creditcard --drift-mode synthetic --rate 10 --phase-seconds 60
```

---

#### Step 4.6: Closed-Loop Automated Retraining & Recovery
When sustained drift alerts trigger in Spark, run the retraining worker to ingest new drifted data from the lake, train `v2`, verify performance recovery, and hot-reload:

```powershell
python 05_retrain_job.py
```

---

### 5. Verification & Benchmark Reports

#### Verify Closed-Loop $F_1$ Recovery (The V-Curve):
```powershell
python verify_closed_loop_recovery.py
```

#### Analyze Drift vs Model Accuracy Correlation:
```powershell
python verify_drift_correlation.py
```

#### Run Comprehensive Benchmark (Pickle vs ONNX, Synthetic vs Natural Drift):
```powershell
python compare_numerical_vs_categorical.py
```

---

## 📊 Proven Results & Empirical Validation

```text
===========================================================================
CLOSED-LOOP V-CURVE PERFORMANCE PROOF:
---------------------------------------------------------------------------
1. Nominal Production (v1 baseline)     : F1 = 0.8457 | Precision = 0.9610
2. Under Severe Feature Drift (v1)      : F1 dropped to ~0.4200
3. Dual-Confirmation Streaming Alert   : Spark PSI >= 0.25 & KS p < 0.05
4. Automated Retraining (v2 promoted)   : F1 recovered to 0.7468 (+77.8% recovery)
===========================================================================
```

- **Inference Latency:** ONNX Runtime yields **~4.5x faster** inference per 1,000 requests compared to standard Python joblib/pickle deserialization.
- **Micro-Batch Processing:** Spark foreachBatch processes 300-event streaming micro-batches with full statistical dual confirmation and SHAP sampling in under **1.2 seconds**.

---

## 🚀 Fast-Track Execution & Showcase Options

### Option A: The Turnkey One-Click Launcher
Starts the ONNX inference microservice and Streamlit Observability Console simultaneously:
```powershell
.\start_driftwatch.ps1
```
*(Or double-click `start_driftwatch.bat` in Windows File Explorer)*

### Option B: The Zero-Docker Standalone Streaming Engine
Demonstrates the full distributed streaming and autonomous closed-loop retraining lifecycle without needing Docker:
```powershell
python demo_mode.py
```

### Option C: Automated System Health Verification
Runs the 6-step automated test suite verifying dataset, profiles, ONNX models, and drift mathematics:
```powershell
python test_pipeline.py
```

### Option D: Interactive Viva & Interview Defense Prep Guide
Launches the interactive defense CLI covering the Top 15 technical viva questions:
```powershell
python viva_prep.py
```
*(Or pass `python viva_prep.py all` to display all questions and answers at once)*

