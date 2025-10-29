# Customer Review Sentiment Intelligence Platform (NLP + Streamlit + SQL)

The **Customer Review Sentiment Intelligence Platform** is a production-ready analytics application that combines **Natural Language Processing (NLP)**, **SQL**, and **interactive dashboards** to deliver actionable insights on customer feedback trends.  
Designed for content, marketing, and operations teams, it provides **data-driven sentiment intelligence** to enhance decision-making, brand perception tracking, and product quality analysis.

---

## Executive Summary

Organizations generate thousands of customer reviews daily, yet most remain underutilized.  
This solution bridges that gap, automatically classifying reviews as *positive* or *negative*, quantifying sentiment confidence, and visualizing feedback patterns over time.

**Business Impact:**
- Optimize product and service strategies based on real-time sentiment signals.
- Identify pain points and satisfaction drivers per product category.
- Streamline performance reporting with ready-to-present dashboards.
- Enable faster decision-making through AI-powered review analysis.

---

## System Architecture

```text
                 +------------------+
                 |  Raw Review CSV  |
                 +------------------+
                          |
                          v
                 +------------------+
                 |  Data Ingestion  |
                 | (ETL via SQLite) |
                 +------------------+
                          |
                          v
                 +------------------+
                 |  NLP Processing  |
                 |  (TF-IDF + LR)   |
                 +------------------+
                          |
                          v
                +---------------------+
                |    Model Storage    |
                | (Joblib Artifacts)  |
                +---------------------+
                          |
                          v
                +---------------------+
                |    Streamlit UI     |
                | Real-time Analytics |
                +---------------------+
```

---

## Repository Structure

```
customer-sentiment-intelligence/
├── app/
│   └── streamlit_app.py
├── data/
│   └── reviews.db
├── models/
│   ├── model.joblib
│   └── vectorizer.joblib
├── src/
│   ├── etl_loader.py
│   ├── preprocess.py
│   └── train_model.py
└── requirements.txt
```

---

## Core Capabilities

- **Automated Sentiment Detection** | Real-time text classification using TF-IDF + Logistic Regression.  
- **Interactive Review Exploration** | Filter and visualize feedback by product, time, or rating.  
- **Confidence-Based Scoring** | Probability-weighted results for transparent interpretation.  
- **Integrated SQL Backend** | All processed reviews are persisted in SQLite for auditability.  
- **Scalable Architecture** | Modular design ready for deployment to cloud or Docker environments.  

---

## Technical Overview

| Layer | Description |
|-------|--------------|
| **Data Source** | CSV or API-based customer reviews |
| **ETL Process** | Data normalization, cleaning, and SQL ingestion |
| **Feature Engineering** | TF-IDF vectorization |
| **Modeling** | Logistic Regression (binary sentiment) |
| **Visualization** | Streamlit UI + Plotly charts |
| **Persistence** | SQLite database with labeled review storage |

---

## Visual Overview

### User Interface  
<img width="1114" height="334" alt="Screenshot 2025-10-28 at 12-43-23 Review Sentiment Analyzer" src="https://github.com/user-attachments/assets/6df47ad3-70e4-44f1-8b80-f723021d7457" />

---

### Review Analytics Explorer  
<img width="1088" height="553" alt="Screenshot 2025-10-28 at 12-43-43 Review Sentiment Analyzer" src="https://github.com/user-attachments/assets/03b6e7cb-269c-4d57-95af-71745a17595e" />

---

### Sentiment Probability Distribution  
<img width="1028" height="395" alt="Screenshot 2025-10-28 at 12-43-53 Review Sentiment Analyzer" src="https://github.com/user-attachments/assets/922b886a-ae02-4192-a383-81a5ebafa038" />

---

### Review Results Table  
<img width="1039" height="465" alt="Screenshot 2025-10-28 at 12-44-08 Review Sentiment Analyzer" src="https://github.com/user-attachments/assets/ef21d1de-476c-4550-9fd9-3718deadb196" />

---

## Deployment Guide

### Local Setup
```bash
git clone https://github.com/yourusername/customer-sentiment-intelligence.git
cd customer-sentiment-intelligence

python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (macOS/Linux)

pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

### Cloud Deployment (Optional)
- Package with **Docker** and deploy via **Streamlit Cloud**, **Render**, or **Azure Web Apps**.  
- For enterprise environments, integrate SQLite → PostgreSQL → Power BI pipeline for advanced analytics.

---

## Data Flow Summary

1. **Ingest Data:** Upload or connect to raw review sources (CSV or API).  
2. **Clean Text:** Tokenization, stopword removal, lemmatization.  
3. **Model Application:** TF-IDF transforms text; logistic regression predicts sentiment.  
4. **SQL Storage:** Save predictions for traceability.  
5. **Visualization:** Streamlit renders metrics, histograms, and review tables.

---

## Example Insights

- 67% of reviews show **positive sentiment**, clustered at **0.8+ probability**.  
- 33% are **negative**, primarily related to logistics and product usability.  
- High-confidence classifications indicate strong model performance.  
- Balanced feedback supports credible brand engagement insights.

---

## Governance & Compliance

- Follows **PEP8** coding standards.  
- Model artifacts tracked via reproducible pipelines.  
- SQLite ensures full audit trail for all predictions.  
- Easily extendable to comply with **GDPR** or internal data retention policies.

---

## Future Roadmap

- Introduce **Neutral sentiment** classification.  
- Add **Aspect-level sentiment** (e.g., “delivery speed”, “customer service”).  
- Enable **real-time feedback API** integration for live review analysis.  
- Extend with **topic clustering** and **keyword extraction**.
