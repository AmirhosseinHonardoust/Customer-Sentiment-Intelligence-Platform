#!/usr/bin/env python3
from __future__ import annotations
import argparse, sqlite3, joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from clean_text import clean_text

def load_labeled(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query('SELECT review_id, text, label FROM reviews WHERE label IS NOT NULL', con)
    finally:
        con.close()
    df = df.dropna(subset=['text','label']).copy()
    df['text'] = df['text'].map(clean_text)
    df['label'] = df['label'].str.upper().map(lambda x: 'POS' if x.startswith('P') else 'NEG')
    return df

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)),
        ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
    ])

def main():
    ap = argparse.ArgumentParser(description='Train TF-IDF + Logistic Regression sentiment classifier.')
    ap.add_argument('--db', default='data/reviews.db')
    ap.add_argument('--out', default='models/pipeline.joblib')
    args = ap.parse_args()

    df = load_labeled(Path(args.db))
    if df.empty:
        raise SystemExit('No labeled data found in DB. Ingest a labeled CSV first.')

    X = df['text'].values
    y = (df['label'] == 'POS').astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline().fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:,1]
    preds = (proba >= 0.5).astype(int)
    print(classification_report(yte, preds, target_names=['NEG','POS']))
    try:
        print('AUC:', roc_auc_score(yte, proba))
    except Exception:
        pass
    print('Confusion matrix:\n', confusion_matrix(yte, preds))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f'✅ Saved pipeline → {args.out}')

if __name__ == '__main__':
    main()
