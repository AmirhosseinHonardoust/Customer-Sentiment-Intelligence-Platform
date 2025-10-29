#!/usr/bin/env python3
from __future__ import annotations
import sqlite3, joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
from clean_text import clean_text

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'reviews.db'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'pipeline.joblib'

st.set_page_config(page_title='Review Sentiment Analyzer', page_icon='💬', layout='wide')
st.title('💬 Customer Review Sentiment Analyzer')

CFG = {'displayModeBar': True, 'responsive': True}

@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('SELECT * FROM reviews', con, parse_dates=['date'])
    finally:
        con.close()
    return df

@st.cache_resource(show_spinner=False)
def load_model() -> Pipeline | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict_texts(pipe: Pipeline, texts: list[str]) -> pd.DataFrame:
    cleaned = [clean_text(t) for t in texts]
    proba = pipe.predict_proba(cleaned)[:,1]
    return pd.DataFrame({'text': texts, 'prob_pos': proba, 'prob_neg': 1-proba,
                         'predicted': ['POS' if p>=0.5 else 'NEG' for p in proba]})

df = load_reviews()
pipe = load_model()

# Sidebar upload
with st.sidebar:
    st.header('Data')
    uploaded = st.file_uploader('Upload reviews CSV', type=['csv'])
    if uploaded is not None:
        tmp = pd.read_csv(uploaded, dtype={'review_id':str, 'product':str, 'text':str, 'label':str})
        st.dataframe(tmp.head(10), hide_index=True)
        if st.button('Ingest into SQLite'):
            con = sqlite3.connect(DB_PATH)
            con.executescript((Path(__file__).resolve().parents[1] / 'sql' / 'schema.sql').read_text(encoding='utf-8'))
            try:
                for _, r in tmp.iterrows():
                    con.execute('DELETE FROM reviews WHERE review_id=?', (str(r['review_id']),))
                    con.execute('INSERT INTO reviews(review_id,date,product,stars,text,label) VALUES(?,?,?,?,?,?)',
                                (str(r['review_id']), r.get('date'), r.get('product'),
                                 int(r['stars']) if pd.notna(r.get('stars')) else None,
                                 str(r['text']), r.get('label')))
                con.commit()
                st.success(f'Ingested {len(tmp)} rows.')
            finally:
                con.close()
            st.cache_data.clear()
            df = load_reviews()

st.subheader('Try it: classify a single review')
colA, colB = st.columns([3,1])
with colA:
    user_text = st.text_area('Paste a review', height=150, placeholder='Example: The battery lasts all day and the camera is fantastic!')
with colB:
    if pipe is None:
        st.warning('Model not trained. Ingest labeled data and run training (see README).')
    else:
        if st.button('Analyze'):
            res = predict_texts(pipe, [user_text])
            row = res.iloc[0]
            st.metric('Prediction', row['predicted'])
            st.progress(float(row['prob_pos']) if row['predicted']=='POS' else float(row['prob_neg']),
                        text=f"Positive probability: {row['prob_pos']:.1%}")

st.divider()
st.subheader('Dataset Explorer')
if df.empty:
    st.info('No reviews in database yet. Upload a CSV in the sidebar or use the CLI.', icon='ℹ️')
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        products = ['All'] + sorted([p for p in df['product'].dropna().unique().tolist()])
        product = st.selectbox('Product', products)
    with c2:
        min_d = df['date'].min() if 'date' in df.columns and not df['date'].isna().all() else None
        max_d = df['date'].max() if 'date' in df.columns and not df['date'].isna().all() else None
        if min_d is not None and max_d is not None:
            d_range = st.date_input('Date range', (min_d, max_d), min_value=min_d, max_value=max_d)
        else:
            d_range = None
    with c3:
        q = st.text_input('Search text', '')

    mask = pd.Series([True]*len(df))
    if product != 'All':
        mask &= df['product'].fillna('') == product
    if d_range is not None:
        mask &= df['date'].between(pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1]))
    if q.strip():
        mask &= df['text'].str.contains(q, case=False, na=False)

    view = df.loc[mask].copy()
    st.write(f"{len(view):,} reviews match the filters.")

    if pipe is not None and not view.empty:
        preds = predict_texts(pipe, view['text'].tolist())
        view = view.join(preds[['predicted','prob_pos','prob_neg']])
        dist = view['predicted'].value_counts().rename_axis('sentiment').reset_index(name='n')
        fig = px.bar(dist, x='sentiment', y='n', title='Predicted Sentiment Distribution')
        st.plotly_chart(fig, config=CFG)

        fig2 = px.histogram(view, x='prob_pos', nbins=20, title='Histogram of Positive Probability')
        fig2.update_xaxes(title='P(Positive)')
        st.plotly_chart(fig2, config=CFG)

    st.dataframe(view[['review_id','date','product','stars','text'] + (['predicted','prob_pos'] if 'predicted' in view else [])], hide_index=True)

st.caption('Built with TF‑IDF + Logistic Regression • SQLite • Streamlit')
