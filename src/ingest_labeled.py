#!/usr/bin/env python3
from __future__ import annotations
import argparse, sqlite3, pandas as pd
from pathlib import Path

def ensure_schema(db: Path, schema: Path) -> None:
    con = sqlite3.connect(db)
    try:
        con.executescript(schema.read_text(encoding='utf-8'))
    finally:
        con.close()

def upsert(df: pd.DataFrame, db: Path) -> int:
    con = sqlite3.connect(db)
    con.execute('PRAGMA foreign_keys = ON;')
    try:
        for _, r in df.iterrows():
            con.execute('DELETE FROM reviews WHERE review_id=?', (str(r['review_id']),))
            con.execute('INSERT INTO reviews(review_id,date,product,stars,text,label) VALUES(?,?,?,?,?,?)',
                        (str(r['review_id']), r.get('date'), r.get('product'),
                         int(r['stars']) if pd.notna(r.get('stars')) else None,
                         str(r['text']), r.get('label')))
        con.commit()
    finally:
        con.close()
    return len(df)

def main():
    ap = argparse.ArgumentParser(description='Ingest labeled reviews CSV into SQLite.')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--db', default='data/reviews.db')
    ap.add_argument('--schema', default='sql/schema.sql')
    args = ap.parse_args()

    db, schema = Path(args.db), Path(args.schema)
    db.parent.mkdir(parents=True, exist_ok=True)
    ensure_schema(db, schema)

    df = pd.read_csv(args.csv, dtype={'review_id':str, 'product':str, 'text':str, 'label':str})
    n = upsert(df, db)
    print(f'✅ Ingested {n} reviews into {db}')

if __name__ == '__main__':
    main()
