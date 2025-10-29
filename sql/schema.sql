PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reviews (
  review_id   TEXT PRIMARY KEY,
  date        TEXT,
  product     TEXT,
  stars       INTEGER,
  text        TEXT NOT NULL,
  label       TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
  review_id   TEXT NOT NULL,
  predicted   TEXT NOT NULL,
  prob_pos    REAL NOT NULL,
  prob_neg    REAL NOT NULL,
  created_at  TEXT NOT NULL,
  PRIMARY KEY (review_id)
);

CREATE VIEW IF NOT EXISTS v_sentiment_summary AS
SELECT
  COALESCE(label, predicted) AS sentiment,
  COUNT(*) AS n
FROM (
  SELECT r.review_id, r.label, p.predicted
  FROM reviews r
  LEFT JOIN predictions p ON p.review_id = r.review_id
)
GROUP BY COALESCE(label, predicted);
