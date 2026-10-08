"""How much of live headline vocabulary does the T1.4 TF-IDF vectorizer know?"""
import argparse
import collections
import sqlite3
from datetime import datetime, timedelta, timezone

import joblib

from newsmood.preprocessing import clean_for_tfidf as CLEAN 

p = argparse.ArgumentParser()
p.add_argument("--db", required=True)
p.add_argument("--vectorizer", required=True)
p.add_argument("--days", type=int, default=7)
p.add_argument("--top", type=int, default=20)
a = p.parse_args()

obj = joblib.load(a.vectorizer)
if hasattr(obj, "named_steps"):                             # saved as a Pipeline
    obj = next(s for s in obj.named_steps.values() if hasattr(s, "vocabulary_"))
analyzer, vocab = obj.build_analyzer(), obj.vocabulary_

since = (datetime.now(timezone.utc) - timedelta(days=a.days)).strftime("%Y-%m-%d")
rows = sqlite3.connect(a.db).execute(
    "SELECT title FROM headlines WHERE published_at >= ?", (since,)).fetchall()

total, missing, all_oov = 0, collections.Counter(), 0
for (title,) in rows:
    text = CLEAN(title) if CLEAN else title
    toks = [t for t in analyzer(text) if " " not in t]       # unigrams only
    miss = [t for t in toks if t not in vocab]
    total += len(toks)
    missing.update(miss)
    all_oov += bool(toks) and len(miss) == len(toks)

types = {t for (title,) in rows
         for t in analyzer(CLEAN(title) if CLEAN else title) if " " not in t}
print(f"headlines since {since}: {len(rows)}")
print(f"token OOV: {sum(missing.values())}/{total} = {sum(missing.values()) / max(total, 1):.1%}")
print(f"type OOV:  {len(missing)}/{len(types)} = {len(missing) / max(len(types), 1):.1%}")
print(f"headlines with every token unknown: {all_oov}")
print(f"top {a.top} missing:", ", ".join(f"{t}({c})" for t, c in missing.most_common(a.top)))