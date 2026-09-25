# RSS sources

Written by T3.1 on 2026-09-24. Probed 2026-09-25 01:0x–01:28 UTC (Thu Sep 24 evening, US
time), two fetches per feed about 20 minutes apart, sequential with a 2 s
gap, a descriptive `User-Agent` and a 15 s timeout. Counts were identical
across both fetches for every feed.

Rerun the liveness check any time with `newsmood ingest --dry-run`.

## Verdict

| Feed | Responded | Entries per fetch | Status |
|---|---|---|---|
| `yahoo-finance` | 200, `application/xml`, ~0.1 s | 49 | **Live, kept, stale** (see below) |
| `cnbc-finance` | 200, `application/xml`, ~0.7 s | 30 | **Live, kept** |
| `marketwatch-top` | 200, `application/xml`, ~0.1 s | 10 | **Live, kept** |

All three candidates from guide T3.1 returned entries, so none were dropped.
All three parsed cleanly (`bozo = False`). The three feeds shared no
headlines: exact normalized-title overlap was 0 for each pair.

## Per feed

### yahoo-finance: `https://finance.yahoo.com/news/rssindex`

- **Fields:** `title`, `link`, `published`, `source` (the originating
  publisher), `id`, `media_content`. **There is no summary field at all.**
  `summary` is `""` for every Yahoo row.
- **Headline length:** 32–100 characters, median 81; 5–18 words, median 12.
  The longest title (100 characters) is complete, so the titles are not
  truncated.
- **Freshness is the problem.** The newest item was published
  2026-09-23 06:00 UTC, about 43 hours before the probe. The server's
  `Last-Modified` was 2026-09-24 12:21 UTC, so the file itself changes but its
  content lags. Of 49 items, 45 are from Sep 23. Four are older: one each
  from 2026-09-22, 2026-06-09, 2025-01-28 and 2024-11-20.
- **Aggregator skew.** Every link is on `finance.yahoo.com`, but the `source`
  field shows 29 of 49 items came from Insider Monkey. The next largest were
  The Daily Upside and Yahoo Personal Finance (4 each) and 24/7 Wall St. (3).
  Many titles are question-shaped analyst pieces ("…Can Profits Grow?")
  rather than event reports.
- **Why it stays:** it is the highest-volume feed and it does respond. It is
  stale, not dead. However, the old items mean ingest cannot assume
  "fetched today" is the same as "published today". Bucketing by
  `published_at` belongs to T3.3, and a freshness gate belongs to T3.5. If the
  lag holds across several days, T3.5 is where to decide whether to drop the
  feed.

### cnbc-finance: `https://www.cnbc.com/id/10000664/device/rss/rss.html`

- **Fields:** `title`, `link`, `summary`, `published`, `id`, plus CNBC
  `metadata:type` (all `cnbcnewsstory`), `metadata:id` and
  `metadata:sponsored` (all `false` in this sample).
- **Headline length:** 49–118 characters, median 81.5; 8–19 words, median 13.
- **Summary:** 66–168 characters, median 138.5, plain text.
- **Window:** the 30 items span 2026-09-11 to 2026-09-25, about 2 per day, so
  most of a fetch is items already seen. The feed is labelled "Finance" but
  also carries political items, e.g. "Here's who is attending the Trump-Xi
  state dinner".

### marketwatch-top: `https://feeds.content.dowjones.io/public/rss/mw_topstories`

- **Fields:** `title`, `link`, `summary`, `published`, `id`, `author`,
  `media_content`.
- **Headline length:** 57–112 characters, median 85; 10–22 words, median 14.5.
  This is the longest and most conversational of the three, with personal
  finance advice-column titles ("My friend grosses $300,000 a year…").
- **Summary:** 47–185 characters, median 103. Two of 10 summaries contain
  literal `&amp;` after parsing, because the source double-escapes them.
  Summaries never reach the classifier, so this is left as-is.
- **Window:** only 10 items, covering about 3 hours (20:20–23:35 UTC). A
  once-daily fetch sees only the last few hours of MarketWatch's day.
- **URL quirk:** every link carries `?mod=mw_rss_topstories`. The id is the
  sha256 of the URL as given, so it stays stable only while that parameter
  does. Title-based dedupe (T3.2) is the backstop.

## Consequences for later tasks

- **Daily volume is small and uneven.** One fetch yields about 89 rows, but
  only 16 were published on the probe's US date (Sep 24 Eastern): 6 from
  CNBC, 10 from MarketWatch and none from Yahoo. A single daily fetch gives the mood
  index a thin, Yahoo-dominated sample. T3.3's `n_headlines` floor and T3.5's
  gates need to allow for this. Fetching more than once a day is a T4.4
  scheduling question, not an ingest change.
- **`published_at` is present for 89/89 rows** (`published_parsed` on every
  entry), converted to UTC.
- **Titles are passed through untouched**, including curly quotes and `&`,
  which feedparser already unescapes.
