# RSS sources

Written by T3.1 on 2026-09-24. Probed 2026-09-25 01:0x–01:28 UTC (Thu Sep 24 evening, US
time), two fetches per feed about 20 minutes apart, sequential with a 2 s
gap, a descriptive `User-Agent` and a 15 s timeout. Counts were identical
across both fetches for every feed.

Rerun the liveness check any time with `newsmood ingest --dry-run`.

## Verdict

*Updated 2026-10-09 (T3.5): Yahoo removed, Investing.com added. See
"Changes since T3.1" below.*

| Feed | Responded | Entries per fetch | Status |
|---|---|---|---|
| `cnbc-finance` | 200, `application/xml`, ~0.7 s | 30 | **Live, kept** |
| `marketwatch-top` | 200, `application/xml`, ~0.1 s | 10 | **Live, kept** |
| `investing-stocks` | 200 | 10 | **Live, added 2026-10-09**, last in config order |
| `yahoo-finance` | HTTP 404 on Oct 7 and Oct 8 | — | **Removed 2026-10-09**; stored rows kept |

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

## Changes since T3.1

### yahoo-finance removed (2026-10-09)

Stale from Sep 23 (newest item never moved past it) and HTTP 404 on Oct 7
and Oct 8. Removed from `ingest.feeds`. Its rows already in the store stay:
nothing was deleted, and past days' index values keep them.

### investing-stocks added (2026-10-09): `https://www.investing.com/rss/news_25.rss`

Investing.com "Stock Market News". Probed with `newsmood ingest --dry-run`
(feed list overridden to this one feed via `NEWSMOOD_INGEST__FEEDS`) plus a
parse of the same body, three times on Fri Oct 9, 16 minutes from first to last:

| Fetch (UTC) | Entries | Newest item, as UTC | Oldest item | Templated | Overlap with stored titles |
|---|---|---|---|---|---|
| 13:57:52 | 10 | 13:40:45 (0.3 h old) | 13:18:13 | 0 | 0 |
| 13:59:59 | 10 | 13:40:45, identical set | 13:18:13 | 0 | 0 |
| 14:13:58 | 10, all new | 14:01:48 (0.2 h old) | 13:52:04 | 0 | 0 |

- **Timezone: UTC.** `pubDate` has no offset, e.g. `2026-10-09 13:36:22`.
  feedparser reads an offset-less date as UTC: `published_parsed` is
  `(2026, 10, 9, 13, 36, 22)` and `store._ts` gives
  `2026-10-09T13:36:22.000000+00:00`. That reading is right:
  - Every item in all three fetches is 0.2–0.7 h older than the fetch. Read as ET, every item
    would be about 4 h in the future. Read as Israel time (UTC+3, the
    company's home), the newest would be 3.3 h old.
  - Reuters' "US stocks open higher as oil slips; telecoms hit by SpaceX
    spectrum deal" is stamped 13:36:22: 9:36 ET as UTC, six minutes after the
    9:30 ET open. Investing.com's "U.S. stocks open higher…" is 13:37:59.
    Under UTC+3 both would land at 6:36 ET, before the market opened.
  - The linked article pages return HTTP 403 to a scripted fetch, so their
    displayed times could not be compared.

  `tests/test_feeds.py::test_offset_less_pubdate_is_read_as_utc` pins
  feedparser's behaviour, since a wrong offset moves headlines to the wrong
  ET day.
- **Templated headlines:** none matching "at close of trade" or "stocks
  higher/lower/mixed at close" in any fetch (0 of 10 each time), below the
  3-of-10 bar for an exclude list. Accepted: no `ingest.title_exclude_patterns`.
  The Oct 8 probe saw "Poland stocks lower at close of trade"; if those grow
  past 3 per fetch, add the list then.
- **Publishers:** `author` is present on every item: Investing.com 6 and
  Reuters 4 in fetch 1, Investing.com 8 and Reuters 2 in fetch 3. Links are
  all on `investing.com`.
- **Fields:** `title`, `link`, `published`, `author`. Every title and link
  present (0 skipped).
- **Window is short.** The 10 items span about 22 minutes. A once-daily
  ingest sees only the last half hour of Investing.com's day; T4.4's every
  4–6 h schedule will catch more.
- **Overlap:** no normalized title matched any stored CNBC or MarketWatch
  row. Some stories will overlap later (the same Reuters wire); config order
  puts this feed last, so CNBC or MarketWatch keeps a shared story.
- **Headline style:** event headlines ("Humana surges after topping 2027
  Medicare star ratings") plus Investing.com's own question format, "Why is
  X stock rallying / sliding today?": 2 of 10 in fetch 1, **7 of 10 in fetch
  3**. Each names one stock and links to its own article, so it is not a
  machine-written market close and isn't filtered. Watch it: if those titles
  dominate the feed, or T3.4 finds the model labels them by the verb alone,
  that's the case for an exclude pattern.

### Candidates rejected (2026-10-08)

| Candidate | Why not |
|---|---|
| Nasdaq Markets RSS | Newest item ~33 h old at the probe, and mostly Motley Fool / RTTNews syndication |
| Seeking Alpha | `robots.txt` disallows automated fetching |
| Fox Business | `robots.txt` disallows automated fetching |

