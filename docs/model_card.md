# Model card — `newsmood-distilbert-financial`

*The remaining sections (intended use, training data, metrics, limitations) come in T2.6.*

## Performance

Reproduce with:

```
newsmood score --file data/sample/headlines_200.jsonl --benchmark
```

| | |
|---|---|
| Hardware | Apple M3 Pro, CPU only (no GPU/MPS) |
| Software | torch 2.14.0, 5 threads |
| Settings | batch size 32, dynamic padding, `max_length` 96 |
| Input | 200 rows, 7 batches, one untimed warmup batch |
| Throughput | **164.9 headlines/s** (1.21 s for 200) |
| Batched latency, per headline | p50 6.09 ms · p95 6.72 ms |
| Cold latency | 3.05 s (median of 3, range 3.03–3.05 s) |

At this rate, 1,000 headlines take about 6 s on an M3 Pro CPU.

**Batched latency** is each batch's wall time divided by the number of rows in it. The percentiles rest on only 7 batches. The final batch has 8 rows instead of 32, so its fixed per-batch cost is shared by fewer headlines, and it probably sets the p95.

**Cold latency** is the wall time of a fresh `newsmood score --text` process with the model already in `~/.cache/newsmood/`. It includes interpreter start, importing torch and transformers, and loading the model. It is what a user waits for when they run the command once. Inference itself takes milliseconds of that time.

**Input caveat.** The 200 rows are held-out Financial PhraseBank sentences (mean about 30 tokens), not live headlines. Live RSS headlines are usually shorter, and with dynamic padding shorter batches cost less, so live throughput is likely higher, not lower.
