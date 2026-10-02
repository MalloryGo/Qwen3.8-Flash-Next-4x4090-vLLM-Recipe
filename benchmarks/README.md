# Benchmarks

The public benchmark data is intentionally reduced to the measurements needed to
understand and reproduce the serving profile.

Machine-readable results: [results.json](results.json)

## Metric definitions

Single-request decode numbers use:

~~~text
derived decode tok/s = 1000 / median TPOT(ms)
~~~

This describes token-to-token decode cadence. It is not end-to-end request
throughput.

Concurrency numbers use:

~~~text
aggregate output throughput =
total generated tokens / request-wave elapsed time
~~~

Do not directly compare the concurrency throughput column with the single-request
decode column.

## Headline results

| Prompt | Median TTFT | Median TPOT | Decode |
|---:|---:|---:|---:|
| 128K | 10.88 s | 4.860 ms | 205.8 tok/s |
| 220K | 19.12 s | 4.837 ms | 206.7 tok/s |
| 255K | 22.31 s | 5.191 ms | 192.6 tok/s |
| 260K | 22.74 s | 4.837 ms | 206.8 tok/s |

A 261,888-token prompt plus 256 output tokens also completed at the exact
262,144-token sequence boundary.

| Workload | Success | Aggregate output |
|---|---:|---:|
| 2 × 128K | 2/2 | 38.42 tok/s |
| 3 × 80K | 3/3 | 53.80 tok/s |
| 4 × 60K | 4/4 | 75.81 tok/s |
| 100K/60K/40K/20K | 4/4 | 85.84 tok/s |
| 12 arrivals, max active 4 | 12/12 | 95.30 tok/s |

The recommended active inference concurrency is **4**. Larger arrival bursts should
be queued rather than made simultaneously resident.

## Privacy boundary

This repository does not publish the private benchmark prompt corpus, raw model
responses, tool schemas, logs, screenshots, images or videos used during testing.
Only allowlisted metrics and final correctness results are included.
