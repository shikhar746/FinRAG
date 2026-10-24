### Paired bootstrap: B minus A

10000 resamples of questions, seed 13. * = 95% CI excludes zero.

| A | B | scope | metric | mean(B-A) | 95% CI | n |
|---|---|---|---|---|---|---|
| bm25_page | dense_page_truncate | document | MRR@100 | +0.158 * | [+0.089, +0.228] | 150 |
| bm25_page | dense_page_truncate | document | Recall@5 | +0.260 * | [+0.167, +0.353] | 150 |
| bm25_page | dense_page_truncate | document | Recall@10 | +0.300 * | [+0.207, +0.393] | 150 |
| bm25_page | dense_page_truncate | corpus | MRR@100 | +0.117 * | [+0.064, +0.172] | 150 |
| bm25_page | dense_page_truncate | corpus | Recall@5 | +0.117 * | [+0.047, +0.187] | 150 |
| bm25_page | dense_page_truncate | corpus | Recall@10 | +0.203 * | [+0.123, +0.283] | 150 |
| bm25_page | dense_page_maxwindow | document | MRR@100 | +0.156 * | [+0.101, +0.213] | 150 |
| bm25_page | dense_page_maxwindow | document | Recall@5 | +0.243 * | [+0.163, +0.323] | 150 |
| bm25_page | dense_page_maxwindow | document | Recall@10 | +0.333 * | [+0.250, +0.420] | 150 |
| bm25_page | dense_page_maxwindow | corpus | MRR@100 | +0.077 * | [+0.042, +0.113] | 150 |
| bm25_page | dense_page_maxwindow | corpus | Recall@5 | +0.113 * | [+0.060, +0.170] | 150 |
| bm25_page | dense_page_maxwindow | corpus | Recall@10 | +0.180 * | [+0.107, +0.253] | 150 |
| dense_page_truncate | dense_page_maxwindow | document | MRR@100 | -0.002 | [-0.042, +0.039] | 150 |
| dense_page_truncate | dense_page_maxwindow | document | Recall@5 | -0.017 | [-0.073, +0.043] | 150 |
| dense_page_truncate | dense_page_maxwindow | document | Recall@10 | +0.033 | [-0.027, +0.093] | 150 |
| dense_page_truncate | dense_page_maxwindow | corpus | MRR@100 | -0.040 * | [-0.077, -0.005] | 150 |
| dense_page_truncate | dense_page_maxwindow | corpus | Recall@5 | -0.003 | [-0.050, +0.043] | 150 |
| dense_page_truncate | dense_page_maxwindow | corpus | Recall@10 | -0.023 | [-0.083, +0.037] | 150 |
