### Document scope, by question type

| config | question_type | n | Hit@5 | Recall@10 | MRR@100 |
|---|---|---|---|---|---|
| bm25_page | domain-relevant | 50 | 0.200 | 0.340 | 0.176 |
| bm25_page | metrics-generated | 50 | 0.020 | 0.010 | 0.023 |
| bm25_page | novel-generated | 50 | 0.540 | 0.660 | 0.431 |
| dense_page_truncate | domain-relevant | 50 | 0.340 | 0.470 | 0.282 |
| dense_page_truncate | metrics-generated | 50 | 0.740 | 0.730 | 0.433 |
| dense_page_truncate | novel-generated | 50 | 0.560 | 0.710 | 0.389 |
| dense_page_maxwindow | domain-relevant | 50 | 0.360 | 0.500 | 0.234 |
| dense_page_maxwindow | metrics-generated | 50 | 0.620 | 0.670 | 0.364 |
| dense_page_maxwindow | novel-generated | 50 | 0.600 | 0.840 | 0.500 |
