# Phase 3 results (best alpha per method)

Eval cap: 1024 samples/task. Datasets: MNIST, EuroSAT, DTD, GTSRB, SVHN.

| method | alpha | avg | MNIST | EuroSAT | DTD | GTSRB | SVHN |
|---|---|---|---|---|---|---|---|
| zero_shot | None | 48.65 | 52.25 | 51.27 | 44.92 | 42.68 | 52.15 |
| individual | None | 92.85 | 99.12 | 99.61 | 73.24 | 96.97 | 95.31 |
| task_arithmetic | 0.5 | 80.74 | 97.36 | 73.93 | 58.79 | 86.82 | 86.82 |
| ties | 0.5 | 71.91 | 93.65 | 68.85 | 51.17 | 70.21 | 75.68 |
| tsv_compress_sum | 0.5 | 80.37 | 97.36 | 73.54 | 58.20 | 86.13 | 86.62 |
| tsv_merge | 0.5 | 80.45 | 95.90 | 91.60 | 58.40 | 77.83 | 78.52 |
