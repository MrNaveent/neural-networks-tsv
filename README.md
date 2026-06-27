# Task Singular Vectors for Model Merging

**Course:** Neural Networks 2024/2025 — Prof. Simone Scardapane (Sapienza University of Rome)  
**Author:** Naveen Tiwari (Student ID: 2261826) — solo project  
**Paper i implemented:** Gargiulo et al., *Task Singular Vectors: Reducing Task Interference in Model Merging*

---

## What this project is about

The idea behind this project is model merging — basically trying to combine multiple fine-tuned models into one model that can do all the tasks at once, without having to retrain everything from scratch.

The problem is that when you just average the weights of different fine-tuned models, the updates from different tasks end up conflicting with each other and hurting the final performance. This is called task interference.

I implemented two methods from the TSV paper to deal with this:

- **TSV Compress** — uses truncated SVD to compress each task vector down to about 10% of its original size, with barely any accuracy loss
- **TSV Merge** — decorrelates the task subspaces using something called Procrustes whitening before merging them, so the conflicting updates don't cancel each other out

I also compared these against two baselines: Task Arithmetic (just summing the task vectors) and TIES-Merging (trims small values, elects a sign per parameter, averages only the agreeing tasks).



## How to run this

First set up the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then the pipeline goes in this order:

```bash
# 1. save the pretrained CLIP image encoder 
python scripts/save_pretrained.py

# 2. fine-tune on each dataset 
python scripts/finetune_one.py --dataset MNIST
python scripts/finetune_one.py --dataset EuroSAT
python scripts/finetune_one.py --dataset DTD
python scripts/finetune_one.py --dataset GTSRB
python scripts/finetune_one.py --dataset SVHN

# 3. quick sanity check on zero-shot accuracy before merging
python scripts/zero_shot_sanity.py

# 4. main evaluation — runs all merge methods and saves results CSVs
python scripts/evaluate_merges.py

# 5. ablation — compression ratio sweep
python scripts/ablation_rank_sweep.py

# 6. ablation — per-layer analysis and interference heatmap
python scripts/ablation_layer_analysis.py

# 7. generate all figures
python scripts/plot_all.py
```

The figures end up in `results/figures/` and the CSVs in `results/`.

---

## Project structure

```
NN-Project/
├── src/                  
├── scripts/              
├── notebooks/
│   └── final_report.ipynb   
├── results/
│   ├── figures/         
│   └── *.csv            
├── checkpoints/vitb16/  
├── data/                 
└── requirements.txt
```

---

## Results summary

| Method | Avg | MNIST | EuroSAT | DTD | GTSRB | SVHN |
|---|---|---|---|---|---|---|
| Zero-shot | 48.7% | 52.3 | 51.3 | 44.9 | 42.7 | 52.2 |
| Individual | 92.9% | 99.1 | 99.6 | 73.2 | 97.0 | 95.3 |
| Task Arithmetic | 80.7% | 97.4 | 73.9 | 58.8 | 86.8 | 86.8 |
| TIES-Merging | 71.9% | 93.7 | 68.9 | 51.2 | 70.2 | 75.7 |
| TSV-Compress | 80.4% | 97.4 | 73.5 | 58.2 | 86.1 | 86.6 |
| **TSV-Merge** | **80.5%** | 95.9 | **91.6** | 58.4 | 77.8 | 78.5 |

All numbers at alpha=0.5, evaluated on 1,024 test samples per task.

---

## References

- Gargiulo et al., 2024 — Task Singular Vectors (the paper i implemented)
- Ilharco et al., 2023 — Task Arithmetic
- Yadav et al., 2023 — TIES-Merging
- Pre-released checkpoints: https://github.com/mlfoundations/task_vectors
