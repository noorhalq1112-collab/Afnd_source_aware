# Source-Aware Evaluation of AraBERT for Arabic Fake News Detection

CAI 421 Deep Learning group project (Princess Nourah bint Abdulrahman University). Target venue: ICMI 2027 (IEEE template).

AFND labels belong to whole news websites, not to single articles. A random article-level split therefore puts the same website in both training and test sets. This project compares that random split with a **source-disjoint split** (whole websites held out) for two models: TF-IDF + linear SVM and AraBERT, and tests layer-wise learning-rate decay (LLRD) and class-weighted loss in an ablation.

## Main results (macro-F1, mean ± std over seeds 42-44)

| Model | Random split | Source-disjoint split | Gap |
|---|---|---|---|
| TF-IDF + linear SVM | 0.771 ± 0.002 | 0.536 ± 0.022 | 0.235 |
| AraBERT (default) | 0.883 ± 0.004 | 0.584 ± 0.071 | 0.299 |

Ablation on the source-disjoint split (macro-F1): default 0.583 ± 0.071, LLRD 0.610 ± 0.102, weighted loss 0.588 ± 0.078, LLRD + weighted 0.605 ± 0.101. The differences are small and not consistent across seeds.

## Files

| File | Content |
|---|---|
| `AFND_source_aware_checkpoint2.ipynb` | Full code with outputs (data loading, splits, SVM, AraBERT, ablation, SVM feature weights, SHAP) |
| `requirements.txt` | Python dependencies |
| `results.csv` | One row per run (model, split, seed, n_train, n_test, macro_f1, acc, per-class recall, val_macro_f1, train_seconds) |
| `summary.csv` | Mean and std per model and split (from `results.csv`) |
| `gap_table.csv` | Random vs. source-disjoint macro-F1 and the gap |
| `ablation.csv` | One row per ablation run (config, seed, metrics) |

The result files are the raw outputs of the notebook and are not edited by hand.
`sample_afnd.csv` (the exact sample used, 27,795 articles from 101 sources) is written by the notebook to `results/` and is not stored here because of its size; running the notebook with the same seed recreates it.

## How to reproduce (Google Colab, T4 GPU)

1. Open `AFND_source_aware_checkpoint2.ipynb` in Google Colab (File > Upload notebook).
2. Runtime > Change runtime type > T4 GPU.
3. In the config cell set `SMOKE = False`, `bert_repeats = 3`, `RUN_ABLATION = True`, `RUN_ANALYSIS = True`.
4. Run all. The data are downloaded automatically from Kaggle (`murtadhayaseen/arabic-fake-news-dataset-afnd`). A full run takes a few hours; one AraBERT run takes about 5 minutes on a T4.
5. Outputs are written to `results/`.

For a quick test, set `SMOKE = True` (a few minutes, not for reporting).

## Settings

Seed 42 (+ repeat index 0, 1, 2) for sampling, splits and training; at most 300 articles per source; text = title + first 2,000 characters, 128 tokens; AraBERT v0.2 (`aubmindlab/bert-base-arabertv02`); 2 epochs, batch size 32, peak learning rate 2e-5, LLRD factor 0.9, warm-up 10%; 70/15/15 splits (stratified by class; by source in the source-disjoint protocol).

## Notes and limitations

- GPU kernels make runs differ slightly even with fixed seeds (for example, the `default` ablation run for seed 42 differs from the main AraBERT run by 0.001 macro-F1).
- The sample has 101 sources instead of the 103 expected for the binary task (52 credible, 49 not credible).
- AFND labels are weak (assigned per website). The SHAP analysis uses a small sample of test articles and is preliminary.

## Data and citation

AFND, CC BY 4.0, Mendeley Data DOI 10.17632/67mhx6hhzd.1. A. Khalil, M. Jarrah, M. Aldwairi, M. Jaradat, "AFND: Arabic fake news dataset for the detection and classification of articles credibility," Data in Brief, vol. 42, 108141, 2022, doi: 10.1016/j.dib.2022.108141. The data are not redistributed in this repository.
