# KB-DSS: Knowledge-Based AI Decision Support for Digital Transformation in Entrepreneurial Firms

Code and results for the short communication *"A Knowledge-Based AI Decision Support System for Data-Driven Digital Transformation and Innovation in Entrepreneurial Firms"* (submitted to *Knowledge-Based Systems*).

**Principle: knowledge proposes, data disposes.** A knowledge layer proposes digital-transformation (DT) moves that are plausible for a venture; a data layer estimates how much each move would raise the venture's chance of success; a single trade-off weight λ, chosen on a validation cohort, combines the two.

| Layer | Component |
|---|---|
| Data | First-round venture records from the public Crunchbase 2015 export |
| Knowledge | Expert DT ontology (9 pillars); category knowledge graph (transitions P(c\|s), PPMI, SVD embeddings, out-of-fold success priors) |
| Decision | (1) learned option-generation ranker; (2) gradient-boosted success estimator; (3) recommendation utility U = (1 − λ)·plausibility + λ·predicted gain |

## Reproduce

```bash
pip install -r requirements.txt
python kbdss.py
```

The script downloads the data on first run (about 30 MB), runs every experiment and writes all tables and figures to `results/`. It takes about 4 minutes on a laptop CPU and is deterministic (`SEED = 42`).

| Output | Paper |
|---|---|
| `results/table1_option_generation.csv` | Table 1, option generation on 2011–2015 ventures |
| `results/table2_recommendation.csv` | Table 2, recommendations scored on realised outcomes |
| `results/Figure_1.png` / `.pdf` | Fig. 1, KB-DSS architecture |
| `results/Figure_2.png` / `.pdf` | Fig. 2, (a) Hits@k, (b) λ trade-off, (c) temporal drift |
| `results/lambda_sweep.csv`, `results/prediction_ablation.csv`, `results/summary.json` | Numbers quoted in the text |

## Evaluation protocol (no look-ahead)

- **Option generation:** the ranker is fitted on ventures first funded in 2009–2010 using a knowledge graph built only from ventures funded up to 2008. It is tested on ventures first funded in 2011–2015 (one category hidden per venture) with a knowledge graph built up to 2010.
- **Success estimation:** trained on labelled ventures (acquired/IPO = success, closed = failure) first funded up to 2010 and tested on 2011–2015. Only first-round information is used.
- **Recommendation:** λ is selected on the 2009–2010 validation cohort and applied once to the 2011–2015 cohort. Each recommendation is scored against the **realised** outcomes of other 2011–2015 ventures that share a category with the target venture (success-rate difference with vs. without the recommended category, shrunk by n/(n+10)). These outcomes are never used for training.
- **Statistics:** Wilcoxon signed-rank tests on matched ventures and 1,000-sample bootstrap 95% confidence intervals.

## Data and license

The data are the December 2015 Crunchbase export as mirrored at [notpeter/crunchbase-data](https://github.com/notpeter/crunchbase-data), licensed CC BY-NC 4.0 by Crunchbase. The data are **not** redistributed here; they are downloaded at run time. Use them for non-commercial research only. The code is released under the MIT License.

## Citation

```
S.Y.R. Ravula, A Knowledge-Based AI Decision Support System for Data-Driven Digital Transformation
and Innovation in Entrepreneurial Firms, manuscript submitted to Knowledge-Based Systems, 2026.
```
