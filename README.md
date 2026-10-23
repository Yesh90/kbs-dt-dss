# KB-DSS: Knowledge-Based AI Decision Support for Digital Transformation in Entrepreneurial Firms

Code and results for the short communication *"A Knowledge-Based AI Decision Support System for Data-Driven Digital Transformation and Innovation in Entrepreneurial Firms"* (submitted to *Knowledge-Based Systems*).

The whole system is a single Python file, `kbdss.py`, organized in three layers:

| Layer | What it does |
|---|---|
| Data | Firm records from the public Crunchbase export (first funding round, market categories, location, timing) |
| Knowledge | (a) an expert digital-transformation ontology that maps market categories to 9 DT pillars; (b) a category knowledge graph learned from co-occurrence, with PPMI-SVD embeddings, out-of-fold success priors and a recombinant-novelty score |
| Decision | Venture-success prediction, knowledge-graph completion of plausible DT options, and success-aware DT recommendations |

## Reproduce

```bash
pip install -r requirements.txt
python kbdss.py
```

The script downloads the data on first run (about 30 MB), runs every experiment and writes everything to `results/`. It takes about 3 minutes on a laptop CPU and is fully deterministic (`SEED = 42`).

| Output | Paper |
|---|---|
| `results/table1_prediction.csv` | Table 1, ablation of knowledge sources (temporal hold-out and in-period CV) |
| `results/table2_kg_completion.csv` | Table 2, knowledge-graph completion of DT options |
| `results/Figure_1.png` | Fig. 1, KB-DSS architecture |
| `results/Figure_2.png` | Fig. 2, ROC curves and grouped permutation importance |
| `results/permutation_importance.csv`, `results/pillar_summary.csv`, `results/summary.json` | Numbers quoted in the text, including recommendation results |

## Evaluation design

- **Outcome:** success = acquired or IPO; failure = closed. Firms still operating have no outcome and are excluded from supervised learning, but they are used (without labels) to build the knowledge graph.
- **No look-ahead:** only information available at the first funding round is used. Total funding, number of rounds and last-funding dates are excluded.
- **Temporal hold-out:** train on firms first funded in 1995–2010, test on 2011–2015. The knowledge graph is built only from firms first funded up to 2010.
- **Leakage control:** category success priors are computed out-of-fold on the training set.
- **Statistics:** 1,000-sample bootstrap 95% CIs and paired bootstrap tests against the data-driven baseline (M2).

## Data and license

The data are the December 2015 Crunchbase export as mirrored at [notpeter/crunchbase-data](https://github.com/notpeter/crunchbase-data), licensed CC BY-NC 4.0 by Crunchbase. The data are **not** redistributed here and are downloaded at run time. Use them for non-commercial research only.

The code is released under the MIT License.

## Citation

```
Ravula, S. Y. R. (2026). A Knowledge-Based AI Decision Support System for Data-Driven Digital
Transformation and Innovation in Entrepreneurial Firms. Manuscript submitted to Knowledge-Based Systems.
```
