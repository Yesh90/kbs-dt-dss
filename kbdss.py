"""
KB-DSS: a knowledge-based AI decision support system for data-driven
digital transformation (DT) and innovation in entrepreneurial firms.

One file, three layers:
  1. Data layer       - firm records from the public Crunchbase 2015 export
  2. Knowledge layer  - (a) an expert DT ontology that maps market categories
                        to DT pillars, (b) a category knowledge graph learned
                        from co-occurrence (PPMI + SVD embeddings, success
                        priors, recombinant novelty)
  3. Decision layer   - success prediction, knowledge-graph completion, and
                        success-aware DT recommendations

Run:  python kbdss.py            (downloads data, writes results/)
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

SEED = 42
SPLIT_YEAR = 2010            # train: first funding <= 2010, test: 2011-2015
EMB_DIM = 32
PRIOR_M = 20                 # Bayesian smoothing strength for success priors
MIN_CAT = 10                 # minimum firms per category in the knowledge graph
BOOT = 1000
URL = "https://raw.githubusercontent.com/notpeter/crunchbase-data/master/{}.csv"
DATA, OUT = "data", "results"

# --------------------------------------------------------------------------
# Knowledge layer (a): expert digital-transformation ontology
# Each pillar is defined by keyword patterns matched against category names.
# --------------------------------------------------------------------------
ONTOLOGY = {
    "AI & Analytics": ["analytics", "big data", "machine learning", "artificial intelligence", "data mining",
                       "predictive", "business intelligence", "data visualization", "visualization", "semantic",
                       "image recognition", "natural language", "databases", "data integration", "optimization"],
    "Cloud & Infrastructure": ["cloud", "saas", "paas", "web hosting", "data center", "virtualization", "storage",
                               "infrastructure", "developer api", "developer tools", "application platforms",
                               "open source", "content delivery", "development platforms"],
    "Mobile": ["mobile", "apps", "android", "iphone", "ios", "ipad", "tablets", "wireless", "sms",
               "location based", "gps"],
    "Platforms & Marketplaces": ["marketplace", "crowdsourcing", "crowdfunding", "peer-to-peer",
                                 "collaborative consumption", "freelancers", "social network", "communities",
                                 "sharing economy"],
    "Digital Commerce & Marketing": ["e-commerce", "online shopping", "advertising", "marketing", "seo",
                                     "ad targeting", "lead generation", "social commerce", "coupons", "loyalty",
                                     "price comparison", "online reservations", "online travel"],
    "FinTech": ["fintech", "finance technology", "payments", "bitcoin", "banking", "trading", "personal finance",
                "insurance", "point of sale"],
    "Cybersecurity": ["security", "identity", "privacy", "fraud", "risk management"],
    "IoT & Smart Hardware": ["internet of things", "sensors", "wearables", "home automation", "robotics",
                             "3d printing", "consumer electronics", "semiconductors", "augmented reality",
                             "hardware"],
    "Digital Content": ["video", "music", "games", "digital media", "content", "publishing", "streaming", "photo",
                        "audio", "television", "news", "blogging"],
}
PILLARS = list(ONTOLOGY)


def pillars_of(category: str) -> set[str]:
    c = category.lower()
    return {p for p, kws in ONTOLOGY.items() if any(k in c for k in kws)}


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
def load() -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    for name in ("companies", "rounds"):
        path = f"{DATA}/{name}.csv"
        if not os.path.exists(path):
            urllib.request.urlretrieve(URL.format(name), path)
    c = pd.read_csv(f"{DATA}/companies.csv", encoding="latin1")
    r = pd.read_csv(f"{DATA}/rounds.csv", encoding="latin1")
    r["funded_at"] = pd.to_datetime(r["funded_at"], errors="coerce")
    first = (r.dropna(subset=["funded_at"]).sort_values("funded_at")
              .groupby("company_permalink").first()[["funding_round_type", "raised_amount_usd", "funded_at"]])
    df = c.merge(first, left_on="permalink", right_index=True, how="inner")
    df = df[df["category_list"].notna()].copy()
    df["y"] = df["status"].isin(["acquired", "ipo"]).astype(int)
    df["cats"] = df["category_list"].str.split("|").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    df["year"] = df["funded_at"].dt.year
    df = df[(df["year"] >= 1995) & (df["year"] <= 2015)]
    founded = pd.to_datetime(df["founded_at"], errors="coerce")
    df["lag_years"] = (df["funded_at"] - founded).dt.days / 365.25
    df["labeled"] = df["status"].isin(["acquired", "ipo", "closed"])
    return df.reset_index(drop=True)


def base_features(df: pd.DataFrame, top_countries: list[str]) -> pd.DataFrame:
    """Data-only features available at the time of the first funding round (no outcome leakage)."""
    X = pd.DataFrame(index=df.index)
    X["log_first_round"] = np.log1p(df["raised_amount_usd"].fillna(0))
    X["first_round_missing_amt"] = df["raised_amount_usd"].isna().astype(int)
    X["first_year"] = df["year"]
    X["lag_years"] = df["lag_years"].clip(-5, 50)
    X["n_categories"] = df["cats"].str.len()
    for t in ["seed", "angel", "venture", "debt_financing", "equity_crowdfunding", "grant", "private_equity"]:
        X[f"round_{t}"] = (df["funding_round_type"] == t).astype(int)
    for cc in top_countries:
        X[f"country_{cc}"] = (df["country_code"] == cc).astype(int)
    return X


def raw_category_features(df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    """Data-driven but knowledge-free: one-hot of the most frequent raw categories."""
    idx = {c: i for i, c in enumerate(vocab)}
    M = np.zeros((len(df), len(vocab)), dtype=np.int8)
    for row, cats in enumerate(df["cats"]):
        for c in cats:
            if c in idx:
                M[row, idx[c]] = 1
    return pd.DataFrame(M, index=df.index, columns=[f"cat_{c}" for c in vocab])


def ontology_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(0, index=df.index, columns=[f"pillar_{p}" for p in PILLARS])
    share = np.zeros(len(df))
    for row, cats in enumerate(df["cats"]):
        hit = [pillars_of(c) for c in cats]
        for ps in hit:
            for p in ps:
                X.iat[row, PILLARS.index(p)] = 1
        share[row] = np.mean([bool(h) for h in hit]) if hit else 0.0
    X["dt_intensity"] = X[[f"pillar_{p}" for p in PILLARS]].sum(1)
    X["digital_share"] = share
    X["ai_x_cloud"] = X["pillar_AI & Analytics"] * X["pillar_Cloud & Infrastructure"]
    X["ai_x_platform"] = X["pillar_AI & Analytics"] * X["pillar_Platforms & Marketplaces"]
    return X


# --------------------------------------------------------------------------
# Knowledge layer (b): category knowledge graph
# --------------------------------------------------------------------------
class CategoryKG:
    """Weighted co-occurrence graph of market categories with PPMI-SVD embeddings."""

    def __init__(self, cat_lists, min_count=MIN_CAT, dim=EMB_DIM):
        cnt = Counter(c for cats in cat_lists for c in cats)
        self.nodes = sorted(c for c, n in cnt.items() if n >= min_count)
        self.idx = {c: i for i, c in enumerate(self.nodes)}
        self.freq = np.array([cnt[c] for c in self.nodes], dtype=float)
        n = len(self.nodes)
        rows, cols = [], []
        for cats in cat_lists:
            ids = sorted({self.idx[c] for c in cats if c in self.idx})
            for a, b in combinations(ids, 2):
                rows += [a, b]
                cols += [b, a]
        A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        A.sum_duplicates()
        self.A = A
        total = A.sum()
        deg = np.asarray(A.sum(1)).ravel()
        A = A.tocoo()
        pmi = np.log((A.data * total) / (deg[A.row] * deg[A.col] + 1e-12))
        P = sparse.coo_matrix((np.maximum(pmi, 0), (A.row, A.col)), shape=(n, n)).tocsr()
        svd = TruncatedSVD(n_components=dim, random_state=SEED)
        E = svd.fit_transform(P)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.degree = deg / (deg.max() + 1e-12)
        self.n_edges = int(A.nnz // 2)

    def ids(self, cats):
        return [self.idx[c] for c in cats if c in self.idx]

    def firm_vector(self, cats):
        ids = self.ids(cats)
        if not ids:
            return np.zeros(self.E.shape[1])
        v = self.E[ids].mean(0)
        return v / (np.linalg.norm(v) + 1e-12)

    def novelty(self, cats):
        """Recombinant novelty: mean pairwise cosine distance between the firm's categories."""
        ids = self.ids(cats)
        if len(ids) < 2:
            return 0.0
        S = self.E[ids] @ self.E[ids].T
        iu = np.triu_indices(len(ids), 1)
        return float(np.mean(1 - S[iu]))


def success_priors(cat_lists, y, kg, m=PRIOR_M):
    """Smoothed success rate for every KG node (beta prior centered on the global rate)."""
    g = float(np.mean(y))
    s, n = np.zeros(len(kg.nodes)), np.zeros(len(kg.nodes))
    for cats, yi in zip(cat_lists, y):
        for i in kg.ids(cats):
            s[i] += yi
            n[i] += 1
    return (s + m * g) / (n + m)


def kg_features(df, kg, prior):
    rows = []
    for cats in df["cats"]:
        ids = kg.ids(cats)
        pr = prior[ids] if ids else None
        rows.append({
            "kg_prior_mean": pr.mean() if pr is not None else np.nan,
            "kg_prior_max": pr.max() if pr is not None else np.nan,
            "kg_prior_min": pr.min() if pr is not None else np.nan,
            "kg_centrality": kg.degree[ids].mean() if ids else 0.0,
            "kg_novelty": kg.novelty(cats), "kg_coverage": len(ids) / max(len(cats), 1),
        })
    X = pd.DataFrame(rows, index=df.index)
    V = np.vstack([kg.firm_vector(c) for c in df["cats"]])
    return pd.concat([X, pd.DataFrame(V, index=df.index, columns=[f"kg_emb_{i}" for i in range(V.shape[1])])], axis=1)


def kg_features_oof(train, kg, folds=5):
    """Out-of-fold success priors on the training set to avoid target leakage."""
    out = pd.DataFrame(index=train.index)
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    parts = []
    for tr, va in skf.split(train, train["y"]):
        pr = success_priors(train["cats"].iloc[tr], train["y"].iloc[tr].values, kg)
        parts.append(kg_features(train.iloc[va], kg, pr))
    return pd.concat(parts).loc[out.index]


# --------------------------------------------------------------------------
# Decision layer
# --------------------------------------------------------------------------
def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15, random_state=SEED)


def boot_ci(y, p, fn, B=BOOT):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def paired_delta(y, p_a, p_b, B=BOOT):
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.mean(d <= 0))


def kg_completion(test, kg, train_freq, k_list=(5, 10)):
    """Hide one category per test firm and rank all KG nodes as candidates to recover it."""
    rng = np.random.default_rng(SEED)
    pop = train_freq.copy()
    A = kg.A.toarray()
    res = {"Popularity": [], "Co-occurrence count": [], "KG embedding (PPMI-SVD)": [], "KB-DSS hybrid (co-occurrence + embedding)": []}
    deg = A.sum(1) + 1e-12
    rank01 = lambda v: (np.argsort(np.argsort(v)) / (len(v) - 1))
    for cats in test["cats"]:
        ids = kg.ids(cats)
        if len(ids) < 2:
            continue
        hidden = ids[rng.integers(len(ids))]
        ctx = [i for i in ids if i != hidden]
        v = kg.E[ctx].mean(0)
        cond = (A[ctx] / deg[ctx][:, None]).mean(0)              # mean P(candidate | context category)
        emb = kg.E @ v + 1e-3 * np.log1p(pop) / np.log1p(pop).max()
        scores = {"Popularity": pop.copy(), "Co-occurrence count": A[ctx].sum(0).astype(float) + 1e-3 * pop / pop.max(),
                  "KG embedding (PPMI-SVD)": emb,
                  "KB-DSS hybrid (co-occurrence + embedding)": 0.5 * rank01(cond + 1e-6 * pop) + 0.5 * rank01(emb)}
        for name, s in scores.items():
            s = s.copy()
            s[ctx] = -np.inf
            rank = int((s > s[hidden]).sum()) + 1
            res[name].append(rank)
    rows = []
    for name, ranks in res.items():
        r = np.array(ranks)
        rows.append({"Method": name, **{f"Hits@{k}": np.mean(r <= k) for k in k_list}, "MRR": np.mean(1 / r),
                     "Median rank": float(np.median(r)), "n": len(r)})
    return pd.DataFrame(rows)


def recommend(firm, kg, prior, clf, cols, top_countries, n_candidates=20, top_k=3):
    """Success-aware DT recommendation: KG-plausible categories re-ranked by predicted success gain."""
    one = firm.to_frame().T.infer_objects()
    one["cats"] = [firm["cats"]]
    feats = lambda d: assemble(d, top_countries, kg, prior)[cols]
    base = clf.predict_proba(feats(one))[0, 1]
    v = kg.firm_vector(firm["cats"])
    s = kg.E @ v
    s[kg.ids(firm["cats"])] = -np.inf
    cand = np.argsort(-s)[:n_candidates]
    out = []
    for c in cand:
        alt = one.copy()
        alt["cats"] = [firm["cats"] + [kg.nodes[c]]]
        gain = clf.predict_proba(feats(alt))[0, 1] - base
        out.append((kg.nodes[c], float(s[c]), float(gain)))
    out.sort(key=lambda t: -t[2])
    return base, out[:top_k]


VOCAB: list[str] = []


def assemble(df, top_countries, kg, prior, kg_block=None):
    blocks = [base_features(df, top_countries), raw_category_features(df, VOCAB), ontology_features(df),
              kg_block if kg_block is not None else kg_features(df, kg, prior)]
    return pd.concat(blocks, axis=1)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_framework(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.05, 4.6); ax.axis("off")
    col = {"data": "#dbe9f6", "know": "#fde7c8", "dec": "#d9f0d9"}

    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=c, ec="#333", lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))

    ax.text(1.35, 4.45, "Data layer", ha="center", fontsize=10, weight="bold", color="#1f4e79")
    ax.text(5.0, 4.45, "Knowledge layer", ha="center", fontsize=10, weight="bold", color="#9c5700")
    ax.text(8.65, 4.45, "Decision layer", ha="center", fontsize=10, weight="bold", color="#2e6b2e")
    box(0.1, 2.35, 2.5, 1.9, "Firm records", "Crunchbase export\nfirst funding round\nmarket categories\nlocation, timing", col["data"])
    box(0.1, 0.25, 2.5, 1.8, "Data-only features", "round type and size\nfounding lag, year\ncountry, breadth", col["data"])
    box(3.4, 2.35, 3.2, 1.9, "Expert DT ontology", "9 DT pillars mapped\nfrom market categories\nDT intensity, digital share\nAI x cloud, AI x platform", col["know"])
    box(3.4, 0.25, 3.2, 1.8, "Category knowledge graph", "co-occurrence graph\nPPMI-SVD embeddings\nsuccess priors (out-of-fold)\nrecombinant novelty", col["know"])
    box(7.4, 2.35, 2.5, 1.9, "Success prediction", "gradient-boosted\nclassifier with\nknowledge features", col["dec"])
    box(7.4, 0.25, 2.5, 1.8, "DT recommendation", "KG-plausible options\nre-ranked by predicted\nsuccess gain", col["dec"])
    arrow(2.6, 3.3, 3.4, 3.3)                     # records -> ontology
    arrow(2.6, 2.6, 3.4, 1.7)                     # records -> knowledge graph
    arrow(1.35, 2.35, 1.35, 2.05)                 # records -> data-only features
    arrow(6.6, 3.3, 7.4, 3.3)                     # ontology -> prediction
    arrow(6.6, 1.75, 7.4, 2.75)                   # KG -> prediction
    arrow(6.6, 1.15, 7.4, 1.15)                   # KG -> recommendation
    arrow(8.65, 2.35, 8.65, 2.05)                 # prediction -> recommendation
    ax.plot([1.35, 1.35, 7.05, 7.05], [0.25, 0.08, 0.08, 2.55], color="#333", lw=1.2)
    arrow(7.05, 2.55, 7.4, 2.55)                  # data-only features -> prediction (routed below)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_results(path, y, preds, groups_imp):
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    styles = ["#9e9e9e", "#6baed6", "#fd8d3c", "#74c476", "#08519c"]
    for (name, p), c in zip(preds.items(), styles):
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=c, lw=1.8 if "KB-DSS" in name else 1.3, label=f"{name.split(' ')[0]} (AUC {roc_auc_score(y, p):.3f})")
    a.plot([0, 1], [0, 1], ls=":", c="#999", lw=1)
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_title("(a) Temporal hold-out ROC (2011-2015 cohort)", fontsize=10)
    a.legend(fontsize=7.2, loc="lower right", frameon=False)
    g = groups_imp.sort_values("importance")
    b.barh(g["group"], g["importance"], xerr=g["sd"], color=["#08519c" if "KG" in s else "#fd8d3c" if "ontology" in s.lower() else "#9e9e9e" for s in g["group"]], capsize=3)
    b.set_xlabel("Drop in AUC when the feature group is permuted")
    b.set_title("(b) Contribution of knowledge sources", fontsize=10)
    b.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def main():
    global VOCAB
    os.makedirs(OUT, exist_ok=True)
    allf = load()
    df = allf[allf["labeled"]].reset_index(drop=True)
    train, test = df[df["year"] <= SPLIT_YEAR].copy(), df[df["year"] > SPLIT_YEAR].copy()
    top_countries = train["country_code"].value_counts().head(8).index.tolist()
    VOCAB = pd.Series([c for cs in train["cats"] for c in cs]).value_counts().head(100).index.tolist()

    # The knowledge graph uses only categories (no outcomes) of ALL firms first funded up to the split year,
    # including still-operating firms, so it carries no label information and no future information.
    kg_source = allf[allf["year"] <= SPLIT_YEAR]["cats"]
    kg = CategoryKG(kg_source)
    prior_full = success_priors(train["cats"], train["y"].values, kg)
    Xtr = assemble(train, top_countries, kg, None, kg_block=kg_features_oof(train, kg))
    Xte = assemble(test, top_countries, kg, prior_full)
    ytr, yte = train["y"].values, test["y"].values

    base_cols = list(base_features(train.head(1), top_countries).columns)
    raw_cols = [c for c in Xtr if c.startswith("cat_")]
    ont_cols = [c for c in Xtr if c.startswith("pillar_")] + ["dt_intensity", "digital_share", "ai_x_cloud", "ai_x_platform"]
    kg_cols = [c for c in Xtr if c.startswith("kg_")]
    dd = base_cols + raw_cols
    configs = {
        "M1 Data only": base_cols,
        "M2 Data-driven (M1 + raw categories)": dd,
        "M3 M2 + DT ontology": dd + ont_cols,
        "M4 M2 + category KG": dd + kg_cols,
        "M5 KB-DSS (M2 + ontology + KG)": dd + ont_cols + kg_cols,
    }

    rows, preds, fitted = [], {}, {}
    for name, cols in configs.items():
        clf = model().fit(Xtr[cols], ytr)
        p = clf.predict_proba(Xte[cols])[:, 1]
        preds[name], fitted[name] = p, clf
        lo, hi = boot_ci(yte, p, roc_auc_score)
        rows.append({"Model": name, "Features": len(cols), "AUC": roc_auc_score(yte, p), "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "PR-AUC": average_precision_score(yte, p), "Brier": brier_score_loss(yte, p)})
    t1 = pd.DataFrame(rows)

    # In-period reference: 5-fold CV inside the training cohort (same era, no temporal shift)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {n: [] for n in configs}
    tr_r = train.reset_index(drop=True)
    Xtr_r = Xtr.reset_index(drop=True)
    for a, b in skf.split(tr_r, tr_r["y"]):
        A, Bf = tr_r.iloc[a], tr_r.iloc[b]
        Xa = Xtr_r.iloc[a].copy(); Xb = Xtr_r.iloc[b].copy()
        Xa[kg_cols] = kg_features_oof(A, kg)[kg_cols].values
        Xb[kg_cols] = kg_features(Bf, kg, success_priors(A["cats"], A["y"].values, kg))[kg_cols].values
        for name, cols in configs.items():
            m = model().fit(Xa[cols], A["y"])
            cv[name].append(roc_auc_score(Bf["y"], m.predict_proba(Xb[cols])[:, 1]))
    t1.insert(2, "In-period CV AUC", [f"{np.mean(cv[n]):.3f} +/- {np.std(cv[n]):.3f}" for n in configs])
    full = "M5 KB-DSS (M2 + ontology + KG)"
    ref = "M2 Data-driven (M1 + raw categories)"
    for name in configs:
        if name != ref:
            d, pval = paired_delta(yte, preds[name], preds[ref])
            t1.loc[t1["Model"] == name, "dAUC vs M2"] = f"{d:+.4f}"
            t1.loc[t1["Model"] == name, "p (bootstrap)"] = round(pval, 3)
    t1.to_csv(f"{OUT}/table1_prediction.csv", index=False)

    # Grouped permutation importance of the full model
    clf, cols = fitted[full], configs[full]
    rng = np.random.default_rng(SEED)
    base_auc = roc_auc_score(yte, preds[full])
    groups = {"Data: raw categories": raw_cols, "Data: funding round": [c for c in base_cols if c.startswith("round_") or "first_round" in c],
              "Data: timing": ["first_year", "lag_years"], "Data: location": [c for c in base_cols if c.startswith("country_")],
              "DT ontology: pillars": [c for c in ont_cols if c.startswith("pillar_")],
              "DT ontology: intensity/share": ["dt_intensity", "digital_share", "ai_x_cloud", "ai_x_platform", "n_categories"],
              "KG: success priors": ["kg_prior_mean", "kg_prior_max", "kg_prior_min"],
              "KG: embeddings": [c for c in kg_cols if c.startswith("kg_emb_")],
              "KG: novelty/centrality": ["kg_novelty", "kg_centrality", "kg_coverage"]}
    imp = []
    for gname, gcols in groups.items():
        drops = []
        for _ in range(10):
            Xp = Xte[cols].copy()
            perm = rng.permutation(len(Xp))
            Xp[gcols] = Xp[gcols].values[perm]
            drops.append(base_auc - roc_auc_score(yte, clf.predict_proba(Xp)[:, 1]))
        imp.append({"group": gname, "importance": np.mean(drops), "sd": np.std(drops)})
    imp = pd.DataFrame(imp)
    imp.to_csv(f"{OUT}/permutation_importance.csv", index=False)

    # Knowledge-graph completion (plausibility of DT options)
    t2 = kg_completion(test, kg, kg.freq)
    t2.to_csv(f"{OUT}/table2_kg_completion.csv", index=False)

    # Pillar-level descriptive knowledge (train cohort)
    O = ontology_features(train)
    pill = pd.DataFrame([{"pillar": p, "share_of_firms": O[f"pillar_{p}"].mean(),
                          "success_rate": train["y"][O[f"pillar_{p}"] == 1].mean()} for p in PILLARS])
    pill.to_csv(f"{OUT}/pillar_summary.csv", index=False)

    # Success-aware recommendations for closed firms in the test cohort
    closed = test[test["y"] == 0]
    sample = closed.sample(300, random_state=SEED)
    gains, rand_gains, top_examples = [], [], []
    rng2 = np.random.default_rng(SEED)
    feats = lambda d: assemble(d, top_countries, kg, prior_full)[cols]
    for _, f in sample.iterrows():
        base_p, recs = recommend(f, kg, prior_full, clf, cols, top_countries)
        gains.append(recs[0][2])
        one = f.to_frame().T.infer_objects()
        pool = [c for c in kg.nodes if c not in f["cats"]]
        alts = []
        for c in rng2.choice(pool, 3, replace=False):          # control: add a random KG category
            alt = one.copy(); alt["cats"] = [f["cats"] + [c]]; alts.append(alt)
        pr = clf.predict_proba(feats(pd.concat(alts)))[:, 1]
        rand_gains.append(float(pr.max() - base_p))
        if len(top_examples) < 3 and len(f["cats"]) >= 2:
            top_examples.append({"firm_categories": f["cats"], "predicted_success": round(base_p, 3),
                                 "recommendations": [{"add": r[0], "kg_plausibility": round(r[1], 3), "predicted_gain": round(r[2], 3)} for r in recs]})
    rec_summary = {"n_firms": len(sample), "mean_top1_predicted_gain": float(np.mean(gains)),
                   "control_random_best_of_3_mean_gain": float(np.mean(rand_gains)),
                   "share_kbdss_beats_control": float(np.mean(np.array(gains) > np.array(rand_gains))),
                   "median_top1_predicted_gain": float(np.median(gains)), "share_positive_gain": float(np.mean(np.array(gains) > 0)),
                   "examples": top_examples}

    summary = {"n_firms": len(df), "n_train": len(train), "n_test": len(test),
               "success_rate_train": float(ytr.mean()), "success_rate_test": float(yte.mean()),
               "kg_nodes": len(kg.nodes), "kg_edges": kg.n_edges, "ontology_pillars": len(PILLARS),
               "ontology_coverage_categories": float(np.mean([bool(pillars_of(c)) for c in kg.nodes])),
               "recommendation": rec_summary}
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    figure_framework(f"{OUT}/Figure_1.png")
    figure_results(f"{OUT}/Figure_2.png", yte, preds, imp)
    pd.set_option("display.width", 200)
    print(t1.round(4).to_string(index=False), "\n")
    print(t2.round(4).to_string(index=False), "\n")
    print(imp.round(4).to_string(index=False), "\n")
    print(pill.round(3).to_string(index=False), "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
