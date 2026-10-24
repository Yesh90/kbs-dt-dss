"""
KB-DSS: a knowledge-based AI decision support system for data-driven
digital transformation (DT) and innovation in entrepreneurial firms.

Principle: knowledge generates plausible options; data ranks them by expected success.
  Data layer       venture records from the public Crunchbase 2015 export
  Knowledge layer  (a) expert DT ontology: market categories -> 9 DT pillars
                   (b) category knowledge graph (KG) learned from co-occurrence:
                       conditional transitions, PPMI, SVD embeddings, success priors
  Decision layer   (1) knowledge-guided option generation: a learned ranker scores
                       every category as a plausible next DT move for a venture
                   (2) success estimation: gradient-boosted venture-success model
                   (3) success-aware recommendation: utility
                       U = (1 - lambda) * plausibility + lambda * predicted gain,
                       with lambda chosen on a validation cohort

Protocol (no look-ahead anywhere):
  ranker   trained on ventures first funded 2009-2010 with a KG built from <=2008
  models   success model trained on labelled ventures first funded <=2010
  test     ventures first funded 2011–2015 with a KG built from <=2010
  outcome  recommendations are scored against REALISED outcomes of other
           2011–2015 ventures (never used for training)

Run:  python kbdss.py          (downloads data on first run, writes results/)
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
KG_VAL, SPLIT = 2008, 2010     # KG cut-off for ranker training; train/test split year
EMB_DIM, MIN_CAT, PRIOR_M = 32, 10, 20
N_NEG, N_REC = 30, 1000            # ranker negatives, ventures per recommendation benchmark
MIN_PEERS, SHRINK = 5, 10           # realised-uplift support threshold and shrinkage
BOOT = 1000
URL = "https://raw.githubusercontent.com/notpeter/crunchbase-data/master/{}.csv"
DATA, OUT = "data", "results"
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Knowledge layer (a): expert digital-transformation ontology
# ----------------------------------------------------------------------------
ONTOLOGY = {
    "AI & Analytics": ["analytics", "big data", "machine learning", "artificial intelligence", "data mining",
                       "predictive", "business intelligence", "visualization", "semantic", "image recognition",
                       "natural language", "databases", "data integration", "optimization"],
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
                             "3d printing", "consumer electronics", "semiconductors", "augmented reality", "hardware"],
    "Digital Content": ["video", "music", "games", "digital media", "content", "publishing", "streaming", "photo",
                        "audio", "television", "news", "blogging"],
}
PILLARS = list(ONTOLOGY)


@lru_cache(maxsize=None)
def _pillars(category: str) -> tuple:
    c = category.lower()
    return tuple(any(k in c for k in kws) for kws in ONTOLOGY.values())


def pillar_vector(category: str) -> np.ndarray:
    return np.array(_pillars(category), dtype=float)


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
def load() -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    for name in ("companies", "rounds"):
        if not os.path.exists(f"{DATA}/{name}.csv"):
            urllib.request.urlretrieve(URL.format(name), f"{DATA}/{name}.csv")
    c = pd.read_csv(f"{DATA}/companies.csv", encoding="latin1")
    r = pd.read_csv(f"{DATA}/rounds.csv", encoding="latin1")
    r["funded_at"] = pd.to_datetime(r["funded_at"], errors="coerce")
    first = (r.dropna(subset=["funded_at"]).sort_values("funded_at").groupby("company_permalink").first()
              [["funding_round_type", "raised_amount_usd", "funded_at"]])
    df = c.merge(first, left_on="permalink", right_index=True)
    df = df[df["category_list"].notna()].copy()
    df["cats"] = df["category_list"].str.split("|").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    df["year"] = df["funded_at"].dt.year
    df = df[(df["year"] >= 1995) & (df["year"] <= 2015)]
    df["lag_years"] = (df["funded_at"] - pd.to_datetime(df["founded_at"], errors="coerce")).dt.days / 365.25
    df["labeled"] = df["status"].isin(["acquired", "ipo", "closed"])
    df["y"] = df["status"].isin(["acquired", "ipo"]).astype(int)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Knowledge layer (b): category knowledge graph
# ----------------------------------------------------------------------------
class CategoryKG:
    """Weighted co-occurrence graph G=(V,E) over market categories."""

    def __init__(self, cat_lists, min_count=MIN_CAT, dim=EMB_DIM):
        cnt = Counter(c for cats in cat_lists for c in cats)
        self.nodes = sorted(c for c, n in cnt.items() if n >= min_count)
        self.idx = {c: i for i, c in enumerate(self.nodes)}
        n = len(self.nodes)
        self.freq = np.array([cnt[c] for c in self.nodes], float)
        rows, cols = [], []
        for cats in cat_lists:
            ids = sorted({self.idx[c] for c in cats if c in self.idx})
            for a, b in combinations(ids, 2):
                rows += [a, b]; cols += [b, a]
        A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).toarray()
        self.A, self.n_edges = A, int((A > 0).sum() // 2)
        deg = A.sum(1)
        self.cond = A / np.maximum(deg, 1)[:, None]                         # transition probability P(c | s)
        pmi = np.log(np.maximum(A * A.sum(), 1e-12) / np.maximum(np.outer(deg, deg), 1e-12))
        self.ppmi = np.where(A > 0, np.maximum(pmi, 0), 0.0)                 # positive pointwise mutual information
        E = TruncatedSVD(dim, random_state=SEED).fit_transform(sparse.csr_matrix(self.ppmi))
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.centrality = deg / max(deg.max(), 1)
        self.Pi = np.vstack([pillar_vector(c) for c in self.nodes])          # ontology membership
        M = self.Pi.T @ A @ self.Pi                                          # ontology-lifted pillar graph
        self.Q = M / np.maximum(M.sum(1, keepdims=True), 1)

    def ids(self, cats):
        return [self.idx[c] for c in cats if c in self.idx]

    def firm_vec(self, ids):
        v = self.E[ids].mean(0)
        return v / (np.linalg.norm(v) + 1e-12)

    def novelty(self, ids):
        if len(ids) < 2:
            return 0.0
        S = self.E[ids] @ self.E[ids].T
        return float(np.mean(1 - S[np.triu_indices(len(ids), 1)]))


# ----------------------------------------------------------------------------
# Decision layer (1): knowledge-guided option generation
# ----------------------------------------------------------------------------
KG_FEATS = ["kg_cond_mean", "kg_cond_max", "kg_ppmi", "kg_emb_cos"]
ONT_FEATS = ["ont_overlap", "ont_lift", "ont_digital"]
POP_FEATS = ["log_pop"]


def candidate_features(kg: CategoryKG, ctx: list[int]) -> np.ndarray:
    """Feature matrix (|V| x 8) scoring every KG node as an addition to context categories ctx."""
    logpop = np.log1p(kg.freq) / np.log1p(kg.freq).max()
    fp = kg.Pi[ctx].max(0)                                    # venture's DT pillars
    n_p = np.maximum(kg.Pi.sum(1), 1)
    overlap = (kg.Pi @ fp) / n_p
    lift = (kg.Pi * kg.Q[fp > 0].mean(0)).max(1) if fp.any() else np.zeros(len(kg.nodes))
    return np.column_stack([kg.cond[ctx].mean(0), kg.cond[ctx].max(0), kg.ppmi[ctx].mean(0),
                            kg.E @ kg.firm_vec(ctx), overlap, lift, kg.Pi.any(1).astype(float), logpop])


FEAT_NAMES = KG_FEATS + ONT_FEATS + POP_FEATS
RANKERS = {"Ontology ranker": ONT_FEATS + POP_FEATS, "KG ranker": KG_FEATS + POP_FEATS,
           "KB-DSS ranker": KG_FEATS + ONT_FEATS + POP_FEATS}


def hide_one(df, kg, rng):
    """For ventures with >=2 KG categories, hide one at random (the future 'next move')."""
    out = []
    for cats in df["cats"]:
        ids = kg.ids(cats)
        if len(ids) >= 2:
            h = ids[rng.integers(len(ids))]
            out.append(([i for i in ids if i != h], h))
    return out


def train_rankers(kg, cases, rng):
    X, y = [], []
    for ctx, h in cases:
        F = candidate_features(kg, ctx)
        pool = np.setdiff1d(np.arange(len(kg.nodes)), ctx + [h])
        neg = rng.choice(pool, min(N_NEG, len(pool)), replace=False)
        X.append(F[[h, *neg]]); y += [1] + [0] * len(neg)
    X = np.vstack(X)
    models = {}
    for name, feats in RANKERS.items():
        cols = [FEAT_NAMES.index(f) for f in feats]
        models[name] = (cols, make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[:, cols], y))
    return models


def score_all(kg, ctx, rankers):
    F = candidate_features(kg, ctx)
    s = {"Popularity": kg.freq.astype(float), "Co-occurrence": F[:, 0] + 1e-9 * kg.freq,
         "KG embedding": F[:, 3] + 1e-9 * kg.freq}
    for name, (cols, m) in rankers.items():
        s[name] = m.decision_function(F[:, cols])
    for v in s.values():
        v[ctx] = -np.inf
    return s


def completion_benchmark(kg, cases, rankers):
    ranks = {}
    for ctx, h in cases:
        for name, s in score_all(kg, ctx, rankers).items():
            ranks.setdefault(name, []).append(int((s > s[h]).sum()) + 1)
    ranks = {k: np.array(v) for k, v in ranks.items()}
    ref = 1 / ranks["KB-DSS ranker"]
    rows = []
    for name, r in ranks.items():
        row = {"Method": name, "Hits@1": np.mean(r <= 1), "Hits@5": np.mean(r <= 5), "Hits@10": np.mean(r <= 10),
               "MRR": np.mean(1 / r), "Median rank": float(np.median(r))}
        row["p vs KB-DSS"] = np.nan if name == "KB-DSS ranker" else wilcoxon(ref, 1 / r).pvalue
        rows.append(row)
    return pd.DataFrame(rows), ranks


# ----------------------------------------------------------------------------
# Decision layer (2): success estimation
# ----------------------------------------------------------------------------
class SuccessModel:
    """Gradient-boosted venture-success estimator on first-round data + knowledge features."""

    def __init__(self, train, kg, use_raw=True, use_ont=True, use_kg=True):
        self.kg, self.use = kg, (use_raw, use_ont, use_kg)
        self.countries = train["country_code"].value_counts().head(8).index.tolist()
        self.vocab = pd.Series([c for cs in train["cats"] for c in cs]).value_counts().head(100).index.tolist()
        self.vidx = {c: i for i, c in enumerate(self.vocab)}
        self.g = float(train["y"].mean())
        self.prior = self._priors(train)
        X = self._oof_features(train)
        self.cols = list(X.columns)
        self.clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
                                                  min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                                                  validation_fraction=0.15, random_state=SEED).fit(X, train["y"])

    def _priors(self, d):
        s, n = np.zeros(len(self.kg.nodes)), np.zeros(len(self.kg.nodes))
        for cats, yi in zip(d["cats"], d["y"]):
            for i in self.kg.ids(cats):
                s[i] += yi; n[i] += 1
        return (s + PRIOR_M * self.g) / (n + PRIOR_M)

    def _oof_features(self, train):
        parts = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, train["y"]):
            saved, self.prior = self.prior, self._priors(train.iloc[a])
            parts.append(self.features(train.iloc[b]))
            self.prior = saved
        return pd.concat(parts).loc[train.index]

    def features(self, d, cat_lists=None):
        cat_lists = list(d["cats"]) if cat_lists is None else cat_lists
        n = len(cat_lists)
        rep = lambda col: np.repeat(d[col].values, n // len(d)) if n != len(d) else d[col].values
        X = {"log_first_round": np.log1p(np.nan_to_num(rep("raised_amount_usd").astype(float))),
             "first_year": rep("year").astype(float), "lag_years": np.clip(rep("lag_years").astype(float), -5, 50),
             "n_categories": np.array([len(c) for c in cat_lists], float)}
        rt = rep("funding_round_type")
        for t in ["seed", "angel", "venture", "debt_financing", "equity_crowdfunding", "grant", "private_equity"]:
            X[f"round_{t}"] = (rt == t).astype(float)
        cc = rep("country_code")
        for c in self.countries:
            X[f"country_{c}"] = (cc == c).astype(float)
        X = pd.DataFrame(X)
        raw_on, ont_on, kg_on = self.use
        if raw_on:
            R = np.zeros((n, len(self.vocab)))
            for i, cats in enumerate(cat_lists):
                for c in cats:
                    if c in self.vidx:
                        R[i, self.vidx[c]] = 1
            X = pd.concat([X, pd.DataFrame(R, columns=[f"cat_{c}" for c in self.vocab])], axis=1)
        if ont_on:
            P = np.vstack([np.vstack([pillar_vector(c) for c in cats]).max(0) if cats else np.zeros(9) for cats in cat_lists])
            O = pd.DataFrame(P, columns=[f"pillar_{p}" for p in PILLARS])
            O["dt_intensity"] = P.sum(1)
            O["digital_share"] = [np.mean([pillar_vector(c).any() for c in cats]) if cats else 0 for cats in cat_lists]
            X = pd.concat([X, O], axis=1)
        if kg_on:
            K = []
            for cats in cat_lists:
                ids = self.kg.ids(cats)
                pr = self.prior[ids] if ids else np.array([np.nan])
                v = self.kg.firm_vec(ids) if ids else np.zeros(EMB_DIM)
                K.append([pr.mean(), pr.max(), pr.min(), self.kg.centrality[ids].mean() if ids else 0,
                          self.kg.novelty(ids), *v])
            K = pd.DataFrame(K, columns=["kg_prior_mean", "kg_prior_max", "kg_prior_min", "kg_centrality",
                                         "kg_novelty"] + [f"kg_emb_{i}" for i in range(EMB_DIM)])
            X = pd.concat([X, K], axis=1)
        X.index = d.index if n == len(d) else range(n)
        return X

    def predict(self, d, cat_lists=None):
        return self.clf.predict_proba(self.features(d, cat_lists)[self.cols])[:, 1]


def prediction_ablation(train, test, kg):
    """M1..M5: does knowledge improve discrimination? In-period CV vs temporal hold-out."""
    specs = {"M1 Data only": (False, False, False), "M2 + raw categories": (True, False, False),
             "M3 M2 + ontology": (True, True, False), "M4 M2 + KG": (True, False, True),
             "M5 M2 + ontology + KG": (True, True, True)}
    rows, models = [], {}
    tr = train.reset_index(drop=True)
    for name, use in specs.items():
        m = SuccessModel(train, kg, *use)
        models[name] = m
        cv = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, tr["y"]):
            mm = SuccessModel(tr.iloc[a], kg, *use)
            cv.append(roc_auc_score(tr.iloc[b]["y"], mm.predict(tr.iloc[b])))
        rows.append({"Model": name, "In-period AUC": np.mean(cv), "In-period SD": np.std(cv),
                     "Hold-out AUC": roc_auc_score(test["y"], m.predict(test))})
    return pd.DataFrame(rows), models


# ----------------------------------------------------------------------------
# Decision layer (3): success-aware recommendation, scored on realised outcomes
# ----------------------------------------------------------------------------
LAMBDAS = np.round(np.linspace(0, 1, 11), 2)


def pct_rank(v):
    """Percentile rank in [0,1] over finite entries (higher is better); -inf -> -1."""
    out = np.full(len(v), -1.0)
    f = np.isfinite(v)
    out[f] = (v[f].argsort().argsort() + 1) / f.sum()
    return out


def recommend_eval(firms, peers_df, kg, ranker, model, rng):
    """Score every strategy on realised outcomes of OTHER ventures in the same cohort.

    For venture f and recommended category c, the realised uplift is the success-rate difference
    between cohort peers (sharing >=1 category with f) that do and do not use c, shrunk by n/(n+SHRINK)."""
    cols, kb = ranker
    sets = [set(c) for c in peers_df["cats"]]
    yv = peers_df["y"].values
    pos = {ix: k for k, ix in enumerate(peers_df.index)}
    by_cat = {}
    for j, s in enumerate(sets):
        for c in s:
            by_cat.setdefault(c, set()).add(j)
    out = {k: {"uplift": [], "support": [], "gain": [], "plaus_rank": []} for k in ["Popularity", *LAMBDAS]}
    for ix, f in firms.iterrows():
        ctx = kg.ids(f["cats"])
        F = candidate_features(kg, ctx)
        plaus = kb.decision_function(F[:, cols]); plaus[ctx] = -np.inf
        prank = (-plaus).argsort().argsort() + 1
        cand = np.where(np.isfinite(plaus))[0]
        one = peers_df.loc[[ix]]
        gains = np.full(len(kg.nodes), -np.inf)
        gains[cand] = model.predict(one, [f["cats"] + [kg.nodes[c]] for c in cand]) - model.predict(one)[0]
        peers = set().union(*[by_cat.get(c, set()) for c in f["cats"]]) - {pos[ix]}
        pp, pg = pct_rank(plaus), pct_rank(gains)
        pop = kg.freq.copy(); pop[ctx] = -np.inf
        picks = {"Popularity": int(np.argmax(pop))}
        for lam in LAMBDAS:
            u = (1 - lam) * pp + lam * pg
            u[ctx] = -np.inf
            picks[lam] = int(np.argmax(u))
        for k, c in picks.items():
            w = peers & by_cat.get(kg.nodes[c], set())
            wo = peers - w
            upl = (yv[list(w)].mean() - yv[list(wo)].mean()) * len(w) / (len(w) + SHRINK) if w and wo else 0.0
            out[k]["uplift"].append(float(upl)); out[k]["support"].append(len(w) >= MIN_PEERS)
            out[k]["gain"].append(float(gains[c])); out[k]["plaus_rank"].append(int(prank[c]))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def boot_ci(x, rng):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT)]
    return np.percentile(bs, [2.5, 97.5])


def recommendation_table(res, lam_star, rng):
    names = {"Popularity": "Popularity", 0.0: "Plausibility only (lambda = 0)", 1.0: "Success only (lambda = 1)",
             lam_star: f"KB-DSS (lambda* = {lam_star:.1f})"}
    ref = res[lam_star]["uplift"]
    rows = []
    for k, label in names.items():
        r = res[k]
        lo, hi = boot_ci(100 * r["uplift"], rng)
        same = np.array_equal(r["uplift"], ref)
        rows.append({"Strategy": label, "Future evidence (%)": 100 * r["support"].mean(),
                     "Median plausibility rank": float(np.median(r["plaus_rank"])),
                     "Predicted gain (pp)": 100 * r["gain"].mean(), "Realised uplift (pp)": 100 * r["uplift"].mean(),
                     "Uplift 95% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "p vs KB-DSS": np.nan if k == lam_star or same else wilcoxon(ref, r["uplift"]).pvalue})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figure_framework(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.0); ax.axis("off")
    C = {"d": ("#e8f1fa", "#2b5c8a"), "k": ("#fdf0dc", "#a35f00"), "x": ("#e6f4e6", "#2e6b2e"), "o": ("#f3e8f7", "#6a3d7c")}

    def box(x, y, w, h, title, lines, kind, fs=8.6):
        fc, ec = C[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.8, weight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.58, "\n".join(lines), ha="center", va="top", fontsize=fs, linespacing=1.45, color="#222")

    def arr(p, q, c="#444", ls="-"):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=1.3, color=c, ls=ls, shrinkA=0, shrinkB=0))

    for x, t, k in [(1.2, "DATA LAYER", "d"), (4.25, "KNOWLEDGE LAYER", "k"), (8.55, "DECISION LAYER", "x")]:
        ax.text(x, 4.75, t, ha="center", fontsize=10.5, weight="bold", color=C[k][1])
    box(0.1, 2.65, 2.2, 1.8, "Venture records", ["Crunchbase export", "first funding round", "market categories",
                                                  "location, timing"], "d")
    box(0.1, 0.45, 2.2, 1.8, "Temporal protocol", ["KG \u2264 2008: fit ranker", "KG \u2264 2010: test",
                                                    "2011–2015 outcomes", "used only for scoring"], "d")
    box(2.9, 2.65, 2.7, 1.8, "Expert DT ontology", ["9 DT pillars from", "keyword patterns", "pillar overlap, lift Q(q|p)"], "k")
    box(2.9, 0.45, 2.7, 1.8, "Category knowledge graph", ["transitions P(c|s), PPMI", "SVD embeddings",
                                                         "out-of-fold success priors"], "k")
    box(6.2, 2.65, 2.3, 1.8, "(1) Option generation", ["learned ranker over", "KG + ontology features", "\u2192 plausibility"], "x")
    box(6.2, 0.45, 2.3, 1.8, "(2) Success estimation", ["gradient-boosted model", "data + knowledge", "\u2192 predicted gain"], "x")
    box(9.05, 1.05, 1.85, 2.8, "(3) Recommend", ["U = (1\u2212\u03bb)\u00b7plaus.", "+ \u03bb\u00b7gain", "",
                                                 "\u03bb* chosen on", "validation", "", "ranked DT moves", "per venture"], "o", fs=8.3)
    arr((2.3, 3.55), (2.9, 3.55)); arr((2.3, 3.0), (2.9, 1.75))
    arr((5.6, 3.55), (6.2, 3.55)); arr((5.6, 1.95), (6.2, 3.0)); arr((5.6, 1.35), (6.2, 1.35)); arr((5.6, 3.0), (6.2, 1.95))
    ax.plot([2.3, 2.6, 2.6, 5.9, 5.9], [2.8, 2.8, 0.25, 0.25, 0.9], color="#2b5c8a", lw=1.2, ls="--")
    arr((5.9, 0.9), (6.2, 0.9), c="#2b5c8a")
    arr((8.5, 3.55), (9.05, 3.1)); arr((8.5, 1.35), (9.05, 1.8))
    fig.tight_layout(); fig.savefig(path, dpi=600); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def figure_results(path, ranks, rec, pred, lam_star):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    col = {"Popularity": "#9e9e9e", "Co-occurrence": "#6baed6", "KG embedding": "#c6dbef", "Ontology ranker": "#fdae6b",
           "KG ranker": "#4292c6", "KB-DSS ranker": "#08306b"}
    ks = np.arange(1, 21)
    for name, r in ranks.items():
        a.plot(ks, [np.mean(r <= k) for k in ks], color=col[name], lw=2.4 if "KB-DSS" in name else 1.4,
               marker="o" if "KB-DSS" in name else None, ms=3, label=name)
    a.set_xlabel("k (options shown)"); a.set_ylabel("Hits@k (future move recovered)")
    a.set_title("(a) Option generation, 2011–2015 ventures", fontsize=9.5, loc="left"); a.legend(frameon=False, fontsize=7.4)
    a.set_xticks([1, 5, 10, 15, 20]); a.grid(alpha=.25)
    lam = np.array(LAMBDAS)
    real = np.array([100 * rec[l]["uplift"].mean() for l in LAMBDAS])
    ci = np.array([boot_ci(100 * rec[l]["uplift"], RNG) for l in LAMBDAS])
    predg = np.array([100 * rec[l]["gain"].mean() for l in LAMBDAS])
    b.fill_between(lam, ci[:, 0], ci[:, 1], color="#08306b", alpha=.15, lw=0)
    b.plot(lam, real, "-o", color="#08306b", ms=4, lw=2.2, label="Realised uplift, 2011–2015 (95% CI)")
    b.set_ylim(3, 9.5)
    b.set_ylabel("Realised success uplift (pp)", color="#08306b")
    bb = b.twinx()
    bb.plot(lam, predg, "--s", color="#e6550d", ms=3.5, lw=1.5, label="Model-predicted gain")
    bb.set_ylim(0, 22); bb.set_ylabel("Model-predicted gain (pp)", color="#e6550d")
    bb.spines["right"].set_visible(True); bb.tick_params(axis="y", colors="#e6550d")
    b.axvline(lam_star, color="#2e6b2e", ls=":", lw=1.5)
    b.annotate("\u03bb* = %.1f\n(validation)" % lam_star, (lam_star, 3.4), xytext=(lam_star - .36, 3.35), fontsize=7.3,
               color="#2e6b2e", arrowprops=dict(arrowstyle="-", color="#2e6b2e", lw=.8))
    b.set_xlabel("\u03bb   (0 = knowledge only,  1 = predicted success only)")
    b.set_title("(b) Knowledge as a guardrail for recommendations", fontsize=9.5, loc="left")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = bb.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.2, loc="upper left")
    b.grid(alpha=.25)
    x = np.arange(len(pred))
    c.errorbar(x - .12, pred["In-period AUC"], yerr=pred["In-period SD"], fmt="o", color="#6baed6", capsize=3, label="In-period CV (\u22642010)")
    c.plot(x + .12, pred["Hold-out AUC"], "s", color="#08306b", label="Temporal hold-out (2011–2015)")
    c.set_xticks(x); c.set_xticklabels([m.split(" ")[0] for m in pred["Model"]]); c.set_ylabel("AUC")
    c.set_title("(c) Success estimation and temporal drift", fontsize=9.5, loc="left"); c.legend(frameon=False, fontsize=7.2, loc="lower right")
    c.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=600); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    allf = load()
    kg_val = CategoryKG(allf[allf["year"] <= KG_VAL]["cats"])            # knowledge available in 2008
    kg = CategoryKG(allf[allf["year"] <= SPLIT]["cats"])                 # knowledge available in 2010

    # (1) option generation: train on 2009-2010 ventures, test on 2011–2015 ventures (all statuses)
    rng = np.random.default_rng(SEED)
    rankers = train_rankers(kg_val, hide_one(allf[(allf["year"] > KG_VAL) & (allf["year"] <= SPLIT)], kg_val, rng), rng)
    test_cases = hide_one(allf[allf["year"] > SPLIT], kg, np.random.default_rng(SEED + 1))
    t1, ranks = completion_benchmark(kg, test_cases, rankers)
    coefs = {n: dict(zip([FEAT_NAMES[i] for i in cols], m[-1].coef_[0].round(3))) for n, (cols, m) in rankers.items()}

    # (2) success estimation: labelled ventures, temporal hold-out
    lab = allf[allf["labeled"]]
    train, test = lab[lab["year"] <= SPLIT].copy(), lab[lab["year"] > SPLIT].copy()
    pred, models = prediction_ablation(train, test, kg)

    # (3) recommendation: choose lambda on the 2009-2010 validation cohort, then test once on 2011–2015
    val_train = lab[lab["year"] <= KG_VAL]
    val_cohort = lab[(lab["year"] > KG_VAL) & (lab["year"] <= SPLIT)]
    val_model = SuccessModel(val_train, kg_val, True, True, True)
    val_firms = val_cohort[val_cohort["cats"].apply(lambda c: len(kg_val.ids(c)) >= 1)]
    val_firms = val_firms.sample(min(N_REC, len(val_firms)), random_state=SEED)
    val_rankers = rankers  # ranker weights were fitted with the 2008 KG on 2009-2010 ventures
    val = recommend_eval(val_firms, val_cohort, kg_val, val_rankers["KB-DSS ranker"], val_model, rng)
    val_curve = {float(l): float(val[l]["uplift"].mean()) for l in LAMBDAS}
    lam_star = float(max(LAMBDAS, key=lambda l: (round(val_curve[float(l)], 6), -l)))
    test_firms = test[test["cats"].apply(lambda c: len(kg.ids(c)) >= 1)].sample(N_REC, random_state=SEED)
    rec = recommend_eval(test_firms, test, kg, rankers["KB-DSS ranker"], models["M5 M2 + ontology + KG"], rng)
    t2 = recommendation_table(rec, lam_star, rng)
    sweep = pd.DataFrame([{"lambda": l, "validation_uplift_pp": 100 * val_curve[float(l)],
                           "test_uplift_pp": 100 * rec[l]["uplift"].mean(), "test_predicted_gain_pp": 100 * rec[l]["gain"].mean(),
                           "test_future_evidence_pct": 100 * rec[l]["support"].mean()} for l in LAMBDAS])
    sweep.to_csv(f"{OUT}/lambda_sweep.csv", index=False)
    n_rec = len(test_firms)

    t1.to_csv(f"{OUT}/table1_option_generation.csv", index=False)
    t2.to_csv(f"{OUT}/table2_recommendation.csv", index=False)
    pred.to_csv(f"{OUT}/prediction_ablation.csv", index=False)
    summary = {"ventures_total": len(allf), "kg_2008": {"nodes": len(kg_val.nodes), "edges": kg_val.n_edges},
               "kg_2010": {"nodes": len(kg.nodes), "edges": kg.n_edges,
                           "source_ventures": int((allf["year"] <= SPLIT).sum()),
                           "ontology_coverage": float(kg.Pi.any(1).mean())},
               "ranker_train_cases": int(sum(1 for _ in hide_one(allf[(allf["year"] > KG_VAL) & (allf["year"] <= SPLIT)], kg_val, np.random.default_rng(SEED)))),
               "completion_test_cases": len(test_cases), "labelled_train": len(train), "labelled_test": len(test),
               "success_rate_train": float(train["y"].mean()), "success_rate_test": float(test["y"].mean()),
               "recommendation_ventures": n_rec, "lambda_star": lam_star, "ranker_coefficients": coefs}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    figure_framework(f"{OUT}/Figure_1.png")
    figure_results(f"{OUT}/Figure_2.png", ranks, rec, pred, lam_star)
    pd.set_option("display.width", 220)
    for t in (t1, t2, pred, sweep):
        print(t.round(4).to_string(index=False), "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
