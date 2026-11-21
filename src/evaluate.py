"""
evaluate.py - Offline evaluation and charts for the Fitness Workout Recommender.

Protocol
--------
* For every user, hide 3 random ratings (≈20% of each user's ~15 ratings) as the
  test set; everything else is the training set.
* Each method ranks all workouts the user has NOT rated in training; we take Top-5.
* Relevant item = hidden workout with rating >= 4.
* Ranking metrics: Precision@5, Recall@5, HitRate@5 (users with >= 1 relevant item).
* Rating metrics : RMSE and MAE on the hidden ratings.
* The whole split is repeated with 5 random seeds and averaged (mean ± std),
  because one split of 42 test ratings is very noisy.

Charts are saved in results/.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp_pipeline import FitnessNLP, note_sentiment
from recommender import FitnessRecommender

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
METHODS = ["Popularity", "Content (NLP)", "User-CF", "Hybrid"]
COLORS = {"Popularity": "#9aa5b1", "Content (NLP)": "#2f7ed8", "User-CF": "#e8743b", "Hybrid": "#19a979"}


def load_data():
    return pd.read_csv(os.path.join(DATA, "items.csv")), pd.read_csv(os.path.join(DATA, "ratings.csv"))


def train_test_split_per_user(ratings: pd.DataFrame, n_test: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in ratings.groupby("user_id"):
        test_idx.extend(rng.choice(g.index.values, size=min(n_test, len(g) - 2), replace=False))
    test = ratings.loc[test_idx]
    return ratings.drop(index=test_idx), test


def rank_items(rec: FitnessRecommender, method: str, user: str, k: int = 5):
    if method == "Popularity":
        df = rec.popularity(user, k=k)
    elif method == "Content (NLP)":
        df = rec.content_for_user(user, k=k)
    elif method == "User-CF":
        df = rec.cf_for_user(user, k=k)
        if len(df) < k:                                    # pad with popularity if CF has no evidence
            pop = rec.popularity(user, k=len(rec.item_ids))
            extra = [i for i in pop["item_id"] if i not in set(df.get("item_id", []))]
            return list(df.get("item_id", [])) + extra[: k - len(df)]
    else:
        df = rec.hybrid_for_user(user, k=k)
    return list(df["item_id"]) if len(df) else []


def predict(rec: FitnessRecommender, method: str, user: str, item: str) -> float:
    return {"Popularity": rec.predict_popularity, "Content (NLP)": rec.predict_content,
            "User-CF": rec.predict_cf, "Hybrid": rec.predict_hybrid}[method](user, item)


def evaluate_split(items, ratings, nlp, seed=42, k=5, threshold=4):
    train, test = train_test_split_per_user(ratings, seed=seed)
    rec = FitnessRecommender(items, train, nlp=nlp)
    out = []
    for m in METHODS:
        P, R, H, err = [], [], [], []
        for u, g in test.groupby("user_id"):
            rel = set(g.loc[g["rating"] >= threshold, "item_id"])
            for _, row in g.iterrows():
                err.append(predict(rec, m, u, row["item_id"]) - row["rating"])
            if not rel:
                continue
            top = rank_items(rec, m, u, k)
            hits = len(rel & set(top))
            P.append(hits / k); R.append(hits / len(rel)); H.append(hits > 0)
        err = np.array(err)
        out.append({"method": m, "seed": seed, "Precision@5": np.mean(P), "Recall@5": np.mean(R),
                    "HitRate@5": np.mean(H), "RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err))})
    # Upper bound for P@5: at most |relevant| of 5 slots can be hits.
    ub = np.mean([min(len(set(g.loc[g.rating >= threshold, "item_id"])), k) / k
                  for _, g in test.groupby("user_id") if (g.rating >= threshold).any()])
    return pd.DataFrame(out), ub


def run_evaluation(seeds=(42, 7, 123, 2024, 99), verbose=True):
    items, ratings = load_data()
    nlp = FitnessNLP(items)
    frames, ubs = [], []
    for s in seeds:
        df, ub = evaluate_split(items, ratings, nlp, seed=s)
        frames.append(df); ubs.append(ub)
    allr = pd.concat(frames)
    metrics = ["Precision@5", "Recall@5", "HitRate@5", "RMSE", "MAE"]
    summary = allr.groupby("method")[metrics].agg(["mean", "std"]).reindex(METHODS)
    os.makedirs(RESULTS, exist_ok=True)
    allr.to_csv(os.path.join(RESULTS, "evaluation_per_seed.csv"), index=False)
    flat = pd.DataFrame({mt: summary[(mt, "mean")].round(3).astype(str) + " ± " + summary[(mt, "std")].round(3).astype(str)
                         for mt in metrics})
    flat.to_csv(os.path.join(RESULTS, "evaluation_results.csv"))
    if verbose:
        print(f"Evaluation over {len(seeds)} random splits (3 hidden ratings per user, relevant = rating >= 4)")
        print(flat.to_string())
        print(f"Oracle upper bound for Precision@5: {np.mean(ubs):.3f}")
    return summary, np.mean(ubs)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def plot_data_overview(items, ratings, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rc = ratings["rating"].value_counts().sort_index()
    ax[0].bar(rc.index, rc.values, color="#2f7ed8")
    ax[0].set_title("Rating distribution (210 ratings)"); ax[0].set_xlabel("rating"); ax[0].set_ylabel("count")
    for x, v in zip(rc.index, rc.values):
        ax[0].text(x, v + 1, str(v), ha="center", fontsize=9)
    dc = items["difficulty"].value_counts().reindex(["Beginner", "Intermediate", "Advanced"])
    ax[1].bar(dc.index, dc.values, color=["#19a979", "#f2b134", "#e8743b"])
    ax[1].set_title("Workouts by difficulty"); ax[1].set_ylabel("workouts")
    cats = {"cardio": "Cardio", "strength": "Strength", "core": "Core", "stretching": "Stretching",
            "flexibility": "Flexibility", "yoga": "Yoga", "hiit": "HIIT", "mobility": "Mobility",
            "recovery": "Recovery", "bodyweight": "No equipment"}
    tags = items["target_muscles_tags"].str.split(";").explode()
    tc = tags[tags.isin(cats)].map(cats).value_counts().sort_values()
    ax[2].barh(tc.index, tc.values, color="#9aa5b1")
    ax[2].set_title("Workouts per key tag"); ax[2].set_xlabel("workouts")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_user_item_matrix(items, ratings, path):
    R = ratings.pivot_table(index="user_id", columns="item_id", values="rating").reindex(columns=items["item_id"])
    fig, ax = plt.subplots(figsize=(14, 4.6))
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(R.values), cmap=cmap, vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(R.shape[1])); ax.set_xticklabels(R.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(R.shape[0])); ax.set_yticklabels(R.index, fontsize=8)
    sp = 1 - R.notna().values.mean()
    ax.set_title(f"User-item rating matrix ({R.shape[0]} users x {R.shape[1]} workouts, {sp:.1%} missing)")
    fig.colorbar(im, ax=ax, label="rating", fraction=0.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_evaluation(summary, ub, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    rank_m, err_m = ["Precision@5", "Recall@5", "HitRate@5"], ["RMSE", "MAE"]
    for a, mets, title in [(ax[0], rank_m, "Top-5 ranking quality (higher is better)"),
                           (ax[1], err_m, "Rating prediction error (lower is better)")]:
        x = np.arange(len(mets)); w = 0.2
        for n, m in enumerate(METHODS):
            means = [summary.loc[m, (mt, "mean")] for mt in mets]
            stds = [summary.loc[m, (mt, "std")] for mt in mets]
            bars = a.bar(x + (n - 1.5) * w, means, w, yerr=stds, capsize=3, label=m, color=COLORS[m])
            for b, v, sd in zip(bars, means, stds):
                a.text(b.get_x() + b.get_width() / 2, v + sd + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(x); a.set_xticklabels(mets); a.set_title(title)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].axhline(ub, ls="--", color="grey", lw=1)
    ax[0].text(-0.45, ub + 0.01, f"P@5 upper bound {ub:.2f}", fontsize=8, color="grey")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_ylim(0, 1.15); ax[1].set_ylim(0, 1.65)
    fig.suptitle("Offline evaluation – mean ± std over 5 random splits", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_sentiment_vs_rating(ratings, path):
    df = ratings.assign(sentiment=ratings["workout_note"].map(note_sentiment))
    g = df.groupby("rating")["sentiment"].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(g.index, g.values, color=["#e8743b" if v < 0 else "#19a979" for v in g.values])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("rating"); ax.set_ylabel("mean note sentiment")
    corr = df["rating"].corr(df["sentiment"])
    ax.set_title(f"Lexicon sentiment of workout notes vs rating (r = {corr:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return corr


def make_charts(summary=None, ub=None):
    items, ratings = load_data()
    os.makedirs(RESULTS, exist_ok=True)
    plot_data_overview(items, ratings, os.path.join(RESULTS, "data_overview.png"))
    plot_user_item_matrix(items, ratings, os.path.join(RESULTS, "user_item_matrix.png"))
    corr = plot_sentiment_vs_rating(ratings, os.path.join(RESULTS, "sentiment_vs_rating.png"))
    if summary is None:
        summary, ub = run_evaluation(verbose=False)
    plot_evaluation(summary, ub, os.path.join(RESULTS, "evaluation_comparison.png"))
    print(f"Charts saved to {RESULTS} (note sentiment vs rating correlation r = {corr:.2f})")


if __name__ == "__main__":
    s, ub = run_evaluation()
    make_charts(s, ub)
