"""
recommender.py - Recommendation engines for the Fitness Workout Recommender.

1. Popularity baseline     : Bayesian-weighted average rating, min-count threshold.
2. NLP content-based       : cosine similarity over TF-IDF vectors
                             (a) item-to-item, (b) user profile, (c) free-text query.
3. User-based CF (extension): mean-centred cosine similarity between users
                             (>= 3 co-rated workouts), k = 5 nearest neighbours,
                             significance weighting and damping towards the user mean.
4. Hybrid (bonus)          : 0.5 * normalised CF prediction + 0.5 * content score.
5. Cold start              : query-based content score blended with popularity.

All methods return a pandas DataFrame with item_id, workout_name, score and a
short "why" explanation (matching keywords / neighbours / popularity).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import FitnessNLP, extract_constraints


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


class FitnessRecommender:
    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame, min_count: int = 3,
                 n_neighbors: int = 5, min_overlap: int = 3, damping: float = 1.0, nlp: Optional[FitnessNLP] = None):
        self.items = items.reset_index(drop=True)
        self.ratings = ratings.reset_index(drop=True)
        self.min_count = min_count
        self.n_neighbors = n_neighbors
        self.min_overlap = min_overlap
        self.damping = damping          # shrinks CF towards the user mean when evidence is weak
        # NLP is fitted on item text only, so it can be shared between train/test splits.
        self.nlp = nlp or FitnessNLP(self.items)
        self.item_ids = self.nlp.item_ids
        self.idx = self.nlp.index
        self.names = dict(zip(self.items["item_id"], self.items["workout_name"]))
        self.item_sim = cosine_similarity(self.nlp.item_matrix)          # item-item TF-IDF cosine
        self._fit_ratings()

    # ------------------------------------------------------------------
    def _fit_ratings(self):
        r = self.ratings
        self.global_mean = r["rating"].mean()
        self.R = r.pivot_table(index="user_id", columns="item_id", values="rating").reindex(columns=self.item_ids)
        self.user_mean = self.R.mean(axis=1)
        stats = r.groupby("item_id")["rating"].agg(["mean", "count"]).reindex(self.item_ids)
        stats["count"] = stats["count"].fillna(0)
        m = self.min_count
        stats["bayes"] = ((stats["count"] * stats["mean"].fillna(0) + m * self.global_mean)
                          / (stats["count"] + m))
        self.item_stats = stats
        self.profiles = self.nlp.build_all_profiles(r)
        self.user_sim = self._user_similarity()

    def _user_similarity(self) -> pd.DataFrame:
        """Mean-centred cosine (Pearson-like) on co-rated items, with significance weighting."""
        centred = self.R.sub(self.user_mean, axis=0)
        users = self.R.index
        S = pd.DataFrame(0.0, index=users, columns=users)
        mask = self.R.notna().values
        C = centred.fillna(0).values
        for a in range(len(users)):
            for b in range(a + 1, len(users)):
                common = mask[a] & mask[b]
                n = common.sum()
                if n < self.min_overlap:
                    continue
                x, y = C[a, common], C[b, common]
                den = np.linalg.norm(x) * np.linalg.norm(y)
                if den == 0:
                    continue
                s = float(x @ y / den) * min(n, 5) / 5      # shrink sims built on few items
                S.iat[a, b] = S.iat[b, a] = s
        return S

    def rated_items(self, user_id: str) -> set:
        return set(self.ratings.loc[self.ratings["user_id"] == user_id, "item_id"])

    def _frame(self, rows: List[dict], k: int) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        df.index = df.index + 1
        return df

    def _meta(self, iid: str) -> dict:
        it = self.items.iloc[self.idx[iid]]
        return {"item_id": iid, "workout_name": it["workout_name"], "difficulty": it["difficulty"],
                "duration_min": int(it["duration_min"])}

    # ------------------------------------------------------------------
    # 1. Popularity baseline
    # ------------------------------------------------------------------
    def popularity(self, user_id: Optional[str] = None, k: int = 5) -> pd.DataFrame:
        seen = self.rated_items(user_id) if user_id else set()
        rows = []
        for iid, s in self.item_stats.iterrows():
            if s["count"] < self.min_count or iid in seen:
                continue
            rows.append({**self._meta(iid), "score": round(s["bayes"], 3),
                         "why": f"avg {s['mean']:.2f} from {int(s['count'])} ratings"})
        return self._frame(rows, k)

    def predict_popularity(self, user_id: str, item_id: str) -> float:
        return float(self.item_stats.loc[item_id, "bayes"])

    # ------------------------------------------------------------------
    # 2. NLP content-based
    # ------------------------------------------------------------------
    def similar_items(self, item_id: str, k: int = 5) -> pd.DataFrame:
        i = self.idx[item_id]
        qv = self.nlp.item_matrix[i]
        rows = []
        for j, iid in enumerate(self.item_ids):
            if j == i:
                continue
            rows.append({**self._meta(iid), "score": round(float(self.item_sim[i, j]), 3),
                         "why": ", ".join(self.nlp.top_keywords(qv, j))})
        return self._frame(rows, k)

    def content_for_user(self, user_id: str, k: int = 5) -> pd.DataFrame:
        prof = self.profiles.get(user_id)
        if prof is None:
            return pd.DataFrame()
        sims = self.nlp.item_matrix @ prof
        seen = self.rated_items(user_id)
        rows = [{**self._meta(iid), "score": round(float(sims[j]), 3),
                 "why": ", ".join(self.nlp.top_keywords(prof, j))}
                for j, iid in enumerate(self.item_ids) if iid not in seen]
        return self._frame(rows, k)

    def content_scores_user(self, user_id: str) -> np.ndarray:
        prof = self.profiles.get(user_id)
        return np.zeros(len(self.item_ids)) if prof is None else self.nlp.item_matrix @ prof

    def content_for_query(self, query: str, k: int = 5, exclude: Optional[set] = None,
                          use_constraints: bool = True) -> pd.DataFrame:
        """Free-text query -> cosine(query, item); optional duration/difficulty fit bonus.

        final = 0.8*cosine + 0.1*duration_fit + 0.1*difficulty_fit  (only when the
        query mentions a duration / level; otherwise final = cosine).
        """
        qv = self.nlp.transform(query)
        cos = cosine_similarity(qv, self.nlp.item_matrix).ravel()
        cons = extract_constraints(query) if use_constraints else {"duration": None, "difficulty": None}
        exclude = exclude or set()
        rows = []
        for j, iid in enumerate(self.item_ids):
            if iid in exclude:
                continue
            it = self.items.iloc[j]
            score, extra = cos[j], []
            if cons["duration"] or cons["difficulty"]:
                score = 0.8 * cos[j]
                if cons["duration"]:
                    t = cons["duration"]
                    fit = max(0.0, 1 - abs(it["duration_min"] - t) / t)
                    score += 0.1 * fit
                    if fit >= 0.75:
                        extra.append(f"{it['duration_min']} min")
                if cons["difficulty"]:
                    fit = 1.0 if it["difficulty"] == cons["difficulty"] else 0.0
                    score += 0.1 * fit
                    if fit:
                        extra.append(it["difficulty"].lower())
            kws = self.nlp.top_keywords(qv, j)
            covered = {w for kw in kws for w in kw.split()}
            extra = [e for e in extra if e not in covered]
            rows.append({**self._meta(iid), "cosine": round(float(cos[j]), 3), "score": round(float(score), 3),
                         "why": ", ".join(kws + extra)})
        return self._frame(rows, k)

    def predict_content(self, user_id: str, item_id: str, k: int = 5) -> float:
        """Item-kNN rating prediction using TF-IDF similarity to the user's rated workouts."""
        ur = self.ratings[self.ratings["user_id"] == user_id]
        mu = self.user_mean.get(user_id, self.global_mean)
        if ur.empty:
            return float(self.global_mean)
        j = self.idx[item_id]
        sims = np.array([self.item_sim[j, self.idx[i]] for i in ur["item_id"]])
        order = np.argsort(-sims)[:k]
        s, r = sims[order], ur["rating"].values[order]
        if s.sum() <= 0:
            return float(mu)
        return float(np.clip(mu + (s @ (r - mu)) / s.sum(), 1, 5))

    # ------------------------------------------------------------------
    # 3. User-based collaborative filtering
    # ------------------------------------------------------------------
    def predict_cf(self, user_id: str, item_id: str, return_neighbors: bool = False):
        mu = self.user_mean.get(user_id, self.global_mean)
        if user_id not in self.user_sim.index:
            return (float(self.global_mean), []) if return_neighbors else float(self.global_mean)
        raters = self.R[item_id].dropna()
        raters = raters[raters.index != user_id]
        sims = self.user_sim.loc[user_id, raters.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(self.n_neighbors)
        if sims.empty:
            pred, nbrs = float(mu), []
        else:
            dev = raters[sims.index] - self.user_mean[sims.index]
            pred = float(np.clip(mu + (sims @ dev) / (sims.abs().sum() + self.damping), 1, 5))
            nbrs = [f"{u}({int(raters[u])}*)" for u in sims.index[:3]]
        return (pred, nbrs) if return_neighbors else pred

    def cf_for_user(self, user_id: str, k: int = 5) -> pd.DataFrame:
        seen = self.rated_items(user_id)
        rows = []
        for iid in self.item_ids:
            if iid in seen:
                continue
            pred, nbrs = self.predict_cf(user_id, iid, return_neighbors=True)
            if not nbrs:          # no similar user has rated it -> no CF evidence
                continue
            rows.append({**self._meta(iid), "score": round(pred, 3),
                         "why": "liked by similar users " + ", ".join(nbrs)})
        return self._frame(rows, k)

    def top_neighbors(self, user_id: str, n: int = 3) -> pd.Series:
        return self.user_sim.loc[user_id].drop(user_id).sort_values(ascending=False).head(n)

    # ------------------------------------------------------------------
    # 4. Hybrid (CF + content)
    # ------------------------------------------------------------------
    def hybrid_for_user(self, user_id: str, k: int = 5, alpha: float = 0.5) -> pd.DataFrame:
        seen = self.rated_items(user_id)
        cand = [iid for iid in self.item_ids if iid not in seen]
        cf = np.array([self.predict_cf(user_id, i) for i in cand])
        cs = self.content_scores_user(user_id)[[self.idx[i] for i in cand]]
        score = alpha * _minmax(cf) + (1 - alpha) * _minmax(cs)
        prof = self.profiles.get(user_id)
        rows = [{**self._meta(iid), "score": round(float(score[n]), 3),
                 "why": f"CF {cf[n]:.2f}*, content {cs[n]:.2f} ({', '.join(self.nlp.top_keywords(prof, self.idx[iid], 2))})"}
                for n, iid in enumerate(cand)]
        return self._frame(rows, k)

    def predict_hybrid(self, user_id: str, item_id: str, alpha: float = 0.5) -> float:
        return alpha * self.predict_cf(user_id, item_id) + (1 - alpha) * self.predict_content(user_id, item_id)

    # ------------------------------------------------------------------
    # 5. Cold start
    # ------------------------------------------------------------------
    def cold_start(self, goal: str, k: int = 5, beta: float = 0.15) -> pd.DataFrame:
        """New user with only a 1-sentence goal: query score blended with popularity prior."""
        q = self.content_for_query(goal, k=len(self.item_ids))
        pop = self.item_stats.loc[q["item_id"], "bayes"].values
        pop_n = (pop - 1) / 4                                    # rating scale 1-5 -> 0-1
        q["popularity"] = np.round(pop, 2)
        q["score"] = np.round((1 - beta) * q["score"].values + beta * pop_n, 3)
        q["why"] = q["why"] + q["popularity"].map(lambda p: f" | pop {p:.2f}")
        return self._frame(q.drop(columns=["cosine"]).to_dict("records"), k)
