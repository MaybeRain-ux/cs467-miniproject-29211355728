"""
nlp_pipeline.py - Text preprocessing and TF-IDF feature extraction for the
Fitness Workout and Training Plan Recommender (CS467 MP-A16).

Main pieces
-----------
1. Text preprocessing: lowercasing, punctuation/extra-space removal,
   fitness phrase normalisation ("no equipment" -> "bodyweight"),
   tokenisation, stop-word removal, light lemmatisation and synonym mapping.
2. Item documents: workout name + description + muscle/equipment tags
   (tags repeated so they carry more weight) + difficulty level.
3. TF-IDF vectoriser fitted on the item documents (unigrams + bigrams).
4. Query vectorisation helper that reuses the SAME fitted vocabulary.
5. Rating-weighted user fitness profiles built from completed workouts
   (weights also nudged by a small lexicon-based sentiment of the note).
6. Explanation helpers: top matching keywords, constraint parsing
   (duration / difficulty) and note sentiment.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

# --------------------------------------------------------------------------
# 1. Text preprocessing
# --------------------------------------------------------------------------

# Multi-word fitness expressions are collapsed into one token BEFORE
# stop-word removal; otherwise "no equipment" would lose the word "no".
PHRASES = [
    (r"\b(?:no|zero|without(?: any)?)[\s-]+(?:equipment|gear|weights?)\b", " bodyweight "),
    (r"\bequipment[\s-]+free\b", " bodyweight "),
    (r"\bbody[\s-]?weight\b", " bodyweight "),
    (r"\blow[\s-]+impact\b", " low_impact "),
    (r"\bfull[\s-]+body\b", " full_body "),
    (r"\btotal[\s-]+body\b", " full_body "),
    (r"\bwhole[\s-]+body\b", " full_body "),
    (r"\bupper[\s-]+body\b", " upper_body "),
    (r"\blower[\s-]+body\b", " lower_body "),
    (r"\blower[\s-]+back\b", " lower_back "),
    (r"\bupper[\s-]+back\b", " upper_back "),
    (r"\bsix[\s-]+pack\b", " abs "),
    (r"\bjump(?:ing)?[\s-]+rope\b", " jump_rope "),
    (r"\bpull[\s-]+up[\s-]+bar\b", " pull_up_bar "),
    (r"\bresistance[\s-]+bands?\b", " resistance_band "),
    (r"\bfoam[\s-]+roll(?:er|ing)?\b", " foam_roller "),
    (r"\bmedicine[\s-]+ball\b", " medicine_ball "),
    (r"\b(?:stationary|exercise|spin)[\s-]+bike\b", " stationary_bike "),
    (r"\bindoor[\s-]+cycling\b", " stationary_bike "),
    (r"\browing[\s-]+machine\b", " rowing_machine "),
    (r"\bliving[\s-]+room\b", " home "),
    (r"\bat[\s-]+home\b", " home "),
    (r"\bhigh[\s-]+intensity\b", " hiit "),
    (r"\bwork[\s-]+out\b", " workout "),
]

# Single-word synonyms -> canonical token (applied after lemmatisation).
SYNONYMS = {
    "ab": "abs", "abdominal": "abs", "tummy": "core", "belly": "core", "midsection": "core",
    "stretch": "stretching", "stretched": "stretching", "stretchy": "stretching",
    "flexible": "flexibility", "limber": "flexibility",
    "jog": "running", "jogging": "running", "run": "running", "runner": "running",
    "cardiovascular": "cardio", "aerobic": "cardio",
    "tabata": "hiit", "interval": "hiit",
    "relax": "relaxation", "relaxing": "relaxation", "relaxed": "relaxation", "calm": "relaxation",
    "apartment": "home", "house": "home", "indoors": "home",
    "dumbbell": "dumbbells", "kettlebells": "kettlebell",
    "bike": "stationary_bike", "cycling": "stationary_bike",
    "lifting": "strength", "strong": "strength", "strengthen": "strength",
    "toning": "strength", "tone": "strength", "muscle": "strength",
    "mobile": "mobility", "recover": "recovery",
    "short": "quick", "fast": "quick", "busy": "quick",
    "beginners": "beginner", "newbie": "beginner", "novice": "beginner",
}

# Generic words that appear in almost every request/description.
DOMAIN_STOPWORDS = {
    "workout", "workouts", "routine", "routines", "session", "exercise", "exercises",
    "want", "need", "like", "looking", "help", "recommend", "something", "good", "great",
    "minute", "minutes", "min", "mins", "hour", "hours", "day", "days", "week",
    "plus", "using", "done", "set", "sets", "make", "build", "improve", "better",
    "really", "just", "im", "ive", "id", "get", "try",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS

_KEEP_PLURAL = {"abs", "pilates", "glutes", "quads", "hamstrings", "calves", "biceps",
                "triceps", "obliques", "dumbbells", "plus", "focus", "stress", "fitness"}


def clean_text(text: str) -> str:
    """Lowercase, normalise fitness phrases, strip punctuation and extra spaces."""
    text = str(text).lower()
    text = text.replace(";", " ").replace("/", " ")
    for pattern, repl in PHRASES:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)   # keep underscores of joined tags
    text = re.sub(r"\b\d+\b", " ", text)          # bare numbers carry no meaning here
    return re.sub(r"\s+", " ", text).strip()


def lemmatize(token: str) -> str:
    """Very small rule-based lemmatiser (plural -> singular)."""
    if token in _KEEP_PLURAL or "_" in token or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def preprocess(text: str) -> List[str]:
    """Full pipeline: clean -> tokenize -> stop-words -> lemmatise -> synonyms."""
    tokens = []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        tok = lemmatize(tok)
        tok = SYNONYMS.get(tok, tok)
        if tok in STOPWORDS or len(tok) < 2:
            continue
        tokens.append(tok)
    return tokens


def preprocess_to_string(text: str) -> str:
    return " ".join(preprocess(text))


# --------------------------------------------------------------------------
# Constraint parsing + sentiment helpers
# --------------------------------------------------------------------------

def extract_constraints(query: str) -> Dict[str, Optional[object]]:
    """Pull a target duration (minutes) and difficulty level out of free text."""
    q = str(query).lower()
    duration = None
    m = re.search(r"(\d{1,3})\s*-?\s*(?:minutes?|mins?|m)\b", q)
    if m:
        duration = int(m.group(1))
    elif re.search(r"\bhalf (?:an )?hour\b", q):
        duration = 30
    elif re.search(r"\b(?:an|one|1) hour\b", q):
        duration = 60
    elif re.search(r"\b(?:quick|short|busy|lunch break)\b", q):
        duration = 15

    difficulty = None
    if re.search(r"\b(beginner|beginners|easy|gentle|new to|first time|novice|senior)\b", q):
        difficulty = "Beginner"
    elif re.search(r"\b(advanced|hard|intense|expert|challenging|athlete)\b", q):
        difficulty = "Advanced"
    elif re.search(r"\b(intermediate|moderate)\b", q):
        difficulty = "Intermediate"
    return {"duration": duration, "difficulty": difficulty}


POSITIVE_WORDS = {"love", "loved", "amazing", "perfect", "best", "great", "awesome", "enjoyed",
                  "good", "nice", "effective", "fun", "energized", "solid", "repeat", "spot"}
NEGATIVE_WORDS = {"hated", "boring", "terrible", "painful", "struggled", "meh", "not", "didn",
                  "too", "skipped", "won", "uneven", "repetitive", "average"}


def note_sentiment(note: str) -> float:
    """Lexicon sentiment of a short workout note in [-1, 1]."""
    toks = clean_text(note).split()
    if not toks:
        return 0.0
    pos = sum(t in POSITIVE_WORDS for t in toks)
    neg = sum(t in NEGATIVE_WORDS for t in toks)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


# --------------------------------------------------------------------------
# 2-6. TF-IDF model, query vectorisation, user profiles, keywords
# --------------------------------------------------------------------------

def build_item_document(row: pd.Series, tag_weight: int = 2) -> str:
    tags = str(row["target_muscles_tags"]).replace(";", " ")
    parts = [row["workout_name"], row["description"], (tags + " ") * tag_weight,
             str(row["difficulty"]).lower()]
    return " ".join(parts)


class FitnessNLP:
    """Fits TF-IDF on workout documents and offers all NLP helpers."""

    def __init__(self, items: pd.DataFrame, ngram_range=(1, 2), tag_weight: int = 2):
        self.items = items.reset_index(drop=True)
        self.item_ids: List[str] = self.items["item_id"].tolist()
        self.index = {iid: i for i, iid in enumerate(self.item_ids)}
        self.raw_docs = self.items.apply(build_item_document, axis=1, tag_weight=tag_weight)
        self.clean_docs = self.raw_docs.apply(preprocess_to_string)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9_]+", ngram_range=ngram_range,
                                          sublinear_tf=True, min_df=1)
        self.item_matrix = self.vectorizer.fit_transform(self.clean_docs)   # L2-normalised rows
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    # ---- query vectorisation -------------------------------------------
    def transform(self, texts) -> "np.ndarray":
        """Vectorise new free-text queries / descriptions with the fitted vocabulary."""
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform([preprocess_to_string(t) for t in texts])

    # ---- user profiles -------------------------------------------------
    def build_user_profile(self, user_ratings: pd.DataFrame, neutral: float = 3.0,
                           sentiment_weight: float = 0.5) -> np.ndarray:
        """Rating-weighted TF-IDF profile.

        weight_i = (rating_i - neutral) + sentiment_weight * sentiment(note_i)
        profile  = normalise( max(0, sum_i weight_i * tfidf(item_i)) )
        Workouts rated below neutral push the profile AWAY from their terms.
        """
        vec = np.zeros(self.item_matrix.shape[1])
        for _, r in user_ratings.iterrows():
            if r["item_id"] not in self.index:
                continue
            w = (r["rating"] - neutral)
            if sentiment_weight and "workout_note" in r and pd.notna(r["workout_note"]):
                w += sentiment_weight * note_sentiment(r["workout_note"])
            vec += w * self.item_matrix[self.index[r["item_id"]]].toarray().ravel()
        vec = np.clip(vec, 0, None)
        if not vec.any():   # user only rated things low -> fall back to plain rating weights
            for _, r in user_ratings.iterrows():
                if r["item_id"] in self.index:
                    vec += r["rating"] * self.item_matrix[self.index[r["item_id"]]].toarray().ravel()
        return normalize(vec.reshape(1, -1)).ravel()

    def build_all_profiles(self, ratings: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {u: self.build_user_profile(g) for u, g in ratings.groupby("user_id")}

    # ---- explanations --------------------------------------------------
    def top_keywords(self, query_vec, item_idx: int, k: int = 3) -> List[str]:
        """Terms that contribute most to cosine(query, item)."""
        q = query_vec.toarray().ravel() if hasattr(query_vec, "toarray") else np.asarray(query_vec).ravel()
        contrib = q * self.item_matrix[item_idx].toarray().ravel()
        out, covered = [], set()
        for j in np.argsort(-contrib):
            if contrib[j] <= 0 or len(out) >= k:
                break
            words = set(self.feature_names[j].split())
            if words <= covered:          # skip "cardio" when "home cardio" already listed
                continue
            out.append(self.feature_names[j])
            covered |= words
        return out

    def top_terms(self, vec, k: int = 8) -> List[str]:
        v = vec.toarray().ravel() if hasattr(vec, "toarray") else np.asarray(vec).ravel()
        order = np.argsort(-v)
        return [self.feature_names[j] for j in order[:k] if v[j] > 0]


if __name__ == "__main__":
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    items = pd.read_csv(os.path.join(base, "items.csv"))
    nlp = FitnessNLP(items)
    q = "20 minute home cardio workout with no equipment"
    print("Query      :", q)
    print("Tokens     :", preprocess(q))
    print("Constraints:", extract_constraints(q))
    print("TF-IDF matrix:", nlp.item_matrix.shape)
    print("W23 doc    :", nlp.clean_docs[nlp.index['W23']])
