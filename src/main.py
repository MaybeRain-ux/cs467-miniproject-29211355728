"""
main.py - CLI demo for the Fitness Workout and Training Plan Recommender (MP-A16).

Examples
--------
python src/main.py                         # run the 3 required test cases (+ save results/sample_outputs.txt)
python src/main.py --query "30 min beginner yoga for flexibility"
python src/main.py --user U05 --method all
python src/main.py --item W23              # workouts similar to a given workout
python src/main.py --cold "I want to get stronger at home with dumbbells"
python src/main.py --evaluate              # offline evaluation + charts
python src/main.py --interactive           # type queries / user IDs in a loop
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlp_pipeline import extract_constraints, preprocess  # noqa: E402
from recommender import FitnessRecommender  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

TEST_QUERY = "20 minute home cardio workout with no equipment"
TEST_USER = "U03"   # rated core, pilates, yoga and stretching routines 4-5 stars
TEST_COLD_GOAL = "I'm a beginner with a stiff lower back and I want to get more flexible."


def load_recommender() -> FitnessRecommender:
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
    return FitnessRecommender(items, ratings)


def show(df: pd.DataFrame, title: str, score_label: str = "score"):
    print(f"\n  {title}")
    if df is None or df.empty:
        print("    (no recommendations)")
        return
    print(f"    {'#':<2} {'ID':<4} {'Workout':<32} {'Level':<12} {'Min':>3}  {score_label:>7}  Why (top keywords / evidence)")
    print("    " + "-" * 110)
    for rank, r in df.iterrows():
        print(f"    {rank:<2} {r['item_id']:<4} {r['workout_name'][:32]:<32} {r['difficulty']:<12} "
              f"{r['duration_min']:>3}  {r['score']:>7.3f}  {r['why']}")


def banner(text: str):
    print("\n" + "=" * 116)
    print(text)
    print("=" * 116)


# ---------------------------------------------------------------------------
def run_query(rec, query, k=5):
    print(f"\n  Query       : \"{query}\"")
    print(f"  NLP tokens  : {preprocess(query)}")
    c = extract_constraints(query)
    print(f"  Constraints : duration = {c['duration']} min, difficulty = {c['difficulty']}")
    print("  Score       : cosine(query TF-IDF, workout TF-IDF)"
          + ("  ->  0.8*cosine + 0.1*duration_fit + 0.1*level_fit" if (c['duration'] or c['difficulty']) else ""))
    show(rec.content_for_query(query, k=k), "Top-5 NLP content-based matches")


def run_user(rec, user, method="all", k=5):
    if user not in rec.user_mean.index:
        print(f"  Unknown user '{user}'. Known users: {', '.join(rec.user_mean.index)}")
        return
    hist = rec.ratings[rec.ratings.user_id == user].merge(rec.items[["item_id", "workout_name", "target_muscles_tags"]])
    hist = hist.sort_values("rating", ascending=False)
    print(f"\n  User {user}: {len(hist)} rated workouts, mean rating {rec.user_mean[user]:.2f}")
    print("  Highest-rated history:")
    for _, h in hist.head(6).iterrows():
        print(f"    {h['rating']}*  {h['item_id']} {h['workout_name']:<32} [{h['target_muscles_tags']}]  \"{h['workout_note']}\"")
    print("  Lowest-rated:")
    for _, h in hist.tail(3).iterrows():
        print(f"    {h['rating']}*  {h['item_id']} {h['workout_name']:<32} [{h['target_muscles_tags']}]")
    print(f"  Profile top terms (rating-weighted TF-IDF): {', '.join(rec.nlp.top_terms(rec.profiles[user], 8))}")
    nb = rec.top_neighbors(user)
    print(f"  Most similar users (CF): {', '.join(f'{u} ({s:.2f})' for u, s in nb.items())}")
    if method in ("all", "content"):
        show(rec.content_for_user(user, k), "NLP content-based (cosine with user profile)", "cosine")
    if method in ("all", "cf"):
        show(rec.cf_for_user(user, k), "User-based collaborative filtering (predicted rating)", "pred")
    if method in ("all", "hybrid"):
        show(rec.hybrid_for_user(user, k), "Hybrid 0.5*CF + 0.5*content (normalised)", "hybrid")
    if method in ("all", "popularity"):
        show(rec.popularity(user, k), "Popularity baseline (Bayesian avg, >= 3 ratings)", "bayes")


def run_cold(rec, goal, k=5):
    print(f"\n  New user goal : \"{goal}\"")
    print(f"  NLP tokens    : {preprocess(goal)}")
    c = extract_constraints(goal)
    print(f"  Constraints   : duration = {c['duration']} min, difficulty = {c['difficulty']}")
    print("  Score         : 0.85 * query match + 0.15 * popularity prior (no rating history needed)")
    show(rec.cold_start(goal, k), "Cold-start Top-5")


def run_item(rec, item_id, k=5):
    if item_id not in rec.idx:
        print(f"  Unknown item '{item_id}'.")
        return
    print(f"\n  Because you liked {item_id} {rec.names[item_id]}:")
    show(rec.similar_items(item_id, k), "Most similar workouts (item-to-item TF-IDF cosine)", "cosine")


def run_test_cases(rec, save=True):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("CS467 MP-A16 - Fitness Workout and Training Plan Recommender - required test cases")
        print("Student: Ha Van Quoc Tien (29211355728), CS467A No. 16")
        banner("TEST CASE 1 - Goal query: '20 minute home cardio workout with no equipment'")
        run_query(rec, TEST_QUERY)
        banner(f"TEST CASE 2 - Personal history: user {TEST_USER} rated core and stretching routines highly")
        run_user(rec, TEST_USER)
        banner("TEST CASE 3 - Cold start: brand-new user with only a 1-sentence fitness goal")
        run_cold(rec, TEST_COLD_GOAL)
        print("\n  For comparison, a new user gets only this from the popularity baseline:")
        show(rec.popularity(None), "Popularity baseline (no personalisation)", "bayes")
    out = buf.getvalue()
    print(out)
    if save:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "sample_outputs.txt"), "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[saved to {os.path.join('results', 'sample_outputs.txt')}]")


def interactive(rec):
    print("Interactive mode. Enter a user ID (e.g. U03), an item ID (e.g. W23), 'cold: <goal>', "
          "any free-text request, or 'q' to quit.")
    while True:
        try:
            s = input("\n> ").strip()
        except EOFError:
            break
        if not s or s.lower() in ("q", "quit", "exit"):
            break
        if s.upper() in rec.user_mean.index:
            run_user(rec, s.upper())
        elif s.upper() in rec.idx:
            run_item(rec, s.upper())
        elif s.lower().startswith("cold:"):
            run_cold(rec, s[5:].strip())
        else:
            run_query(rec, s)


def main():
    ap = argparse.ArgumentParser(description="Fitness workout recommender demo (CS467 MP-A16)")
    ap.add_argument("--query", help="free-text workout request")
    ap.add_argument("--user", help="existing user ID, e.g. U03")
    ap.add_argument("--method", default="all", choices=["all", "content", "cf", "hybrid", "popularity"])
    ap.add_argument("--item", help="item ID for item-to-item similarity, e.g. W23")
    ap.add_argument("--cold", help="1-sentence goal of a brand-new user")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--evaluate", action="store_true", help="run offline evaluation and save charts")
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args()

    if a.evaluate:
        from evaluate import make_charts, run_evaluation
        s, ub = run_evaluation()
        make_charts(s, ub)
        return
    rec = load_recommender()
    if a.interactive:
        interactive(rec)
    elif a.query or a.user or a.item or a.cold:
        if a.query:
            run_query(rec, a.query, a.k)
        if a.user:
            run_user(rec, a.user.upper(), a.method, a.k)
        if a.item:
            run_item(rec, a.item.upper(), a.k)
        if a.cold:
            run_cold(rec, a.cold, a.k)
    else:
        run_test_cases(rec)


if __name__ == "__main__":
    main()
