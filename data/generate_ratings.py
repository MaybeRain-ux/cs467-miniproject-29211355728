"""Persona-based generator used to create data/ratings.csv (deterministic seed).

Run: python data/generate_ratings.py   -> rewrites data/ratings.csv identically.
"""
import random
import sys
import pandas as pd

import os
repo = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
items = pd.read_csv(f"{repo}/data/items.csv")
items["tags"] = items["target_muscles_tags"].str.split(";")
rng = random.Random(29211355728)

# persona: (liked tags, disliked tags, n_rated, preferred difficulty set)
personas = {
    "U01": (["cardio", "home", "bodyweight", "low_impact", "fun"], ["barbell", "gym", "pull_up_bar"], 15),
    "U02": (["running", "endurance", "outdoor", "recovery", "legs"], ["barbell", "bench", "machines"], 14),
    "U03": (["core", "abs", "stretching", "flexibility", "pilates", "yoga"], ["hiit", "barbell", "running"], 16),
    "U04": (["barbell", "strength", "gym", "bench", "legs"], ["yoga", "fun", "low_impact", "relaxation"], 15),
    "U05": (["hiit", "cardio", "kettlebell", "full_body", "power"], ["relaxation", "low_impact", "pilates"], 16),
    "U06": (["yoga", "flexibility", "balance", "stretching", "mobility"], ["barbell", "hiit", "machines"], 14),
    "U07": (["quick", "office", "bodyweight", "home", "core"], ["endurance", "barbell", "gym"], 15),
    "U08": (["dumbbells", "arms", "chest", "shoulders", "biceps", "triceps"], ["running", "yoga", "relaxation"], 16),
    "U09": (["low_impact", "balance", "recovery", "relaxation", "stretching"], ["hiit", "advanced", "barbell", "power"], 14),
    "U10": (["machines", "gym", "treadmill", "strength", "full_body"], ["outdoor", "pull_up_bar", "hiit"], 15),
    "U11": (["pull_up_bar", "bodyweight", "strength", "abs", "core"], ["machines", "treadmill", "relaxation"], 15),
    "U12": (["lower_back", "rehab", "mobility", "core", "posture"], ["hiit", "barbell", "running", "power"], 14),
    "U13": (["endurance", "stationary_bike", "rowing_machine", "cardio", "quads"], ["yoga", "pilates", "bench"], 15),
    "U14": (["core", "mobility", "cardio", "stretching", "home"], ["barbell", "bench", "power"], 16),
}

aspect = {
    "core": "the core work", "abs": "the ab work", "stretching": "the stretching", "flexibility": "the flexibility work",
    "cardio": "the cardio", "hiit": "the intervals", "strength": "the lifting", "yoga": "the yoga flow",
    "running": "the run", "mobility": "the mobility drills", "recovery": "the recovery feel",
    "legs": "my legs", "glutes": "my glutes", "chest": "the chest pump", "arms": "the arm pump",
    "back": "my back", "shoulders": "my shoulders", "balance": "the balance drills", "endurance": "the endurance block",
    "full_body": "the full-body burn", "relaxation": "the relaxing pace", "pilates": "the pilates moves",
}

pos5 = ["Loved it, {a} felt amazing.", "Perfect for my goals, {a} was spot on.", "Best session this week, will repeat!",
        "Great flow and {a} was exactly what I needed.", "Awesome workout, felt energized after."]
pos4 = ["Solid session, {a} was challenging but good.", "Really enjoyed {a}, slightly long though.",
        "Good workout, I felt it in {a2} the next day.", "Nice routine, would do again.", "Effective and fun, {a} was great."]
neu3 = ["Okay workout, {a} felt a bit average.", "Decent but nothing special.", "Fine routine, a little repetitive.",
        "It was alright, pacing felt uneven.", "Average, {a} could be better structured."]
neg2 = ["Not really for me, too {adj}.", "Struggled with {a}, felt {adj}.", "Didn't enjoy it, quite {adj}.",
        "Too {adj} for my level.", "Meh, {a} did not suit my goals."]
neg1 = ["Hated it, way too {adj}.", "Boring and {adj}, skipped half.", "Terrible fit for me, too {adj}.",
        "Not my thing at all, felt {adj}.", "Painful and {adj}, won't repeat."]

rows = []
for uid, (likes, dislikes, n) in personas.items():
    scores = []
    for _, it in items.iterrows():
        tags = set(it.tags) | {it.difficulty.lower()}
        aff = len(tags & set(likes)) - 1.3 * len(tags & set(dislikes))
        scores.append((aff, it))
    scores.sort(key=lambda x: -x[0] + rng.random() * 0.8)
    top = scores[: n - 6]
    rest = scores[n - 6:]
    chosen = top + rng.sample(rest, 6)
    for aff, it in chosen:
        r = 2.5 + 0.8 * aff + rng.gauss(0, 0.7)
        r = int(min(5, max(1, round(r))))
        tags = it.tags
        key = next((t for t in tags if t in likes and t in aspect), next((t for t in tags if t in aspect), "full_body"))
        key2 = next((t for t in tags if t in ("legs", "glutes", "core", "back", "shoulders", "arms", "chest")), "my muscles")
        if it.difficulty == "Advanced":
            adj = rng.choice(["intense", "hard", "advanced"])
        elif it.duration_min >= 40:
            adj = rng.choice(["long", "tiring", "slow"])
        else:
            adj = rng.choice(["boring", "easy", "repetitive"])
        tpl = rng.choice({5: pos5, 4: pos4, 3: neu3, 2: neg2, 1: neg1}[r])
        note = tpl.format(a=aspect.get(key, "it"), a2=aspect.get(key2, key2), adj=adj)
        rows.append((uid, it.item_id, r, note))

df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "workout_note"]).sort_values(["user_id", "item_id"])
df.to_csv(f"{repo}/data/ratings.csv", index=False)
n_u, n_i = df.user_id.nunique(), items.shape[0]
print(len(df), "ratings; sparsity", round(1 - len(df) / (n_u * n_i), 3))
print(df.groupby("user_id").size().describe())
print(df.rating.value_counts().sort_index())
print(df.groupby("item_id").size().describe())
print(df[df.user_id == "U03"].merge(items[["item_id", "workout_name"]]).to_string())
