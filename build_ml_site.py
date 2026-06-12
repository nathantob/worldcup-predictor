"""
Builds the ML-powered website data.

Trains the logistic-regression combiner on all data (2006+), then saves:
  - each current team's CURRENT stats snapshot (strength, FIFA, value, form)
  - the trained model's weights (scaler + logistic-regression coefficients)
into mlmodel.js, which the web page uses to run the full combined model.
"""

import csv, math, json
from bisect import bisect_right
from collections import defaultdict, deque
from datetime import date as Date
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from build_squad_values import squad_value

START = "2006-01-01"
TODAY = Date.today()   # always use the current date (for the daily auto-update)
MAX_GOALS = 8
C, HOME_ADV, LR = 0.30, 0.40, 0.05

FACT = [math.factorial(k) for k in range(MAX_GOALS + 1)]
def pmf(lam):
    e = math.exp(-lam)
    return [e * lam ** k / FACT[k] for k in range(MAX_GOALS + 1)]
def probs_from(lamH, lamA):
    hp, ap = pmf(lamH), pmf(lamA)
    ph = pd = pa = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = hp[i] * ap[j]
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
    return ph, pd, pa
def to_date(s):
    y, m, d = s.split("-"); return Date(int(y), int(m), int(d))

# --- FIFA points ---
ALIAS = {"United States": ["USA"], "South Korea": ["Korea Republic"],
    "North Korea": ["Korea DPR"], "Ivory Coast": ["Côte d'Ivoire"],
    "Iran": ["IR Iran"], "China": ["China PR"], "DR Congo": ["Congo DR"],
    "Cape Verde": ["Cabo Verde", "Cape Verde Islands"],
    "Kyrgyzstan": ["Kyrgyz Republic"], "Taiwan": ["Chinese Taipei"]}
raw = defaultdict(list)
with open("fifa_ranking.csv", newline="") as f:
    for r in csv.DictReader(f):
        try: raw[r["team"]].append((r["date"], float(r["total_points"])))
        except ValueError: continue
fifa_cache = {}
def fifa_series(team):
    if team in fifa_cache: return fifa_cache[team]
    merged = []
    for n in ALIAS.get(team, [team]): merged.extend(raw.get(n, []))
    merged.sort()
    fifa_cache[team] = ([d for d, _ in merged], [p for _, p in merged])
    return fifa_cache[team]
def fifa_at(team, date):
    dates, pts = fifa_series(team)
    i = bisect_right(dates, date)
    return pts[i - 1] if i > 0 else None

# --- Walk all games: train goals model, collect ML features ---
with open("results.csv", newline="") as f:
    matches = sorted(csv.DictReader(f), key=lambda row: row["date"])

atk, dfn = defaultdict(float), defaultdict(float)
form = defaultdict(lambda: deque(maxlen=5))
games = defaultdict(int)
last_game = {}
X, y = [], []

for m in matches:
    home, away = m["home_team"], m["away_team"]
    try:
        hg = int(m["home_score"]); ag = int(m["away_score"])
    except ValueError:
        continue
    neutral = m["neutral"].strip().upper() == "TRUE"
    boost = 0.0 if neutral else HOME_ADV
    ah, aa, dh, da = atk[home], atk[away], dfn[home], dfn[away]
    lamH = min(max(math.exp(C + boost + ah - da), 0.05), 8.0)
    lamA = min(max(math.exp(C + aa - dh), 0.05), 8.0)
    label = 0 if hg > ag else (1 if hg == ag else 2)

    if m["date"] >= START:
        ph, pd, pa = probs_from(lamH, lamA)
        fh, fa = fifa_at(home, m["date"]), fifa_at(away, m["date"])
        fifa_diff = (fh - fa) if (fh is not None and fa is not None) else 0.0
        fifa_missing = 0 if (fh is not None and fa is not None) else 1
        d = to_date(m["date"])
        vh, va = squad_value(home, d), squad_value(away, d)
        if vh > 0 and va > 0:
            val_diff = math.log10(vh) - math.log10(va); val_missing = 0
        else:
            val_diff = 0.0; val_missing = 1
        hform = sum(form[home]) / len(form[home]) if form[home] else 0.0
        aform = sum(form[away]) / len(form[away]) if form[away] else 0.0
        X.append([lamH, lamA, lamH - lamA, ph, pd, pa, (ah + dh) - (aa + da),
                  0 if neutral else 1, fifa_diff, fifa_missing,
                  val_diff, val_missing, hform, aform])
        y.append(label)

    atk[home] = ah + LR * (hg - lamH); dfn[away] = da - LR * (hg - lamH)
    atk[away] = aa + LR * (ag - lamA); dfn[home] = dh - LR * (ag - lamA)
    form[home].append(hg - ag); form[away].append(ag - hg)
    games[home] += 1; games[away] += 1
    last_game[home] = m["date"]; last_game[away] = m["date"]

X = np.array(X); y = np.array(y)
scaler = StandardScaler().fit(X)
clf = LogisticRegression(C=0.5, max_iter=5000).fit(scaler.transform(X), y)

# Sanity: our manual softmax must match sklearn's predict_proba.
Z = (X[:3] - scaler.mean_) / scaler.scale_
scores = Z @ clf.coef_.T + clf.intercept_
man = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
assert np.allclose(man, clf.predict_proba(scaler.transform(X[:3])), atol=1e-9)
print("Softmax replication check passed.")

# --- Build current snapshot per CURRENT team ---
teams = {}
for t in games:
    if games[t] >= 20 and last_game.get(t, "") >= "2023-01-01":
        f = fifa_at(t, TODAY.isoformat())
        v = squad_value(t, TODAY)
        teams[t] = {
            "atk": round(atk[t], 4), "def": round(dfn[t], 4),
            "fifa": round(f, 1) if f is not None else None,
            "vlog": round(math.log10(v), 4) if v > 0 else None,
            "form": round(sum(form[t]) / len(form[t]), 3) if form[t] else 0.0,
        }

model = {
    "C": C, "HOME_ADV": HOME_ADV, "MAX_GOALS": MAX_GOALS,
    "mean": [round(x, 6) for x in scaler.mean_],
    "scale": [round(x, 6) for x in scaler.scale_],
    "coef": [[round(x, 6) for x in row] for row in clf.coef_],
    "intercept": [round(x, 6) for x in clf.intercept_],
    "classes": clf.classes_.tolist(),   # [0,1,2] = home, draw, away
    "teams": teams,
}
with open("mlmodel.js", "w") as f:
    f.write("const ML = ")
    json.dump(model, f, indent=1)
    f.write(";\n")

print(f"Saved {len(teams)} current teams + model weights to mlmodel.js")
print("Class order:", clf.classes_.tolist(), "(0=home win, 1=draw, 2=away win)")
