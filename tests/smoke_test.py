"""Offline smoke test: exercises the full pipeline on synthetic data, bypassing the
Kaggle download. Validates feature_engineering -> model -> simulator -> report
without needing credentials.  Run from the project root:  python tests/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from feature_engineering import build_rank_map, fit_full_history, make_training_set
from model import train_model
from report import build_html_report
from simulator import (build_tournament, group_stage_predictions,
                       representative_bracket, run_simulations)
import main as m

rng = np.random.default_rng(0)

# 64 synthetic national teams with a latent strength.
teams = [f"Nation{i:02d}" for i in range(64)]
strength = {t: rng.normal(0, 1) for t in teams}

def play(home, away, year, tournament):
    lam_h = max(0.2, 1.3 + 0.6 * (strength[home] - strength[away]) + 0.3)
    lam_a = max(0.2, 1.3 + 0.6 * (strength[away] - strength[home]))
    return {"date": pd.Timestamp(f"{year}-06-15"), "home_team": home, "away_team": away,
            "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
            "tournament": tournament, "neutral": True}

rows = []
# Friendlies/qualifiers across the era (build up team history).
for year in range(1990, 2023):
    for _ in range(120):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append(play(h, a, year, "Friendly"))
# World Cups 1994..2022: 32 teams play a round-robin-ish slate, tagged WC.
for year in range(1994, 2023, 4):
    wc_teams = list(rng.choice(teams, 32, replace=False))
    for _ in range(64):
        h, a = rng.choice(wc_teams, 2, replace=False)
        rows.append(play(h, a, year, "FIFA World Cup"))

history = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# Team dataset (rank correlates with latent strength).
ranked = sorted(teams, key=lambda t: -strength[t])
team_df = pd.DataFrame({"team": ranked, "rank": range(1, len(ranked) + 1)})

# Fixtures: no usable group column -> forces the fallback group builder.
fixtures = pd.DataFrame(columns=["home_team", "away_team"])

print("history:", history.shape, "| WC matches:",
      (history["tournament"] == "FIFA World Cup").sum())

rank_map = build_rank_map(team_df)
X, y, meta = make_training_set(history, rank_map)
print("training X:", X.shape, "| classes:", np.bincount(y),
      "| years:", meta["year"].min(), "-", meta["year"].max())

predictor, ll = train_model(X, y, meta)
print("walk-forward log-loss:", round(ll, 4))

groups, source = m.build_groups(fixtures, team_df)
assert m._valid_groups(groups), groups
print("groups source:", source, "| sizes:", {g: len(v) for g, v in groups.items()})

fb = fit_full_history(history, rank_map)
t = build_tournament(groups, predictor, fb)

summary, finish_counts = run_simulations(t, n=100)
print("summary shape:", summary.shape)
print(summary.head(5).to_string())
assert abs(summary["P(Win)"].sum() - 1.0) < 1e-6, summary["P(Win)"].sum()

gp = group_stage_predictions(t)
br = representative_bracket(t, finish_counts, 100)
print("group preds:", gp.shape, "| bracket rounds:", br["round"].unique().tolist())
assert len(br) == 31, len(br)  # 16+8+4+2+1

build_html_report(summary, gp, br, t, finish_counts, 100, ll, source,
                  out_path=__import__("pathlib").Path("results/_smoke_report.html"))
print("\nSMOKE TEST PASSED")
