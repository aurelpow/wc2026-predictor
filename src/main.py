"""main.py — end-to-end WC2026 prediction pipeline.

Usage:
  python main.py              # full run, 10,000 Monte Carlo simulations
  python main.py --debug      # fast run, 100 simulations
  python main.py --seed 7     # different random seed (varies bracket/outcomes)

Steps: download data -> clean -> features -> train calibrated model ->
build 12 groups -> Monte Carlo tournament -> write results/ + HTML report ->
print the top-10 contenders by P(Win).
"""
from __future__ import annotations

import argparse
import shutil
import string
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
from feature_engineering import (build_rank_map, build_value_map,
                                 fit_full_history, make_training_set)
from model import WCPredictor, train_model
from report import build_html_report
from simulator import (
    build_tournament,
    group_stage_predictions,
    representative_bracket,
    run_simulations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"  # GitHub Pages serves index.html from here
GROUP_LETTERS = list(string.ascii_uppercase[:12])  # A..L

# Actual winners of the March 2026 play-offs, which the fixtures dataset still lists
# as TBD placeholders. Maps the placeholder label -> the real (canonical) team.
# UEFA: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(UEFA)
# Inter-confederation: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(inter-confederation_play-offs)
PLAYOFF_RESULTS = {
    "Winner UEFA Playoff A": "Bosnia and Herzegovina",  # -> Group B
    "Winner UEFA Playoff B": "Sweden",                   # -> Group F
    "Winner UEFA Playoff C": "Turkey",                   # -> Group D
    "Winner UEFA Playoff D": "Czech Republic",           # -> Group A
    "Winner FIFA Playoff 1": "DR Congo",                 # -> Group K
    "Winner FIFA Playoff 2": "Iraq",                     # -> Group I
}


# --------------------------------------------------------------------------- #
# Group construction
# --------------------------------------------------------------------------- #
def build_groups(fixtures: pd.DataFrame, team_df: pd.DataFrame) -> tuple[dict, str]:
    """Derive {group_letter: [4 teams]} from the fixtures draw, with fallback."""
    groups = _groups_from_draw(fixtures, team_df)
    if _valid_groups(groups):
        return groups, "fixtures draw + actual March-2026 play-off winners"

    groups = _fallback_groups(fixtures, team_df)
    return groups, "fallback (top-48 by FIFA rank, seeded draw)"


def _candidate_entrants(team_df: pd.DataFrame) -> list[str]:
    """2026 entrants by rank (used to fill TBD playoff slots)."""
    if team_df is None or "team" not in team_df.columns:
        return []
    df = team_df
    if "version" in df.columns and (df["version"] == 2026).any():
        df = df[df["version"] == 2026]
    rank_map = build_rank_map(team_df)
    return sorted(df["team"].dropna().unique(), key=lambda t: rank_map.get(t, 999))


def _groups_from_draw(fixtures: pd.DataFrame, team_df: pd.DataFrame) -> dict:
    """Use the explicit group/placeholder draw; fill TBD slots with best entrants."""
    if fixtures is None or not {"team", "group", "is_placeholder"}.issubset(fixtures.columns):
        return {}

    groups: dict[str, list[str]] = {g: [] for g in GROUP_LETTERS}
    unresolved: list[str] = []  # group letters whose play-off winner is unknown
    for _, row in fixtures.iterrows():
        g = str(row["group"]).strip().upper()
        if g not in groups:
            continue
        if row["is_placeholder"]:
            real = PLAYOFF_RESULTS.get(row["team"])
            if real:
                groups[g].append(dl.normalize_team(real))
            else:
                unresolved.append(g)  # fall back to a rank-based guess below
        else:
            groups[g].append(row["team"])

    # Any play-off slot we don't have a confirmed result for: fill by FIFA rank.
    if unresolved:
        placed = {t for ts in groups.values() for t in ts}
        pool = [t for t in _candidate_entrants(team_df) if t not in placed]
        for g in unresolved:
            if pool:
                groups[g].append(pool.pop(0))
    return {g: v[:4] for g, v in groups.items()}


def _valid_groups(groups: dict) -> bool:
    return (
        len(groups) == 12
        and all(g in groups for g in GROUP_LETTERS)
        and all(len(v) == 4 for v in groups.values())
    )


def _fallback_groups(fixtures: pd.DataFrame, team_df: pd.DataFrame) -> dict:
    """Build a plausible 48-team draw, seeded by FIFA rank into 12 groups."""
    teams: list[str] = []
    if fixtures is not None and "team" in fixtures.columns:
        teams = list(fixtures.loc[~fixtures.get("is_placeholder", False), "team"].dropna())
    pool = _candidate_entrants(team_df)
    for t in pool:  # top up to 48 with ranked entrants
        if t not in teams:
            teams.append(t)
    teams = teams[:48]
    if len(teams) < 48:
        teams += [f"Team{i}" for i in range(len(teams), 48)]  # last-ditch padding

    # Snake draw across 12 groups by rank pot to spread strength.
    groups = {g: [] for g in GROUP_LETTERS}
    for pot in range(4):
        chunk = teams[pot * 12:(pot + 1) * 12]
        order = GROUP_LETTERS if pot % 2 == 0 else GROUP_LETTERS[::-1]
        for letter, tm in zip(order, chunk):
            groups[letter].append(tm)
    return groups


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(n_sims: int, seed: int = 42) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Downloading datasets (Kaggle)...")
    dl.download_datasets()

    print("[2/6] Loading & cleaning data...")
    data = dl.clean()
    history, team_df, fixtures = data["history"], data["team"], data["fixtures"]
    print(f"      historical matches (post-1990): {len(history):,}")

    print("[3/6] Engineering features & training calibrated model...")
    rank_map = build_rank_map(team_df)
    value_map = build_value_map(team_df)
    X, y, meta = make_training_set(history, rank_map, value_map=value_map)
    predictor, logloss = train_model(X, y, meta, seed=seed)
    print(f"      trained on {len(X):,} matches ({meta['year'].min()}-{meta['year'].max()})"
          f" | walk-forward log-loss: {logloss:.4f}")

    print("[4/6] Building tournament structure...")
    groups, source = build_groups(fixtures, team_df)
    print(f"      groups from: {source}")
    fb = fit_full_history(history, rank_map, value_map=value_map)
    tournament = build_tournament(groups, predictor, fb)

    print(f"[5/6] Running {n_sims:,} Monte Carlo simulations (seed={seed})...")
    summary, finish_counts = run_simulations(tournament, n=n_sims, seed=seed)
    group_preds = group_stage_predictions(tournament)
    bracket = representative_bracket(tournament, finish_counts, n_sims)

    print("[6/6] Writing results...")
    summary.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)
    group_preds.to_csv(RESULTS_DIR / "group_stage_predictions.csv", index=False)
    bracket.to_csv(RESULTS_DIR / "knockout_bracket.csv", index=False)
    report_path = RESULTS_DIR / "full_tournament_report.html"
    build_html_report(
        summary, group_preds, bracket, tournament, finish_counts, n_sims,
        logloss, source, out_path=report_path,
    )
    # Publish a copy for GitHub Pages (docs/index.html).
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, DOCS_DIR / "index.html")

    _print_top10(summary)


def _print_top10(summary: pd.DataFrame) -> None:
    top = summary.head(10)
    print("\n" + "=" * 52)
    print("  TOP 10 WC2026 CONTENDERS  (by P(Win Tournament))")
    print("=" * 52)
    print(f"  {'#':>2}  {'Team':<22}{'P(Win)':>8}{'P(Final)':>10}")
    print("-" * 52)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i:>2}  {r['team']:<22}{r['P(Win)']*100:>7.1f}%{r['P(Final)']*100:>9.1f}%")
    print("=" * 52)
    print(f"\n🏆 Most likely winner: {top.iloc[0]['team']} ({top.iloc[0]['P(Win)']*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="WC2026 prediction pipeline")
    parser.add_argument("--debug", action="store_true",
                        help="run 100 simulations instead of 10,000")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for model + Monte Carlo (default 42)")
    args = parser.parse_args()
    run(n_sims=100 if args.debug else 10000, seed=args.seed)


if __name__ == "__main__":
    main()
