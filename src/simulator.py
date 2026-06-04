"""simulator.py — Monte Carlo simulation of the full WC2026 tournament.

Official WC2026 format: 12 groups of 4 (A-L) -> 72 group matches -> top 2 of each
group + 8 best third-placed teams qualify for a 32-team knockout
(R32 -> R16 -> QF -> SF -> Final).

Design notes
------------
* Match outcome probabilities come from the calibrated XGBoost model. To remove
  spurious home/away orientation bias at neutral World Cup venues we *symmetrize*:
  predict both orderings and average the three classes.
* The model predicts only W/D/L. Goals (needed for GD/GF tiebreakers and for the
  "most frequent score" in the report) are drawn from a lightweight Poisson model
  built on each team's attack/defense strengths, *conditioned* on the sampled
  outcome so the scoreline is always consistent with it.
* Knockout bracket: qualifiers are seeded (group finish, then Pts/GD/GF) and placed
  into a standard single-elimination bracket. This is a transparent approximation of
  FIFA's official slot table (which depends on a 495-row third-place combination
  matrix); it preserves realistic knockout dynamics. The bracket is fixed once formed
  (no reseeding between rounds).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from feature_engineering import FEATURE_COLS, build_fixture_features

ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Winner"]
_MAX_GOALS = 8  # truncation for the Poisson scoreline grid


# --------------------------------------------------------------------------- #
# Tournament container
# --------------------------------------------------------------------------- #
@dataclass
class Tournament:
    groups: dict[str, list[str]]                 # {"A": [t1, t2, t3, t4], ...}
    pred: dict[tuple, np.ndarray]                # (home, away) -> [p_away, p_draw, p_home]
    attack: dict[str, float]                     # team -> avg goals scored
    defense: dict[str, float]                    # team -> avg goals conceded
    teams: list[str] = field(default_factory=list)

    # ----- probability helpers -------------------------------------------- #
    def match_probs(self, a: str, b: str) -> np.ndarray:
        """Symmetric [P(a win), P(draw), P(b win)] averaged over both orderings."""
        p_ab = self.pred[(a, b)]          # [away=b, draw, home=a]
        p_ba = self.pred[(b, a)]          # [away=a, draw, home=b]
        p_a = 0.5 * (p_ab[2] + p_ba[0])
        p_d = 0.5 * (p_ab[1] + p_ba[1])
        p_b = 0.5 * (p_ab[0] + p_ba[2])
        s = p_a + p_d + p_b
        return np.array([p_a, p_d, p_b]) / s

    def ko_win_prob(self, a: str, b: str) -> float:
        """P(a beats b) in a knockout (draw mass redistributed proportionally)."""
        p_a, p_d, p_b = self.match_probs(a, b)
        denom = p_a + p_b
        if denom == 0:
            return 0.5
        return p_a + p_d * (p_a / denom)

    # ----- scoreline (Poisson, conditioned on outcome) -------------------- #
    def _rates(self, home: str, away: str) -> tuple[float, float]:
        lam_h = np.clip((self.attack.get(home, 1.0) + self.defense.get(away, 1.0)) / 2, 0.2, 4.5)
        lam_a = np.clip((self.attack.get(away, 1.0) + self.defense.get(home, 1.0)) / 2, 0.2, 4.5)
        return float(lam_h), float(lam_a)

    def sample_scoreline(self, home: str, away: str, outcome: int, rng: np.random.Generator):
        """Draw (home_goals, away_goals) consistent with outcome (2=home,1=draw,0=away)."""
        lam_h, lam_a = self._rates(home, away)
        for _ in range(20):
            hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            o = 2 if hg > ag else (1 if hg == ag else 0)
            if o == outcome:
                return hg, ag
        # Deterministic fallback if rejection sampling fails.
        if outcome == 1:
            g = int(round((lam_h + lam_a) / 2))
            return g, g
        if outcome == 2:
            return max(1, int(round(lam_h))), max(0, int(round(lam_h)) - 1)
        return max(0, int(round(lam_a)) - 1), max(1, int(round(lam_a)))

    def modal_scoreline(self, home: str, away: str, ko: bool = False) -> tuple[int, int]:
        """Most probable scoreline from the Poisson grid.

        When ko=True the result is the most probable scoreline in which `home` WINS
        (home_goals > away_goals), so it never contradicts a predicted winner.
        """
        lam_h, lam_a = self._rates(home, away)
        gh = np.arange(_MAX_GOALS + 1)
        fact = np.array([math.factorial(int(i)) for i in gh], dtype=float)
        ph = np.exp(-lam_h) * lam_h**gh / fact
        pa = np.exp(-lam_a) * lam_a**gh / fact
        grid = np.outer(ph, pa)
        if ko:
            grid = np.tril(grid, k=-1)  # keep only home_goals > away_goals
            if grid.sum() == 0:
                return 1, 0
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        return int(i), int(j)


# --------------------------------------------------------------------------- #
# Build a Tournament from a model + feature builder
# --------------------------------------------------------------------------- #
def build_tournament(groups: dict[str, list[str]], predictor, fb) -> Tournament:
    """Batch-predict every ordered pairing among the 48 teams once, up front."""
    teams = [t for g in groups.values() for t in g]
    rows, keys = [], []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            rows.append(build_fixture_features(h, a, fb).iloc[0])
            keys.append((h, a))
    X = pd.DataFrame(rows, columns=FEATURE_COLS).reset_index(drop=True)
    proba = predictor.predict_proba(X)  # columns [Away, Draw, Home]
    pred = {k: proba[i] for i, k in enumerate(keys)}

    attack = {t: fb._attack(t) for t in teams}
    defense = {t: fb._defense(t) for t in teams}
    return Tournament(groups=groups, pred=pred, attack=attack, defense=defense, teams=teams)


# --------------------------------------------------------------------------- #
# Group stage
# --------------------------------------------------------------------------- #
def _round_robin(team_list: list[str]) -> list[tuple[str, str]]:
    return [(team_list[i], team_list[j])
            for i in range(len(team_list)) for j in range(i + 1, len(team_list))]


def simulate_group(t: Tournament, group_teams: list[str], rng: np.random.Generator) -> list[dict]:
    """Simulate one group; return standings (sorted best->worst) with stats."""
    st = {tm: {"team": tm, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
          for tm in group_teams}
    h2h_pts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for home, away in _round_robin(group_teams):
        outcome = rng.choice(3, p=t.match_probs(home, away))  # 0 away,1 draw,2 home
        hg, ag = t.sample_scoreline(home, away, int(outcome), rng)
        st[home]["GF"] += hg; st[home]["GA"] += ag
        st[away]["GF"] += ag; st[away]["GA"] += hg
        if outcome == 2:
            st[home]["W"] += 1; st[away]["L"] += 1
            st[home]["Pts"] += 3; h2h_pts[home][away] += 3
        elif outcome == 0:
            st[away]["W"] += 1; st[home]["L"] += 1
            st[away]["Pts"] += 3; h2h_pts[away][home] += 3
        else:
            st[home]["D"] += 1; st[away]["D"] += 1
            st[home]["Pts"] += 1; st[away]["Pts"] += 1
            h2h_pts[home][away] += 1; h2h_pts[away][home] += 1

    for s in st.values():
        s["GD"] = s["GF"] - s["GA"]

    # FIFA tiebreakers: Pts, GD, GF, then head-to-head pts, then random.
    def sort_key(s):
        return (s["Pts"], s["GD"], s["GF"],
                sum(h2h_pts[s["team"]].values()), rng.random())

    return sorted(st.values(), key=sort_key, reverse=True)


def simulate_group_stage(t: Tournament, rng: np.random.Generator):
    """Return (standings_by_group, qualifiers list of dicts with seed info)."""
    standings = {}
    winners, runners, thirds = [], [], []
    for g, teams in t.groups.items():
        table = simulate_group(t, teams, rng)
        for pos, s in enumerate(table):
            s["group"] = g
            s["finish"] = pos + 1
        standings[g] = table
        winners.append(table[0])
        runners.append(table[1])
        thirds.append(table[2])

    # 8 best third-placed teams (Pts, GD, GF).
    thirds_sorted = sorted(
        thirds, key=lambda s: (s["Pts"], s["GD"], s["GF"], rng.random()), reverse=True
    )
    best_thirds = thirds_sorted[:8]

    qualifiers = winners + runners + best_thirds
    return standings, qualifiers


# --------------------------------------------------------------------------- #
# Knockout bracket
# --------------------------------------------------------------------------- #
def _seed_order(n: int) -> list[int]:
    """Standard single-elimination bracket seeding order (1-indexed)."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        order = [x for o in order for x in (o, m + 1 - o)]
    return order


def _seed_qualifiers(qualifiers: list[dict]) -> list[dict]:
    """Rank the 32 qualifiers: group winners, then runners-up, then thirds."""
    return sorted(
        qualifiers,
        key=lambda s: (s["finish"], -s["Pts"], -s["GD"], -s["GF"]),
    )


def build_bracket(qualifiers: list[dict]) -> list[tuple[dict, dict]]:
    """Form the 16 R32 pairings from seeded qualifiers (1 vs 32, ...)."""
    seeded = _seed_qualifiers(qualifiers)
    order = _seed_order(32)
    arranged = [seeded[i - 1] for i in order]
    return [(arranged[i], arranged[i + 1]) for i in range(0, 32, 2)]


def simulate_knockout(t: Tournament, qualifiers: list[dict], rng: np.random.Generator):
    """Run R32->Final. Return (reached dict team->round, champion)."""
    reached = {s["team"]: "R32" for s in qualifiers}
    pairs = build_bracket(qualifiers)
    round_idx = 0
    teams_in_round = pairs
    while True:
        round_name = ROUNDS[round_idx + 1]  # winners advance TO this round
        winners = []
        for a, b in teams_in_round:
            ta, tb = a["team"], b["team"]
            p_a = t.ko_win_prob(ta, tb)
            winner = a if rng.random() < p_a else b
            reached[winner["team"]] = round_name
            winners.append(winner)
        if len(winners) == 1:
            return reached, winners[0]["team"]
        teams_in_round = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        round_idx += 1


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
_RANK = {r: i for i, r in enumerate(["GROUP"] + ROUNDS)}


def run_simulations(t: Tournament, n: int = 10000, seed: int = 42):
    """Run n full tournaments. Return (summary_df, finish_counts)."""
    rng = np.random.default_rng(seed)
    reach_counts = {tm: {r: 0 for r in ROUNDS} for tm in t.teams}
    finish_counts = {tm: {1: 0, 2: 0, 3: 0, 4: 0} for tm in t.teams}
    qualify_counts = {tm: 0 for tm in t.teams}

    for _ in range(n):
        standings, qualifiers = simulate_group_stage(t, rng)
        for table in standings.values():
            for s in table:
                finish_counts[s["team"]][s["finish"]] += 1
        for s in qualifiers:
            qualify_counts[s["team"]] += 1
        reached, _champ = simulate_knockout(t, qualifiers, rng)
        for tm, r in reached.items():
            # credit every round up to and including the furthest reached.
            # _RANK[r] is offset by the leading "GROUP" entry, so ROUNDS[:_RANK[r]]
            # already spans R32..r inclusive.
            for rr in ROUNDS[: _RANK[r]]:
                reach_counts[tm][rr] += 1

    rows = []
    for tm in t.teams:
        group = next(g for g, members in t.groups.items() if tm in members)
        rows.append({
            "team": tm,
            "group": group,
            "P(Qualify)": qualify_counts[tm] / n,
            "P(R32)": reach_counts[tm]["R32"] / n,
            "P(R16)": reach_counts[tm]["R16"] / n,
            "P(QF)": reach_counts[tm]["QF"] / n,
            "P(SF)": reach_counts[tm]["SF"] / n,
            "P(Final)": reach_counts[tm]["Final"] / n,
            "P(Win)": reach_counts[tm]["Winner"] / n,
        })
    summary = pd.DataFrame(rows).sort_values("P(Win)", ascending=False).reset_index(drop=True)
    return summary, finish_counts


# --------------------------------------------------------------------------- #
# Representative (modal) bracket + group predictions for the report
# --------------------------------------------------------------------------- #
def expected_group_table(t: Tournament, group_teams: list[str]) -> list[dict]:
    """Deterministic expected standings using expected points (no sampling)."""
    st = {tm: {"team": tm, "W": 0.0, "D": 0.0, "L": 0.0, "GF": 0.0, "GA": 0.0, "Pts": 0.0}
          for tm in group_teams}
    for home, away in _round_robin(group_teams):
        p_h, p_d, p_a = t.match_probs(home, away)
        lam_h, lam_a = t._rates(home, away)
        st[home]["Pts"] += 3 * p_h + p_d; st[away]["Pts"] += 3 * p_a + p_d
        st[home]["W"] += p_h; st[home]["D"] += p_d; st[home]["L"] += p_a
        st[away]["W"] += p_a; st[away]["D"] += p_d; st[away]["L"] += p_h
        st[home]["GF"] += lam_h; st[home]["GA"] += lam_a
        st[away]["GF"] += lam_a; st[away]["GA"] += lam_h
    for s in st.values():
        s["GD"] = s["GF"] - s["GA"]
    table = sorted(st.values(), key=lambda s: (s["Pts"], s["GD"], s["GF"]), reverse=True)
    for pos, s in enumerate(table):
        s["finish"] = pos + 1
    return table


def group_stage_predictions(t: Tournament) -> pd.DataFrame:
    """Per-match group-stage probabilities for the report/CSV."""
    rows = []
    for g, teams in t.groups.items():
        for home, away in _round_robin(teams):
            p_h, p_d, p_a = t.match_probs(home, away)
            outcomes = {"Home Win": p_h, "Draw": p_d, "Away Win": p_a}
            rows.append({
                "group": g, "home": home, "away": away,
                "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
                "most_likely": max(outcomes, key=outcomes.get),
            })
    return pd.DataFrame(rows)


def representative_bracket(t: Tournament, finish_counts: dict, n: int) -> pd.DataFrame:
    """A single displayable bracket built from each group's modal finishers.

    Qualifiers = the two most-likely-to-finish-1st/2nd per group + 8 teams with the
    highest qualify-ish rate among modal thirds. Then deterministically advance the
    favorite in each KO match, recording win prob and modal score.
    """
    # Expected standings drive the seeding for a stable, readable bracket.
    qualifiers = []
    thirds = []
    for g, teams in t.groups.items():
        table = expected_group_table(t, teams)
        qualifiers.extend(table[:2])
        thirds.append(table[2])
    thirds_sorted = sorted(thirds, key=lambda s: (s["Pts"], s["GD"], s["GF"]), reverse=True)
    qualifiers.extend(thirds_sorted[:8])

    pairs = build_bracket(qualifiers)
    rows = []
    round_idx = 0
    current = pairs
    while True:
        round_name = ROUNDS[round_idx]  # this round's matches
        winners = []
        for a, b in current:
            ta, tb = a["team"], b["team"]
            p_a = t.ko_win_prob(ta, tb)
            winner, loser = (a, b) if p_a >= 0.5 else (b, a)
            wg, lg = t.modal_scoreline(winner["team"], loser["team"], ko=True)
            # Orient the score to the displayed home-away order.
            home_g, away_g = (wg, lg) if winner is a else (lg, wg)
            rows.append({
                "round": round_name, "home": ta, "away": tb,
                "p_home_win": round(p_a, 4), "p_away_win": round(1 - p_a, 4),
                "predicted_winner": winner["team"],
                "score": f"{home_g}-{away_g}",
            })
            winners.append(winner)
        if len(winners) == 1:
            break
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        round_idx += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("simulator.py is a library module; run via main.py.")
