"""feature_engineering.py — build per-fixture features from historical matches.

Features are computed *as of* each match date (only past matches are used) so the
training set has no look-ahead leakage. The same builder, after consuming the full
history, produces features for the 2026 fixtures.

Per-fixture feature vector (matches prompt.md):
  home_form, away_form, home_attack, away_attack,
  home_defense, away_defense, h2h_winrate, rank_delta, wc_exp_delta

Run standalone:  python feature_engineering.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from confederations import confed_strength

FEATURE_COLS = [
    "home_form", "away_form",
    "home_attack", "away_attack",
    "home_defense", "away_defense",
    "h2h_winrate", "rank_delta", "wc_exp_delta",
    "value_log_delta",  # squad market-value gap (log) — anchors true squad quality
    "elo_delta",        # opponent-adjusted strength earned against real opposition
    "confed_delta",     # zone-strength gap (Europe/South America toughest)
]

ELO_START = 1500.0
ELO_K = 32.0          # base learning rate for the Elo update

FORM_WINDOW = 10      # last N matches for form
STRENGTH_WINDOW = 20  # last N matches for attack/defense
DEFAULT_RANK = 50.0   # rank assumed for teams absent from the ranking snapshot
WC_NAME = "FIFA World Cup"

TRAIN_MIN_YEAR = 2014          # emit training rows from here (features still warm up from full history)
RECENCY_HALFLIFE_YEARS = 4.0   # a match this old gets half the weight of a fresh one

# Continental finals / major tournaments -> full competitiveness weight.
_MAJOR_TOURNAMENTS = (
    "fifa world cup", "uefa euro", "copa am", "african cup of nations",
    "afc asian cup", "gold cup", "confederations cup", "nations cup",
)


def competitiveness(tournament: str) -> float:
    """Base sample weight by match importance. Official games count far more than
    friendlies — a friendly is worth only 1/5 of a major-tournament match."""
    t = (tournament or "").lower()
    if "friendly" in t:
        return 0.2
    if "qualifi" in t or "nations league" in t:
        return 0.85
    if any(m in t for m in _MAJOR_TOURNAMENTS):
        return 1.0
    return 0.65  # other competitive (regional cups, playoffs, etc.)


def sample_weights(meta: pd.DataFrame, ref_date, halflife: float = RECENCY_HALFLIFE_YEARS) -> np.ndarray:
    """Per-row training weight = recency decay (vs ref_date) x competitiveness."""
    years_ago = (pd.Timestamp(ref_date) - meta["date"]).dt.days / 365.25
    years_ago = years_ago.clip(lower=0.0)
    recency = 0.5 ** (years_ago / halflife)
    comp = meta["tournament"].map(competitiveness)
    return (recency * comp).to_numpy(dtype=float)


def _result_points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


class FeatureBuilder:
    """Accumulates team history match-by-match and emits as-of features."""

    def __init__(self, rank_map: dict | None = None, default_rank: float = DEFAULT_RANK,
                 value_map: dict | None = None):
        self.rank_map = rank_map or {}
        self.default_rank = default_rank
        self.value_map = value_map or {}
        # Default squad value for unknown teams = median of known values (neutral).
        self.default_value = (
            float(np.median(list(self.value_map.values()))) if self.value_map else 1.0
        )
        self.scored: dict[str, list[int]] = defaultdict(list)
        self.conceded: dict[str, list[int]] = defaultdict(list)
        self.points: dict[str, list[int]] = defaultdict(list)   # recency: newest last
        self.h2h: dict[tuple, list[int]] = defaultdict(list)     # winner from key[0] view
        self.wc_years: dict[str, set] = defaultdict(set)
        self.elo: dict[str, float] = defaultdict(lambda: ELO_START)

    # --- feature components ------------------------------------------------- #
    def _form(self, team: str) -> float:
        pts = self.points[team][-FORM_WINDOW:]
        if not pts:
            return 0.5
        # Linearly increasing weights so recent matches dominate.
        w = np.arange(1, len(pts) + 1, dtype=float)
        return float(np.average(pts, weights=w) / 3.0)  # scale to 0..1

    def _attack(self, team: str) -> float:
        gs = self.scored[team][-STRENGTH_WINDOW:]
        return float(np.mean(gs)) if gs else 1.0

    def _defense(self, team: str) -> float:
        gc = self.conceded[team][-STRENGTH_WINDOW:]
        return float(np.mean(gc)) if gc else 1.0

    def _h2h_winrate(self, home: str, away: str) -> float:
        results = self.h2h[(home, away)]
        if not results:
            return 0.5
        return float(np.mean(results))  # fraction of meetings home team won

    def _rank(self, team: str) -> float:
        r = self.rank_map.get(team, np.nan)
        return float(r) if r == r else self.default_rank  # NaN-safe

    def _wc_exp(self, team: str) -> int:
        return len(self.wc_years[team])

    def _log_value(self, team: str) -> float:
        v = self.value_map.get(team, np.nan)
        v = float(v) if v == v and v and v > 0 else self.default_value  # NaN/0-safe
        return float(np.log1p(v))

    # --- public API --------------------------------------------------------- #
    def features_for(self, home: str, away: str) -> dict:
        return {
            "home_form": self._form(home),
            "away_form": self._form(away),
            "home_attack": self._attack(home),
            "away_attack": self._attack(away),
            "home_defense": self._defense(home),
            "away_defense": self._defense(away),
            "h2h_winrate": self._h2h_winrate(home, away),
            # lower rank number = stronger; positive delta => home is stronger
            "rank_delta": self._rank(away) - self._rank(home),
            "wc_exp_delta": self._wc_exp(home) - self._wc_exp(away),
            # squad market-value gap (log-eur); positive => home has the costlier squad
            "value_log_delta": self._log_value(home) - self._log_value(away),
            # Elo gap (positive => home is the stronger side by earned rating)
            "elo_delta": self.elo[home] - self.elo[away],
            # zone-strength gap (positive => home is from the tougher confederation)
            "confed_delta": confed_strength(home) - confed_strength(away),
        }

    def update(self, home: str, away: str, hs: int, as_: int,
               tournament: str = "", year: int | None = None) -> None:
        self.scored[home].append(hs)
        self.scored[away].append(as_)
        self.conceded[home].append(as_)
        self.conceded[away].append(hs)
        self.points[home].append(_result_points(hs, as_))
        self.points[away].append(_result_points(as_, hs))
        # h2h winner from each ordering's perspective (1 if that side won)
        self.h2h[(home, away)].append(1 if hs > as_ else 0)
        self.h2h[(away, home)].append(1 if as_ > hs else 0)
        self._update_elo(home, away, hs, as_, tournament)
        if tournament == WC_NAME and year is not None:
            self.wc_years[home].add(year)
            self.wc_years[away].add(year)

    def _update_elo(self, home: str, away: str, hs: int, as_: int, tournament: str) -> None:
        """World-football-style Elo update: scaled by margin and match importance."""
        rh, ra = self.elo[home], self.elo[away]
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        score_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        margin = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        k = ELO_K * margin * competitiveness(tournament)
        delta = k * (score_h - exp_h)
        self.elo[home] = rh + delta
        self.elo[away] = ra - delta


def build_rank_map(team_df: pd.DataFrame) -> dict:
    """Map canonical team name -> FIFA rank, from the team dataset snapshot."""
    if team_df is None or "rank" not in team_df.columns:
        return {}
    return {
        row["team"]: row["rank"]
        for _, row in team_df.iterrows()
        if pd.notna(row.get("rank"))
    }


def build_value_map(team_df: pd.DataFrame) -> dict:
    """Map canonical team name -> squad total market value (EUR), from the snapshot."""
    col = "squad_total_market_value_eur"
    if team_df is None or col not in team_df.columns:
        return {}
    return {
        row["team"]: row[col]
        for _, row in team_df.iterrows()
        if pd.notna(row.get(col)) and row.get(col)
    }


def make_training_set(history_df: pd.DataFrame, rank_map: dict | None = None,
                      min_year: int = TRAIN_MIN_YEAR, value_map: dict | None = None):
    """Replay full history; emit (X, y, meta) for *all* matches since `min_year`.

    Unlike the WC-only version, this trains on every international match (qualifiers,
    continental cups, friendlies, ...). Importance and recency are not baked in here —
    they are applied as per-row sample weights at fit time (see `sample_weights`), so
    `meta` carries the date/tournament needed for both weighting and walk-forward CV.
    target: 0=Away win, 1=Draw, 2=Home win.
    """
    fb = FeatureBuilder(rank_map=rank_map, value_map=value_map)
    rows, targets, meta = [], [], []

    for r in history_df.itertuples(index=False):
        home, away = r.home_team, r.away_team
        hs, as_ = int(r.home_score), int(r.away_score)
        year = r.date.year
        tournament = getattr(r, "tournament", "")

        if year >= min_year:
            rows.append(fb.features_for(home, away))
            targets.append(2 if hs > as_ else (1 if hs == as_ else 0))
            meta.append({"date": r.date, "year": year, "tournament": tournament})

        fb.update(home, away, hs, as_, tournament, year)

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = np.array(targets)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    return X, y, meta


def fit_full_history(history_df: pd.DataFrame, rank_map: dict | None = None,
                     value_map: dict | None = None) -> FeatureBuilder:
    """Replay the entire history and return the populated builder (for 2026 preds)."""
    fb = FeatureBuilder(rank_map=rank_map, value_map=value_map)
    for r in history_df.itertuples(index=False):
        fb.update(
            r.home_team, r.away_team, int(r.home_score), int(r.away_score),
            getattr(r, "tournament", ""), r.date.year,
        )
    return fb


def build_fixture_features(home: str, away: str, fb: FeatureBuilder) -> pd.DataFrame:
    """Single-row feature frame for one 2026 fixture."""
    return pd.DataFrame([fb.features_for(home, away)], columns=FEATURE_COLS)


if __name__ == "__main__":
    import data_loader as dl

    data = dl.clean()
    rmap = build_rank_map(data["team"])
    vmap = build_value_map(data["team"])
    X, y, meta = make_training_set(data["history"], rmap, value_map=vmap)
    w = sample_weights(meta, ref_date=meta["date"].max())
    print(f"Training rows: {len(X)} | years: {meta['year'].min()}-{meta['year'].max()}")
    print("Class balance (0=Away,1=Draw,2=Home):",
          {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
    print(f"Sample weights: min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")
    print("By tournament tier (weight share):")
    tier = meta["tournament"].map(competitiveness)
    print(tier.value_counts().sort_index().to_string())
    print(X.describe().round(3).to_string())
