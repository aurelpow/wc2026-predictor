# FIFA World Cup 2026 — Match Outcome Prediction System

Predicts Home/Draw/Away outcomes for the WC2026 fixtures, then runs a Monte Carlo
simulation of the full tournament (official **12 groups of 4 → 72 group matches →
32-team knockout → 104 total** format) to produce per-team win probabilities and a
self-contained HTML bracket report.

### 📊 [**View the live report →**](https://aurelpow.github.io/wc2026-predictor/)

## Project structure

```
wc2026-predictor/
├── src/                 # pipeline modules
│   ├── data_loader.py   # download + clean the 3 Kaggle datasets; team-name normalization
│   ├── feature_engineering.py  # as-of (leak-free) features: Elo, form, attack/defense, ...
│   ├── model.py         # calibrated XGBoost + recency/competitiveness weighting + backtest
│   ├── simulator.py     # group stage + knockout + Monte Carlo (10,000 runs)
│   ├── report.py        # single-file HTML report generator
│   └── main.py          # orchestrates the full pipeline (entry point)
├── tests/smoke_test.py  # offline end-to-end check on synthetic data (no Kaggle needed)
├── docs/index.html      # published report (served by GitHub Pages)
├── requirements.txt
├── data/                # downloaded datasets (gitignored)
└── results/             # generated CSVs / model / report (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

### Kaggle credentials (required for real data)
1. Kaggle → Account → Settings → **Create New Token** → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
   Or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars.

> This project pins `kaggle<1.7` on purpose — the classic CLI uses `kaggle.json`
> (username/key). Kaggle 2.x moved to an OAuth flow that authenticates on import and
> breaks the `~/.kaggle/kaggle.json` workflow.

## Run

```bash
.venv\Scripts\python src/main.py            # full: 10,000 simulations
.venv\Scripts\python src/main.py --debug    # fast: 100 simulations
.venv\Scripts\python src/main.py --seed 7   # different seed -> different bracket/outcomes
```
Run commands from the project root.

Runs are reproducible: the model (`random_state`) and Monte Carlo share `--seed`
(default 42), so the same seed always yields the same report. Change `--seed` to
explore alternative tournament realizations.

Prints the **top-10 contenders by P(Win)** and writes to `results/`:

- `monte_carlo_summary.csv` — main output: P(Qualify/R32/R16/QF/SF/Final/Win) per team
- `group_stage_predictions.csv` — per-match Home/Draw/Away probabilities
- `knockout_bracket.csv` — representative bracket with win % and modal scores
- `feature_importance.png` — XGBoost feature importances
- `full_tournament_report.html` — visual report (also published to `docs/index.html`)

Each module can also be run standalone (e.g. `python src/model.py`) once data is downloaded.

### Offline self-check
`python tests/smoke_test.py` runs the full pipeline on **synthetic** data (no Kaggle
needed) — useful to confirm the install works end-to-end.

## Modeling choices & caveats
- **Opponent-adjusted strength (Elo):** the headline feature. Each team carries a
  running Elo rating updated after every match, scaled by goal margin and match
  importance. Unlike raw form/attack/defense, Elo rewards beating *strong* opponents —
  so a team that piles up goals against weak opposition is no longer mistaken for an
  elite side. This is what brings the favorites in line with expert/bookmaker consensus.
- **Squad market value:** log market-value gap, a static squad-quality anchor for the
  48 entrants (defaults to the median for teams missing a value).
- **Confederation strength (`confed_delta`):** a zone-toughness gap so a result counts
  for how hard the opponent's confederation is — Europe (UEFA) and South America
  (CONMEBOL) are rated toughest, Oceania weakest. This corrects "soft-schedule"
  inflation (e.g. a side padding wins in a weaker zone). Edit `src/confederations.py`
  to retune the zone scores.
- **Training data:** international matches since **2014** (qualifiers, continental
  cups, friendlies, World Cups) — not just World Cup games. Elo/form features still warm
  up from the full history; only the *training rows* start in 2014. Each match gets a
  per-row **sample weight = recency × competitiveness**:
  - *Recency:* exponential time decay with a 4-year half-life, so a game 12 years ago
    counts ~⅛ as much as a recent one (and early-2026 matches count most).
  - *Competitiveness:* friendly 0.2 · other competitive 0.65 · qualifier/Nations League
    0.85 · major tournament final 1.0 — official games are worth up to 5× a friendly.
  Tune all three in `feature_engineering.py` (`TRAIN_MIN_YEAR`, `RECENCY_HALFLIFE_YEARS`,
  `competitiveness`).
- **Backtest:** walk-forward over recent World Cups — for each edition the model trains
  only on *prior* matches, then is scored on that edition (no future leakage).
- **Outcome model** is the source of truth for W/D/L. Goals (for GD/GF tiebreakers and
  the report's "most-frequent score") come from a Poisson model conditioned on the
  sampled outcome.
- **Neutral venues:** match probabilities are *symmetrized* over both home/away
  orderings to remove orientation bias; knockout draws are redistributed proportionally.
- **Knockout bracket** seeds qualifiers (group finish, then Pts/GD/GF) into a standard
  single-elimination bracket — a transparent approximation of FIFA's official slot table
  (which depends on a 495-row third-place combination matrix).
- **FIFA ranking** is a single current snapshot used as a static team-strength prior,
  including for historical training rows (an acknowledged approximation).
- **Groups** are read from the fixtures dataset; if its schema doesn't yield 12×4, the
  code falls back to a rank-seeded snake draw of the top 48 teams.
- Outcome prediction in football is inherently noisy (walk-forward log-loss ≈ 1.0 vs the
  1.099 uniform baseline) — treat the probabilities as reasonable estimates, not forecasts.
