"""data_loader.py — download and clean the datasets for the WC2026 predictor.

Datasets (Kaggle slugs from prompt.md):
  1. martj42/international-football-results-from-1872-to-2017  -> historical matches
  2. harrachimustapha/fifa-world-cup-team-dataset              -> WC team features
  3. areezvisram12/fifa-world-cup-2026-match-data-unofficial   -> 2026 fixtures

Run standalone:  python data_loader.py
Requires Kaggle credentials at ~/.kaggle/kaggle.json (see README).
"""
from __future__ import annotations

import glob
import os
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "history": "martj42/international-football-results-from-1872-to-2017",
    "team": "harrachimustapha/fifa-world-cup-team-dataset",
    "fixtures": "areezvisram12/fifa-world-cup-2026-match-data-unofficial",
}

HISTORY_START_YEAR = 1990  # filter historical matches to the modern era

# Variant spellings -> the canonical name used by the historical results dataset
# (which is the feature space). Matching is accent/punctuation-insensitive (see
# _canon_key), so e.g. "Côte d'Ivoire", "Cote d'Ivoire" both resolve.
TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Turkiye": "Turkey",
    "Turkiye ": "Turkey",
}


def _canon_key(name: str) -> str:
    """Accent-stripped, lowercased, alnum-only key for robust alias matching."""
    decomposed = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(ch for ch in no_accents.lower() if ch.isalnum())


_ALIAS_BY_KEY = {_canon_key(k): v for k, v in TEAM_ALIASES.items()}


def normalize_team(name) -> str:
    """Map a team name to its canonical (historical-dataset) spelling."""
    if not isinstance(name, str):
        return name
    n = name.strip()
    return _ALIAS_BY_KEY.get(_canon_key(n), n)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.exists()


def download_datasets(data_dir: Path = DATA_DIR, force: bool = False) -> None:
    """Download + unzip the three datasets via the Kaggle API. Idempotent."""
    data_dir.mkdir(parents=True, exist_ok=True)

    if not _kaggle_credentials_present():
        raise SystemExit(
            "Kaggle credentials not found.\n"
            "  Place your API token at ~/.kaggle/kaggle.json "
            "(Account -> Settings -> Create New Token),\n"
            "  or set KAGGLE_USERNAME / KAGGLE_KEY environment variables."
        )

    # Import here so the module imports fine even when kaggle isn't configured.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    for key, slug in DATASETS.items():
        dest = data_dir / key
        if dest.exists() and any(dest.glob("*.csv")) and not force:
            print(f"[skip] {slug} already downloaded -> {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[download] {slug} -> {dest}")
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
        # Some datasets land as a zip even with unzip=True; expand any leftovers.
        for z in dest.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _find_csv(subdir: str, *name_hints: str) -> Path:
    """Locate a CSV inside data/<subdir>, preferring files matching name hints."""
    base = DATA_DIR / subdir
    csvs = sorted(Path(p) for p in glob.glob(str(base / "**" / "*.csv"), recursive=True))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found under {base}. Run download_datasets() first."
        )
    if name_hints:
        for hint in name_hints:
            for c in csvs:
                if hint.lower() in c.name.lower():
                    return c
    # Fall back to the largest CSV (usually the main table).
    return max(csvs, key=lambda p: p.stat().st_size)


def load_historical() -> pd.DataFrame:
    """Load + clean historical international results, filtered to the modern era."""
    path = _find_csv("history", "results")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns: date, home_team, away_team, home_score, away_score,
    # tournament, city, country, neutral
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df[df["date"].dt.year >= HISTORY_START_YEAR].copy()

    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"
    if "neutral" not in df.columns:
        df["neutral"] = False

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_team_features() -> pd.DataFrame:
    """Load the WC team dataset (train = past editions, test = 2026 entrants).

    Combines both, coalesces the ranking column (`rank` / `fifa_rank_pre_tournament`),
    and keeps the most recent edition per team. Returns one row per canonical team.
    """
    frames = []
    for fname in ("train.csv", "test.csv"):
        p = DATA_DIR / "team" / fname
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:  # fallback: any CSV in the team dir
        frames = [pd.read_csv(_find_csv("team"))]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    team_col = next(
        (c for c in df.columns if c in ("team", "country", "nation", "team_name")),
        df.columns[0],
    )
    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].map(normalize_team)

    # Coalesce the ranking column across the two files.
    rank_cols = [c for c in ("rank", "fifa_rank_pre_tournament") if c in df.columns]
    if rank_cols:
        df["rank"] = pd.to_numeric(df[rank_cols[0]], errors="coerce")
        for extra in rank_cols[1:]:
            df["rank"] = df["rank"].fillna(pd.to_numeric(df[extra], errors="coerce"))

    if "version" not in df.columns:
        df["version"] = 0
    df = (df.sort_values("version")
            .drop_duplicates(subset="team", keep="last")
            .reset_index(drop=True))
    return df


def load_fixtures() -> pd.DataFrame:
    """Load the WC2026 group draw from the fixtures dataset's teams.csv.

    Returns one row per slot with columns: team, group, is_placeholder. Six slots
    are still TBD playoff winners (is_placeholder=True); main.build_groups fills
    those with plausible entrants.
    """
    path = DATA_DIR / "fixtures" / "teams.csv"
    if not path.exists():
        path = _find_csv("fixtures", "teams")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {}
    for c in df.columns:
        if c in ("team_name", "team", "nation", "country"):
            rename[c] = "team"
        elif c in ("group_letter", "group"):
            rename[c] = "group"
    df = df.rename(columns=rename)

    df["team"] = df["team"].map(normalize_team)
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    if "is_placeholder" not in df.columns:
        df["is_placeholder"] = False
    df["is_placeholder"] = df["is_placeholder"].astype(bool)
    return df[["team", "group", "is_placeholder"]]


def clean() -> dict[str, pd.DataFrame]:
    """Convenience: load all three sources and return them in a dict."""
    return {
        "history": load_historical(),
        "team": load_team_features(),
        "fixtures": load_fixtures(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    download_datasets()
    for name, df in clean().items():
        print(f"\n=== {name} === shape={df.shape}")
        print("columns:", list(df.columns))
        print(df.head(3).to_string())
