"""flags.py — map canonical team names to flag emoji for the report.

Emoji (Unicode regional-indicator) flags keep the report fully self-contained:
no image files or external requests. Unknown teams fall back to a white flag.
"""
from __future__ import annotations

FLAGS = {
    # Group A
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    # B
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    # C
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    # D
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    # E
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    # F
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    # G
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    # H
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    # I
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    # J
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    # K
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    # L
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
    # common play-off / fallback fillers and frequent opponents
    "Italy": "🇮🇹", "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Chile": "🇨🇱",
    "Peru": "🇵🇪", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Serbia": "🇷🇸", "Poland": "🇵🇱",
    "Denmark": "🇩🇰", "Ukraine": "🇺🇦", "Greece": "🇬🇷", "Russia": "🇷🇺",
}

HOST_FLAGS = "🇨🇦 🇺🇸 🇲🇽"  # Canada, USA, Mexico


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳️")
