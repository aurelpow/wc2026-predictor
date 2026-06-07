"""flags.py — render country flags as CSS-backed inline elements.

Each flag's base64 PNG (from flag_assets.FLAG_PNG) is emitted once as a CSS class
via flag_css(); flag_img() then returns a lightweight <span> referencing it. This
keeps the report self-contained (no external requests) and renders on every
platform — including Windows, where emoji flags don't display — without bloating
the file by repeating the image data. Unknown teams fall back to a lettered chip.
"""
from __future__ import annotations

import re

from flag_assets import FLAG_PNG

HOSTS = ["Canada", "United States", "Mexico"]


def _slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(team).lower()).strip("-")


def flag_css() -> str:
    """One background-image rule per flag (emitted once into the <style> block)."""
    return "\n".join(
        f".fl-{_slug(team)}{{background-image:url(data:image/png;base64,{b64})}}"
        for team, b64 in FLAG_PNG.items()
    )


def flag_img(team: str, w: int = 20) -> str:
    if team in FLAG_PNG:
        h = round(w * 0.75)
        return f'<span class="flag fl-{_slug(team)}" style="width:{w}px;height:{h}px"></span>'
    code = "".join(part[0] for part in str(team).split()[:2]).upper() or "?"
    return f'<span class="flagchip">{code}</span>'


def host_flags(w: int = 18) -> str:
    return " ".join(flag_img(t, w) for t in HOSTS)
