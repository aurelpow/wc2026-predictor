"""report.py — render the self-contained HTML tournament report.

Single file, inline CSS, no external dependencies. Color codes probabilities
(green >60%, yellow 40-60%, red <40%), collapsible group sections, and a
print-friendly layout.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from simulator import ROUNDS, expected_group_table

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 0 16px 48px; color: #1b1b1f; background: #f7f8fa; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 22px; margin: 24px 0 4px; }
h2 { font-size: 18px; margin: 28px 0 10px; border-bottom: 2px solid #e3e6ea; padding-bottom: 6px; }
.banner { background: linear-gradient(135deg,#1d3b8b,#2a7ae2); color: #fff;
          padding: 22px 24px; border-radius: 12px; margin: 18px 0 6px; }
.banner .big { font-size: 26px; font-weight: 700; }
.meta { color: #5b6470; font-size: 13px; margin-bottom: 18px; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06); border-radius: 8px; overflow: hidden; }
th, td { padding: 6px 9px; text-align: center; border-bottom: 1px solid #eef0f3; }
th { background: #f0f2f5; font-weight: 600; }
td.team, th.team { text-align: left; font-weight: 600; }
details { background:#fff; border:1px solid #e3e6ea; border-radius:8px; margin:10px 0; padding:6px 12px; }
summary { cursor: pointer; font-weight: 600; font-size: 15px; padding: 6px 0; }
.high { background: #fff7d6; font-weight: 600; }
.g { background:#d6f5dd; } .y { background:#fff3c4; } .r { background:#fbdcdc; }
.win { font-weight: 700; }
.q-1, .q-2 { color:#137333; font-weight:600; } .q-3 { color:#b06000; } .q-4 { color:#a50e0e; }
.note { font-size:12px; color:#5b6470; margin:4px 0 12px; }
.kowrap { display:flex; flex-wrap:wrap; gap:18px; }
.kocol { flex:1; min-width:200px; }
.komatch { background:#fff; border:1px solid #e3e6ea; border-radius:6px; padding:6px 9px; margin:6px 0; font-size:13px; }
.score { color:#5b6470; font-size:12px; }
@media print { body { background:#fff; } details { break-inside: avoid; } details[open] summary ~ * { display:block; } }
"""


def _cls(p: float) -> str:
    return "g" if p > 0.60 else ("y" if p >= 0.40 else "r")


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _esc(s) -> str:
    return html.escape(str(s))


def _group_section(group: str, teams: list, tournament, group_preds: pd.DataFrame) -> str:
    matches = group_preds[group_preds["group"] == group]
    rows = []
    for _, m in matches.iterrows():
        cells = {"Home Win": m["p_home"], "Draw": m["p_draw"], "Away Win": m["p_away"]}
        ml = m["most_likely"]
        tds = []
        for label, val in cells.items():
            hi = " high" if label == ml else ""
            tds.append(f'<td class="{_cls(val)}{hi}">{_pct(val)}</td>')
        score = m["score"] if "score" in m and pd.notna(m["score"]) else ""
        rows.append(
            f"<tr><td class='team'>{_esc(m['home'])}</td>"
            f"<td class='team'>{_esc(m['away'])}</td>{''.join(tds)}"
            f"<td>{_esc(ml)}</td><td class='score'>{_esc(score)}</td></tr>"
        )
    match_table = (
        "<table><tr><th class='team'>Home</th><th class='team'>Away</th>"
        "<th>Home Win</th><th>Draw</th><th>Away Win</th><th>Most likely</th><th>Score</th></tr>"
        + "".join(rows) + "</table>"
    )

    table = expected_group_table(tournament, teams)
    st_rows = []
    for s in table:
        status = {1: "Qualified (1st)", 2: "Qualified (2nd)",
                  3: "3rd — playoff contender", 4: "Eliminated"}[s["finish"]]
        st_rows.append(
            f"<tr><td class='team'>{_esc(s['team'])}</td>"
            f"<td>{s['W']:.1f}</td><td>{s['D']:.1f}</td><td>{s['L']:.1f}</td>"
            f"<td>{s['GF']:.1f}</td><td>{s['GA']:.1f}</td><td>{s['GD']:+.1f}</td>"
            f"<td><b>{s['Pts']:.1f}</b></td><td class='q-{s['finish']}'>{status}</td></tr>"
        )
    standings = (
        "<table><tr><th class='team'>Team</th><th>W</th><th>D</th><th>L</th>"
        "<th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>Qualification</th></tr>"
        + "".join(st_rows) + "</table>"
    )

    return (
        f"<details><summary>Group {group} — {_esc(' · '.join(teams))}</summary>"
        f"<p class='note'>Match probabilities (most-likely outcome highlighted):</p>{match_table}"
        f"<p class='note'>Expected final standings (decimal = expected value over the group):</p>{standings}"
        f"</details>"
    )


def _knockout_section(bracket: pd.DataFrame) -> str:
    cols = []
    for rnd in ROUNDS[:-1]:  # R32..Final
        sub = bracket[bracket["round"] == rnd]
        if sub.empty:
            continue
        cards = []
        for _, m in sub.iterrows():
            ph, pa = m["p_home_win"], m["p_away_win"]
            home_c = "win" if m["predicted_winner"] == m["home"] else ""
            away_c = "win" if m["predicted_winner"] == m["away"] else ""
            cards.append(
                f"<div class='komatch'>"
                f"<span class='{home_c}'>{_esc(m['home'])}</span> "
                f"<span class='{_cls(ph)}'>{_pct(ph)}</span> vs "
                f"<span class='{_cls(pa)}'>{_pct(pa)}</span> "
                f"<span class='{away_c}'>{_esc(m['away'])}</span>"
                f"<br><span class='score'>predicted {_esc(m['predicted_winner'])} · {_esc(m['score'])}</span>"
                f"</div>"
            )
        cols.append(f"<div class='kocol'><h3>{rnd}</h3>{''.join(cards)}</div>")
    return f"<div class='kowrap'>{''.join(cols)}</div>"


def _paths_table(summary: pd.DataFrame, finish_counts: dict, n: int) -> str:
    rows = []
    for _, r in summary.iterrows():
        fc = finish_counts.get(r["team"], {})
        total = sum(fc.values()) or 1
        best_pos = max(fc, key=fc.get) if fc else 0
        finish_lbl = f"{best_pos}{'st' if best_pos==1 else 'nd' if best_pos==2 else 'rd' if best_pos==3 else 'th'} ({fc.get(best_pos,0)/total*100:.0f}%)" if best_pos else "-"
        cells = "".join(
            f"<td class='{_cls(r[c])}'>{_pct(r[c])}</td>"
            for c in ["P(R32)", "P(R16)", "P(QF)", "P(SF)", "P(Final)", "P(Win)"]
        )
        rows.append(
            f"<tr><td class='team'>{_esc(r['team'])}</td>"
            f"<td>{_esc(r['group'])}</td><td>{finish_lbl}</td>{cells}</tr>"
        )
    return (
        "<table><tr><th class='team'>Team</th><th>Grp</th><th>Group finish</th>"
        "<th>R32</th><th>R16</th><th>QF</th><th>SF</th><th>Final</th><th>Winner</th></tr>"
        + "".join(rows) + "</table>"
    )


def build_html_report(summary: pd.DataFrame, group_preds: pd.DataFrame,
                      bracket: pd.DataFrame, tournament, finish_counts: dict,
                      n_sims: int, logloss: float, group_source: str,
                      out_path: Path) -> None:
    top = summary.iloc[0]
    groups_html = "".join(
        _group_section(g, tournament.groups[g], tournament, group_preds)
        for g in sorted(tournament.groups)
    )
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC2026 Prediction Report</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<div class="banner"><div>🏆 Most likely winner:</div>
<div class="big">{_esc(top['team'])} ({_pct(top['P(Win)'])})</div></div>
<div class="meta">Monte Carlo: {n_sims:,} simulations · Walk-forward log-loss: {logloss:.4f}
 · Groups from: {_esc(group_source)} · Color: <span class="g">&nbsp;&gt;60%&nbsp;</span>
 <span class="y">&nbsp;40–60%&nbsp;</span> <span class="r">&nbsp;&lt;40%&nbsp;</span></div>

<h2>Group Stage</h2>{groups_html}

<h2>Knockout Bracket</h2>
<p class="note">A single <b>"chalk" bracket</b> built from each group's expected finishers, with the
favorite advancing each round (win probability per side · <b>predicted winner</b> · most-likely
winning score). The banner champion above comes from the {n_sims:,}-simulation Monte Carlo and
can differ — it averages over <i>all</i> possible draws and upset paths, not just this one.</p>
{_knockout_section(bracket)}

<h2>Tournament Path — all teams (Monte Carlo %)</h2>
{_paths_table(summary, finish_counts, n_sims)}

</div></body></html>"""
    out_path.write_text(doc, encoding="utf-8")
