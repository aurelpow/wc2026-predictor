"""report.py — render the self-contained HTML tournament report.

Single file, inline CSS, emoji flags — no external dependencies (works offline,
on GitHub Pages, or as an email attachment). Poster-style layout: champion hero,
podium, contender cards, collapsible group sections, a flagged knockout bracket,
and a per-team path table. Probabilities are colour-coded (green >60%, amber
40-60%, red <40%) and the layout is print-friendly.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from flags import flag_css, flag_img, host_flags
from simulator import ROUNDS, expected_group_table

_CSS = """
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 0 16px 60px; color: #11161d; background: #eef1f5; }
.wrap { max-width: 1100px; margin: 0 auto; }
h2 { font-size: 20px; margin: 34px 0 14px; display: flex; align-items: center; gap: 8px; }
h2::before { content: ""; width: 6px; height: 22px; border-radius: 3px;
             background: linear-gradient(180deg,#0b6e4f,#2a7ae2); display: inline-block; }
a { color: #1d3b8b; }

/* hero */
.hero { background: linear-gradient(135deg,#0b6e4f 0%,#1d3b8b 55%,#2a7ae2 100%);
        color: #fff; border-radius: 20px; padding: 30px 26px; margin: 22px 0 10px;
        text-align: center; box-shadow: 0 12px 30px rgba(13,40,80,.28); position: relative; overflow: hidden; }
.hero::after { content: "🏆"; position: absolute; right: -10px; top: -20px; font-size: 150px; opacity: .10; }
.hero .tag { letter-spacing: 2px; text-transform: uppercase; font-size: 12px; opacity: .9; }
.hero h1 { font-size: 32px; margin: 6px 0 18px; }
.champ { display: inline-flex; align-items: center; gap: 18px; background: rgba(255,255,255,.12);
         border: 1px solid rgba(255,255,255,.25); padding: 14px 26px; border-radius: 16px; backdrop-filter: blur(4px); }
.champ .cflag { display: flex; align-items: center; }
.champ .clabel { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: .85; }
.champ .cname { font-size: 30px; font-weight: 800; }
.champ .cpct { font-size: 15px; color: #ffe27a; font-weight: 700; }
.hero .hmeta { margin-top: 16px; font-size: 12.5px; opacity: .85; }

/* podium */
.podium { display: flex; justify-content: center; align-items: flex-end; gap: 14px; margin: 6px 0 4px; }
.step { border-radius: 14px 14px 0 0; padding: 12px 10px 14px; text-align: center; color: #1b1b1f;
        width: 150px; box-shadow: 0 6px 16px rgba(0,0,0,.12);
        display: flex; flex-direction: column; align-items: center; justify-content: flex-end; gap: 3px; }
.step .pflag { margin: 2px 0; } .step .pname { font-weight: 700; }
.step .ppct { font-weight: 800; font-size: 18px; } .step .pmedal { font-size: 22px; }
.s1 { background: linear-gradient(180deg,#ffe9a3,#f4b400); min-height: 176px; }
.s2 { background: linear-gradient(180deg,#eef1f5,#c7ced8); min-height: 150px; }
.s3 { background: linear-gradient(180deg,#f4d8bf,#cd7f4d); min-height: 130px; }

/* contender cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill,minmax(160px,1fr)); gap: 14px; }
.card { background: #fff; border-radius: 16px; padding: 16px 14px; box-shadow: 0 4px 14px rgba(20,30,50,.08);
        text-align: center; border: 1px solid #e7ebf0; }
.card .rank { float: left; font-size: 12px; color: #98a2b3; font-weight: 700; }
.card .cf { margin-bottom: 4px; } .card .cn { font-weight: 700; margin: 6px 0 2px; }
.flag { display: inline-block; background-size: cover; background-position: center;
        border-radius: 3px; vertical-align: -3px; box-shadow: 0 0 0 1px rgba(0,0,0,.12); }
.cflag .flag, .pflag .flag, .cf .flag { box-shadow: 0 2px 6px rgba(0,0,0,.28); border-radius: 4px; vertical-align: middle; }
.gflags { display: inline-flex; gap: 3px; align-items: center; } .gflags .flag { box-shadow: 0 0 0 1px rgba(0,0,0,.1); }
.flagchip { display: inline-block; background: #1d3b8b; color: #fff; border-radius: 4px;
            padding: 1px 5px; font-size: 11px; font-weight: 700; vertical-align: 1px; }
.card .grp { font-size: 11px; color: #8a94a6; } .card .pw { font-size: 22px; font-weight: 800; margin-top: 6px; }
.card .sub { font-size: 11px; color: #5b6470; margin-top: 2px; }
.bar { background: #eceff3; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 8px; }
.bar > i { display: block; height: 100%; background: linear-gradient(90deg,#0b6e4f,#2a7ae2); }

/* tables */
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px;
        box-shadow: 0 2px 8px rgba(20,30,50,.06); border-radius: 10px; overflow: hidden; }
th, td { padding: 7px 9px; text-align: center; border-bottom: 1px solid #eef0f3; }
th { background: #f3f5f8; font-weight: 700; font-size: 12px; }
td.team, th.team { text-align: left; font-weight: 600; white-space: nowrap; }
.high { background: #fff3c4; font-weight: 700; }
.g { background:#d6f5dd; } .y { background:#fff3c4; } .r { background:#fbe2e2; }
.win { font-weight: 800; color: #0b6e4f; }
.q-1,.q-2 { color:#137333; font-weight:700; } .q-3 { color:#b06000; font-weight:600; } .q-4 { color:#a50e0e; }
.sbadge { display:inline-block; background:#11161d; color:#fff; border-radius:6px; padding:1px 7px;
          font-weight:700; font-size:12px; }

/* groups */
details { background:#fff; border:1px solid #e7ebf0; border-radius:14px; margin:12px 0; padding:4px 16px;
          box-shadow: 0 2px 8px rgba(20,30,50,.05); }
summary { cursor:pointer; font-weight:700; font-size:15px; padding:12px 0; list-style:none; display:flex; align-items:center; gap:10px; }
summary::-webkit-details-marker { display:none; }
.gbadge { background:linear-gradient(135deg,#0b6e4f,#2a7ae2); color:#fff; width:30px; height:30px;
          border-radius:9px; display:inline-flex; align-items:center; justify-content:center; font-weight:800; }
.note { font-size:12px; color:#5b6470; margin:6px 0 10px; }

/* knockout */
.kowrap { display:flex; gap:16px; overflow-x:auto; padding-bottom:6px; }
.kocol { flex:1; min-width:210px; }
.kocol h3 { font-size:13px; text-transform:uppercase; letter-spacing:1px; color:#5b6470; margin:0 0 8px; text-align:center; }
.komatch { background:#fff; border:1px solid #e7ebf0; border-radius:12px; padding:10px 12px; margin:8px 0;
           box-shadow:0 2px 8px rgba(20,30,50,.05); }
.koside { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:2px 0; font-size:13px; }
.koside .nm { display:flex; align-items:center; gap:7px; }
.koside .wp { font-variant-numeric: tabular-nums; color:#5b6470; font-size:12px; }
.kosep { text-align:center; color:#aeb6c2; font-size:11px; margin:3px 0; }
.kowin { font-weight:800; color:#0b6e4f; }

footer { margin-top:34px; font-size:12px; color:#5b6470; text-align:center; }
.legend span { display:inline-block; padding:2px 8px; border-radius:6px; margin:0 3px; }
.credit { margin:22px auto 6px; padding:18px; max-width:560px; background:#fff; border:1px solid #e7ebf0;
          border-radius:16px; box-shadow:0 4px 14px rgba(20,30,50,.06); }
.credit .by { font-size:15px; color:#11161d; } .credit .by strong { font-weight:800; }
.credit .links { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-top:12px; }
.credit a { text-decoration:none; color:#fff; font-weight:700; font-size:13px; padding:8px 14px;
            border-radius:10px; display:inline-flex; align-items:center; gap:6px; }
.credit a.gh { background:#24292f; } .credit a.li { background:#0a66c2; }
.credit a.pf { background:linear-gradient(135deg,#0b6e4f,#2a7ae2); }
@media print { body { background:#fff; } .hero { box-shadow:none; } details { break-inside:avoid; }
               details[open] summary ~ * { display:block; } .kowrap { overflow:visible; flex-wrap:wrap; } }
"""


def _cls(p: float) -> str:
    return "g" if p > 0.60 else ("y" if p >= 0.40 else "r")


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _esc(s) -> str:
    return html.escape(str(s))


def _tflag(team: str) -> str:
    return f"{flag_img(team)} {_esc(team)}"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _hero(top, n_sims: int, logloss: float) -> str:
    return f"""<header class="hero">
  <div class="tag">FIFA World Cup · {host_flags()} · USA · Mexico · Canada 2026</div>
  <h1>Tournament Predictions</h1>
  <div class="champ"><span class="cflag">{flag_img(top['team'], 64)}</span>
    <div><div class="clabel">Most likely champion</div>
      <div class="cname">{_esc(top['team'])}</div>
      <div class="cpct">🏆 {_pct(top['P(Win)'])} to win it all</div></div></div>
  <div class="hmeta">{n_sims:,} Monte Carlo simulations · model walk-forward log-loss {logloss:.3f}</div>
</header>"""


def _podium(summary: pd.DataFrame) -> str:
    t = summary.head(3).reset_index(drop=True)
    if len(t) < 3:
        return ""
    g, s, b = t.iloc[0], t.iloc[1], t.iloc[2]

    def step(row, cls, medal):
        return (f"<div class='step {cls}'><div class='pmedal'>{medal}</div>"
                f"<div class='pflag'>{flag_img(row['team'], 48)}</div>"
                f"<div class='pname'>{_esc(row['team'])}</div>"
                f"<div class='ppct'>{_pct(row['P(Win)'])}</div></div>")

    return ("<section class='podium'>"
            + step(s, "s2", "🥈") + step(g, "s1", "🥇") + step(b, "s3", "🥉")
            + "</section>")


def _contenders(summary: pd.DataFrame, k: int = 12) -> str:
    top = summary.head(k)
    pmax = float(top["P(Win)"].max()) or 1.0
    cards = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        width = r["P(Win)"] / pmax * 100
        cards.append(
            f"<div class='card'><span class='rank'>#{i}</span>"
            f"<div class='cf'>{flag_img(r['team'], 46)}</div>"
            f"<div class='cn'>{_esc(r['team'])}</div><div class='grp'>Group {_esc(r['group'])}</div>"
            f"<div class='pw'>{_pct(r['P(Win)'])}</div>"
            f"<div class='sub'>Final {_pct(r['P(Final)'])} · SF {_pct(r['P(SF)'])}</div>"
            f"<div class='bar'><i style='width:{width:.0f}%'></i></div></div>"
        )
    return f"<section class='cards'>{''.join(cards)}</section>"


def _group_section(group: str, teams: list, tournament, group_preds: pd.DataFrame) -> str:
    matches = group_preds[group_preds["group"] == group]
    rows = []
    for _, m in matches.iterrows():
        cells = {"Home Win": m["p_home"], "Draw": m["p_draw"], "Away Win": m["p_away"]}
        ml = m["most_likely"]
        tds = "".join(
            f'<td class="{_cls(v)}{" high" if lab==ml else ""}">{_pct(v)}</td>'
            for lab, v in cells.items()
        )
        score = m["score"] if "score" in m and pd.notna(m["score"]) else ""
        rows.append(
            f"<tr><td class='team'>{_tflag(m['home'])}</td><td class='team'>{_tflag(m['away'])}</td>"
            f"{tds}<td><span class='sbadge'>{_esc(score)}</span></td></tr>"
        )
    match_table = (
        "<table><tr><th class='team'>Home</th><th class='team'>Away</th>"
        "<th>Home</th><th>Draw</th><th>Away</th><th>Score</th></tr>" + "".join(rows) + "</table>"
    )

    st_rows = []
    for s in expected_group_table(tournament, teams):
        status = {1: "✅ Qualified (1st)", 2: "✅ Qualified (2nd)",
                  3: "↘ 3rd — playoff", 4: "❌ Eliminated"}[s["finish"]]
        st_rows.append(
            f"<tr><td class='team'>{_tflag(s['team'])}</td>"
            f"<td>{s['W']:.1f}</td><td>{s['D']:.1f}</td><td>{s['L']:.1f}</td>"
            f"<td>{s['GF']:.1f}</td><td>{s['GA']:.1f}</td><td>{s['GD']:+.1f}</td>"
            f"<td><b>{s['Pts']:.1f}</b></td><td class='q-{s['finish']}'>{status}</td></tr>"
        )
    standings = (
        "<table><tr><th class='team'>Team</th><th>W</th><th>D</th><th>L</th>"
        "<th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>Qualification</th></tr>"
        + "".join(st_rows) + "</table>"
    )

    flags = " ".join(flag_img(t, 22) for t in teams)
    return (
        f"<details><summary><span class='gbadge'>{group}</span> Group {group}"
        f"<span class='gflags'>{flags}</span></summary>"
        f"<p class='note'>Match probabilities — most-likely outcome highlighted; score allows draws:</p>{match_table}"
        f"<p class='note'>Expected final standings (decimal = expected value over the group):</p>{standings}"
        f"</details>"
    )


def _knockout_section(bracket: pd.DataFrame) -> str:
    titles = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
              "SF": "Semi-finals", "Final": "Final"}
    cols = []
    for rnd in ROUNDS[:-1]:
        sub = bracket[bracket["round"] == rnd]
        if sub.empty:
            continue
        cards = []
        for _, m in sub.iterrows():
            ph, pa = m["p_home_win"], m["p_away_win"]
            hw = "kowin" if m["predicted_winner"] == m["home"] else ""
            aw = "kowin" if m["predicted_winner"] == m["away"] else ""
            cards.append(
                f"<div class='komatch'>"
                f"<div class='koside'><span class='nm {hw}'>{_tflag(m['home'])}</span><span class='wp'>{_pct(ph)}</span></div>"
                f"<div class='kosep'>— {_esc(m['score'])} —</div>"
                f"<div class='koside'><span class='nm {aw}'>{_tflag(m['away'])}</span><span class='wp'>{_pct(pa)}</span></div>"
                f"</div>"
            )
        cols.append(f"<div class='kocol'><h3>{titles.get(rnd, rnd)}</h3>{''.join(cards)}</div>")
    return f"<div class='kowrap'>{''.join(cols)}</div>"


def _paths_table(summary: pd.DataFrame, finish_counts: dict) -> str:
    rows = []
    for _, r in summary.iterrows():
        fc = finish_counts.get(r["team"], {})
        total = sum(fc.values()) or 1
        bp = max(fc, key=fc.get) if fc else 0
        sfx = {1: "st", 2: "nd", 3: "rd"}.get(bp, "th")
        finish_lbl = f"{bp}{sfx} ({fc.get(bp, 0) / total * 100:.0f}%)" if bp else "-"
        cells = "".join(
            f"<td class='{_cls(r[c])}'>{_pct(r[c])}</td>"
            for c in ["P(R32)", "P(R16)", "P(QF)", "P(SF)", "P(Final)", "P(Win)"]
        )
        rows.append(
            f"<tr><td class='team'>{_tflag(r['team'])}</td>"
            f"<td>{_esc(r['group'])}</td><td>{finish_lbl}</td>{cells}</tr>"
        )
    return (
        "<table><tr><th class='team'>Team</th><th>Grp</th><th>Group finish</th>"
        "<th>R32</th><th>R16</th><th>QF</th><th>SF</th><th>Final</th><th>🏆 Win</th></tr>"
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
<style>{_CSS}
{flag_css()}</style></head>
<body><div class="wrap">
{_hero(top, n_sims, logloss)}
{_podium(summary)}

<h2>Title contenders</h2>
{_contenders(summary)}

<h2>Group stage</h2>
{groups_html}

<h2>Knockout bracket</h2>
<p class="note">A single "chalk" bracket built from each group's expected finishers, favorite advancing each round
(win % per side · <b>winner in green</b> · most-likely score). The champion above comes from the full
{n_sims:,}-simulation Monte Carlo and can differ — it averages over all draws and upset paths.</p>
{_knockout_section(bracket)}

<h2>Tournament path — all 48 teams</h2>
{_paths_table(summary, finish_counts)}

<div class="credit">
  <div class="by">Built by <strong>Aurélien Darracq</strong> · <span style="color:#5b6470">@aurelpow</span></div>
  <div class="links">
    <a class="gh" href="https://github.com/aurelpow" target="_blank" rel="noopener">⌨ GitHub</a>
    <a class="li" href="https://www.linkedin.com/in/aur%C3%A9lien-darracq/" target="_blank" rel="noopener">in LinkedIn</a>
    <a class="pf" href="https://aurelpow.github.io/portofolio-website/" target="_blank" rel="noopener">🌐 Portfolio</a>
  </div>
</div>
<footer>
<p class="legend">Probability colour key:
<span class="g">&gt; 60%</span><span class="y">40–60%</span><span class="r">&lt; 40%</span></p>
<p>Groups: {_esc(group_source)} · Generated by the WC2026 predictor · estimates, not guarantees ⚽</p>
</footer>
</div></body></html>"""
    out_path.write_text(doc, encoding="utf-8")
