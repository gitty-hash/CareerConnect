"""Job-portal styling for the Streamlit app (LinkedIn-inspired: warm gray page, white cards,
blue accents, pill buttons, company-logo avatars).

Cards are Streamlit containers created with a key that starts with "card"; Streamlit adds a
`st-key-<key>` CSS class to them, which the stylesheet below turns into white cards.
"""

import html
import re

import streamlit as st

BLUE = "#0a66c2"

CATEGORY_COLORS = {
    "Data Analytics": "#0a66c2",
    "Software Engineering": "#7c3aed",
    "Product Management": "#c2185b",
    "Marketing": "#e8590c",
    "UX/UI Design": "#0d9488",
    "Finance": "#057642",
    "Cybersecurity": "#cc1016",
    "Operations & Supply Chain": "#915907",
}

LOGO_COLORS = ["#0a66c2", "#057642", "#7c3aed", "#c2185b", "#e8590c", "#0d9488", "#915907", "#004182", "#5f4b8b"]

CSS = """
<style>
html, body, .stApp, button, input, textarea, select {
  font-family: -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.block-container {padding-top: 1rem; padding-bottom: 3rem; max-width: 1200px;}
header[data-testid="stHeader"] {background: transparent; height: 2.2rem;}

/* ---------- top bar ---------- */
.cc-topbar {display:flex; align-items:center; gap:14px; background:#fff; border:1px solid #e0dfdc;
  border-radius:10px; padding:10px 18px; margin-bottom:10px;}
.cc-logo {width:38px; height:38px; border-radius:6px; background:#0a66c2; color:#fff; font-weight:800;
  display:flex; align-items:center; justify-content:center; font-size:17px; letter-spacing:-.5px;}
.cc-brand {font-size:1.25rem; font-weight:700; color:#191919; line-height:1.1;}
.cc-brand small {display:block; font-size:.78rem; font-weight:400; color:#666;}
.cc-spacer {flex:1;}
.cc-mode {font-size:.78rem; font-weight:600; padding:5px 12px; border-radius:999px;}
.cc-mode.demo {background:#fff4e5; color:#915907; border:1px solid #f5c26b;}
.cc-mode.live {background:#e6f4ea; color:#057642; border:1px solid #9bd3ae;}

/* ---------- tabs as a nav bar ---------- */
.stTabs [data-baseweb="tab-list"] {gap:2px; background:#fff; border:1px solid #e0dfdc; border-radius:10px;
  padding:0 10px; margin-bottom:14px;}
.stTabs [data-baseweb="tab"] {height:50px; padding:0 16px; font-weight:600; color:#666; background:transparent;}
.stTabs [data-baseweb="tab"]:hover {color:#191919;}
.stTabs [aria-selected="true"] {color:#191919 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#191919; height:2px;}
.stTabs [data-baseweb="tab-border"] {display:none;}
/* nested (sub) tabs: lighter */
.stTabs .stTabs [data-baseweb="tab-list"] {border:none; background:transparent; padding:0; margin-bottom:6px;}
.stTabs .stTabs [data-baseweb="tab"] {height:38px; padding:0 12px; font-size:.9rem;}
.stTabs .stTabs [data-baseweb="tab-highlight"] {background-color:#0a66c2;}

/* ---------- cards ---------- */
[class*="st-key-card"] {background:#fff; border:1px solid #e0dfdc !important; border-radius:10px;
  padding:16px 18px !important;}
[class*="st-key-cardsel"] {border:2px solid #0a66c2 !important; background:#f5f9fe;}

/* ---------- buttons & inputs ---------- */
.stButton button, .stFormSubmitButton button, .stPopover button, .stDownloadButton button {
  border-radius:999px !important; font-weight:600;}
/* inputs: visible outlines on white cards; the main search box is a LinkedIn-style pill */
.stTextInput div[data-baseweb="input"], .stTextArea div[data-baseweb="textarea"],
.stSelectbox div[data-baseweb="select"] > div {border:1px solid #c7c7c7 !important; border-radius:8px;}
.stTextInput div[data-baseweb="input"]:focus-within, .stTextArea div[data-baseweb="textarea"]:focus-within
  {border-color:#0a66c2 !important; box-shadow:0 0 0 1px #0a66c2;}
.st-key-card_search .stTextInput div[data-baseweb="input"] {border-radius:999px; background:#edf3f8;
  border-color:#d0dbe6 !important; min-height:48px;}
.st-key-card_search .stTextInput input {background:#edf3f8; font-size:1.02rem; padding-left:18px;}
.stTextArea textarea {border-radius:8px;}
/* pills */
[data-testid="stButtonGroup"] button {border-radius:999px !important;}

/* ---------- text helpers ---------- */
.job-title {color:#0a66c2; font-weight:700; font-size:1.02rem; line-height:1.25;}
.job-title.big {font-size:1.45rem; color:#191919;}
.job-co {color:#191919; font-size:.92rem;}
.job-meta {color:#666; font-size:.85rem;}
.muted {color:#666; font-size:.86rem;}
.section-h {font-size:1.15rem; font-weight:700; color:#191919; margin:2px 0 6px 0;}
.kicker {font-size:.75rem; font-weight:700; color:#666; text-transform:uppercase; letter-spacing:.04em;}
.logo {border-radius:8px; color:#fff; font-weight:700; display:flex; align-items:center;
  justify-content:center; flex-shrink:0;}
.row {display:flex; gap:12px; align-items:flex-start;}

/* ---------- chips ---------- */
.chip {display:inline-block; padding:3px 10px; margin:2px 5px 2px 0; border-radius:999px; font-size:.78rem;
  font-weight:600; background:#eef3f8; color:#38434f; border:1px solid #dce6f1;}
.chip.blue {background:#e8f1fb; color:#0a66c2; border-color:#b9d4f1;}
.chip.green {background:#e6f4ea; color:#057642; border-color:#9bd3ae;}
.chip.red {background:#fdecea; color:#b3261e; border-color:#f3b3ae;}
.chip.amber {background:#fff4e5; color:#915907; border-color:#f5c26b;}
.chip.gray {background:#f3f2ef; color:#56687a; border-color:#e0dfdc;}

/* ---------- match meter ---------- */
.match {font-weight:800; font-size:1.6rem; line-height:1;}
.match.strong {color:#057642;} .match.moderate {color:#915907;} .match.weak {color:#b3261e;}
.meter {height:6px; background:#e9e5df; border-radius:6px; overflow:hidden; margin-top:6px;}
.meter > div {height:6px; border-radius:6px;}

/* ---------- coach ---------- */
.badge {display:inline-flex; align-items:center; gap:6px; padding:4px 11px; border-radius:999px;
  font-weight:700; font-size:.78rem;}
.badge.pass {background:#e6f4ea; color:#057642;}
.badge.retry {background:#fff4e5; color:#915907;}
.badge.fail {background:#fdecea; color:#b3261e;}
.badge.refused {background:#eef3f8; color:#38434f;}
.cite {display:inline-block; background:#0a66c2; color:#fff; border-radius:4px; padding:0 6px;
  font-size:.72rem; font-weight:700; margin:0 1px; vertical-align:1px;}
.quote-ok {color:#057642;} .quote-bad {color:#b3261e;}
.check-yes {color:#057642; font-weight:600;} .check-no {color:#8c8c8c;}

/* ---------- KPI tiles ---------- */
.kpi {background:#fff; border:1px solid #e0dfdc; border-radius:10px; padding:14px 16px;}
.kpi .v {font-size:1.7rem; font-weight:800; color:#191919; line-height:1.1;}
.kpi .l {font-size:.8rem; color:#666; margin-top:3px;}
.kpi .d {font-size:.8rem; font-weight:700; color:#057642;}
.stage {border-left:4px solid #0a66c2; padding:.35rem .8rem; margin:.35rem 0; background:#fff;
  border-radius:4px; font-size:.9rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    # "$" -> entity, otherwise Streamlit markdown renders "$28 ... $32" as a LaTeX formula.
    return html.escape(str(text)).replace("$", "&#36;")


def nomath(text):
    """Escape dollar signs for plain st.write / st.caption / st.markdown text."""
    return str(text).replace("$", "\\$")


def chip(text, kind=""):
    return f'<span class="chip {kind}">{esc(text)}</span>'


def chips_html(items):
    return " ".join(chip(t, k) for t, k in items)


def chips(items):
    st.markdown(chips_html(items), unsafe_allow_html=True)


def logo(company, size=48):
    words = [w for w in re.split(r"[\s&]+", company) if w]
    initials = "".join(w[0] for w in words[:2]).upper()
    color = LOGO_COLORS[sum(map(ord, company)) % len(LOGO_COLORS)]
    return (f'<div class="logo" style="width:{size}px;height:{size}px;background:{color};'
            f'font-size:{int(size * 0.38)}px">{esc(initials)}</div>')


def posting_chips(p):
    return [
        ((p["remote"].capitalize()), "gray"),
        ("Paid " + p["pay"] if p["paid"] else "Unpaid (credit)", "green" if p["paid"] else "amber"),
        ("✓ CPT/OPT supported" if p["visa_sponsorship"] else "No visa sponsorship",
         "blue" if p["visa_sponsorship"] else "gray"),
    ]


def job_header(p, big=False, extra=""):
    size = 64 if big else 46
    cls = "job-title big" if big else "job-title"
    return (f'<div class="row">{logo(p["company"], size)}<div>'
            f'<div class="{cls}">{esc(p["title"])}</div>'
            f'<div class="job-co">{esc(p["company"])} · <span class="job-meta">{esc(p["company_stage"])}</span></div>'
            f'<div class="job-meta">{esc(p["location"])} · {esc(p["category"])} {extra}</div>'
            f'</div></div>')


def kpi(value, label, delta=""):
    d = f'<div class="d">{esc(delta)}</div>' if delta else ""
    return f'<div class="kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>{d}</div>'


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(kpi(*item), unsafe_allow_html=True)


def match_meter(score, label):
    kind = label if label in ("strong", "moderate", "weak") else "weak"
    color = {"strong": "#057642", "moderate": "#e8a33d", "weak": "#cc1016"}[kind]
    word = {"strong": "Strong match", "moderate": "Possible match", "weak": "Weak match"}[kind]
    pct = max(0, min(100, int(score or 0)))
    return (f'<div class="match {kind}">{pct}%</div><div class="muted">{word}</div>'
            f'<div class="meter"><div style="width:{pct}%;background:{color}"></div></div>')


def badge(status):
    kind, label = {
        "pass": ("pass", "✓ Fact-checked against the guides"),
        "pass after retry": ("retry", "↻ Fixed after fact-check, then passed"),
        "refused": ("refused", "→ Outside the guides: routed to an advisor"),
    }.get(status, ("fail", "✕ Failed fact-check: routed to an advisor"))
    return f'<span class="badge {kind}">{label}</span>'


def render_citations(text):
    safe = esc(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)       # keep the model's bold markers
    safe = safe.replace("\n", "<br>")
    return re.sub(r"\[(G\d{2})\]", r'<span class="cite">\1</span>', safe)


def section(title, sub=""):
    s = f'<div class="section-h">{esc(title)}</div>'
    if sub:
        s += f'<div class="muted" style="margin-top:-4px;margin-bottom:8px">{esc(sub)}</div>'
    st.markdown(s, unsafe_allow_html=True)
