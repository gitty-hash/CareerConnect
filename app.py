"""CareerCompass - one retrieval backbone, three AI products (+ an employer copilot).

Run:  streamlit run app.py

Tabs map to the homework parts and course chapters:
  Jobs            Part A + B  (Ch3 few-shot filter extraction, Ch4 BM25/vector/hybrid + semantic map)
  For You         Part C      (Ch4 profile-as-query, Ch3 grounded honest explanations)
  Career Coach    Part D      (Ch3 persona + LLM-as-judge, Ch4 RAG with citations and refusal)
  For Employers   Part F      (innovation feature: Employer Reach Copilot)
  Insights        Part E      (Ch4 advanced RAG + lab-style RAGAS metrics)
  How it works    architecture, data design, chapter map
"""

import hashlib
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from careercompass import advanced, coach, data, employer, evaluation, llm, recommend, search, ui
from careercompass.config import LLM_MODEL, REACH_MIN_COSINE, SCOPE_GATE_MIN_COSINE, TOP_K_POSTINGS

st.set_page_config(page_title="CareerCompass", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()


@st.cache_resource(show_spinner="Loading CareerCompass (embeddings, FAISS, BM25)...")
def get_backbone():
    from careercompass.backbone import Backbone
    return Backbone()


bb = get_backbone()
SCEN = data.load_scenarios()
VOCAB = sorted({s for p in bb.postings for s in p["skills"]} | {s for c in bb.candidates for s in c["skills"]})
NOT_CACHED_MSG = ("This exact request has no recorded AI output, and the app is in **demo mode**. "
                  "Pick one of the suggested examples, or set `OPENAI_API_KEY` to run it live.")
ENGINES = {"Smart": "hybrid", "Meaning": "vector", "Keywords": "bm25"}
ENGINE_HELP = ("Smart = hybrid (keywords + meaning, 50/50). Meaning = embeddings only. "
               "Keywords = classic BM25.")


def qkey(text):
    """Stable short key for widgets that should reset when the text changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def has_skill(skill, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", text.lower()) is not None


# ----------------------------------------------------------------------------- settings (sidebar)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if llm.api_key_available():
        use_cache = st.toggle("Demo mode (replay recorded AI outputs)", value=False)
        llm.set_force_cached(use_cache)
    else:
        llm.set_force_cached(True)
        st.toggle("Demo mode (replay recorded AI outputs)", value=True, disabled=True,
                  help="No OPENAI_API_KEY found, so the app replays recorded outputs.")
    n_cached, _ = llm.cache_stats()
    st.caption(f"{n_cached} recorded {LLM_MODEL} outputs. Retrieval always runs live and locally.")
    st.divider()
    st.markdown("**Knowledge base**")
    st.markdown(f"- {len(bb.postings)} internship postings\n"
                f"- {len(bb.guides)} career guides ({len(bb.chunks)} chunks)\n"
                f"- {len(bb.candidates)} synthetic candidate profiles")

mode = ('<span class="cc-mode live">● Live AI</span>' if llm.is_live()
        else '<span class="cc-mode demo">● Demo mode · recorded AI outputs</span>')
st.markdown(f"""<div class="cc-topbar"><div class="cc-logo">cc</div>
<div class="cc-brand">CareerCompass<small>Internships &amp; early careers for university students</small></div>
<div class="cc-spacer"></div>{mode}</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🔎  Jobs", "⭐  For You", "💬  Career Coach", "🏢  For Employers", "📊  Insights", "ℹ️  How it works"])


def semantic_map(query_vecs=(), query_labels=(), highlight_ids=(), height=440):
    """Part A: PCA projection of the posting embeddings, colored by category, query marked."""
    df = pd.DataFrame({"x": bb.posting_xy[:, 0], "y": bb.posting_xy[:, 1],
                       "category": [p["category"] for p in bb.postings],
                       "title": [f"{p['title']} - {p['company']}" for p in bb.postings]})
    fig = px.scatter(df, x="x", y="y", color="category", hover_name="title",
                     hover_data={"x": False, "y": False, "category": True},
                     color_discrete_map=ui.CATEGORY_COLORS, height=height)
    fig.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="white")))
    if highlight_ids:
        pts = [i for i, p in enumerate(bb.postings) if p["posting_id"] in highlight_ids]
        fig.add_trace(go.Scatter(x=bb.posting_xy[pts, 0], y=bb.posting_xy[pts, 1], mode="markers",
                                 marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(width=2, color="#191919")),
                                 name="your results", hoverinfo="skip"))
    for vec, label in zip(query_vecs, query_labels):
        xy = bb.project(vec)
        fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers+text", text=[label],
                                 textposition="top center", name=label, textfont=dict(size=13, color="#191919"),
                                 marker=dict(symbol="star", size=24, color="#f5c518", line=dict(width=1.5, color="#191919"))))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), legend=dict(font=dict(size=11), title=None),
                      xaxis=dict(showticklabels=False, title="", showgrid=False, zeroline=False),
                      yaxis=dict(showticklabels=False, title="", showgrid=False, zeroline=False),
                      plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each dot is a posting. Nearby dots mean similar jobs. The map is a PCA projection of the "
               f"384-dimension embeddings (2 components keep {bb.pca_explained:.0%} of the variance). PCA, not "
               "t-SNE, because a fixed linear map can place a *new* query into the same space. ⭐ = you.")


# ============================================================================= JOBS
def _pick_query():
    if st.session_state.get("q_pill"):
        st.session_state.q = st.session_state.q_pill
        st.session_state.q_pill = None


def _select_job(pid):
    st.session_state.sel_job = pid


def _toggle_save(pid):
    saved = st.session_state.setdefault("saved", set())
    saved.symmetric_difference_update({pid})


with tabs[0]:
    if "q" not in st.session_state:
        st.session_state.q = SCEN["search_presets"][0]
    with st.container(key="card_search"):
        st.markdown('<div class="section-h">Find your next internship</div>'
                    '<div class="muted" style="margin-bottom:6px">Describe it in your own words, including '
                    'pay, remote work, or visa needs. CareerCompass searches by meaning, not just keywords.</div>',
                    unsafe_allow_html=True)
        query = st.text_input("Search", key="q", label_visibility="collapsed",
                              placeholder="e.g. paid remote UX internship that supports CPT/OPT")
        st.pills("Try a search", SCEN["search_presets"], key="q_pill", on_change=_pick_query,
                 format_func=ui.nomath)

    if query.strip():
        filters, source = search.extract_filters(query)
        k = qkey(query)
        c_chips, c_edit, c_rank = st.columns([0.62, 0.19, 0.19], vertical_alignment="center")
        with c_edit.popover("✏️ Edit filters", width="stretch"):
            st.caption("Correct anything the AI got wrong.")
            cat_opts = ["Any"] + search.CATEGORIES
            cat = st.selectbox("Job category", cat_opts, key=f"cat_{k}",
                               index=cat_opts.index(filters["category"]) if filters["category"] else 0)
            mode_opts = ["Any", "remote", "hybrid", "onsite"]
            wmode = st.selectbox("Work mode", mode_opts, key=f"mode_{k}",
                                 index=mode_opts.index(filters["remote"]) if filters["remote"] else 0)
            pay_opts = ["Any", "Paid only", "Unpaid only"]
            pay = st.selectbox("Pay", pay_opts, key=f"pay_{k}",
                               index={True: 1, False: 2}.get(filters["paid"], 0))
            visa_opts = ["Any", "Supports CPT/OPT", "No sponsorship"]
            visa = st.selectbox("Visa / work authorization", visa_opts, key=f"visa_{k}",
                                index={True: 1, False: 2}.get(filters["visa_sponsorship"], 0))
        with c_rank.popover("↕ Ranking", width="stretch"):
            engine_label = st.segmented_control("Ranking engine", list(ENGINES), default="Smart",
                                                key="engine", help=ENGINE_HELP) or "Smart"
            method = ENGINES[engine_label]
            alpha = st.slider("Keyword weight (Smart only)", 0.0, 1.0, 0.5, 0.1, disabled=method != "hybrid")
            use_rerank = st.toggle("AI re-rank the top 10 (cross-encoder)")
        applied = {"semantic_query": filters["semantic_query"],
                   "category": None if cat == "Any" else cat,
                   "remote": None if wmode == "Any" else wmode,
                   "paid": {"Paid only": True, "Unpaid only": False}.get(pay),
                   "visa_sponsorship": {"Supports CPT/OPT": True, "No sponsorship": False}.get(visa)}
        active = []
        if applied["category"]:
            active.append((applied["category"], "blue"))
        if applied["remote"]:
            active.append((applied["remote"].capitalize(), "blue"))
        if applied["paid"] is not None:
            active.append(("Paid" if applied["paid"] else "Unpaid", "blue"))
        if applied["visa_sponsorship"] is not None:
            active.append(("Supports CPT/OPT" if applied["visa_sponsorship"] else "No sponsorship", "blue"))
        src_note = {"cache": "AI (recorded)", "llm": "AI", "rule-based fallback": "keyword fallback, please check"}[source]
        c_chips.markdown(
            '<span class="muted">AI understood your search as </span>'
            + (ui.chips_html(active) if active else ui.chip("no filters, matching by meaning only", "gray"))
            + f' <span class="muted">· searching for “{ui.esc(applied["semantic_query"])}” · {src_note}</span>',
            unsafe_allow_html=True)
        with st.expander("See the raw JSON the AI extracted"):
            st.json({key: filters[key] for key in search.FILTER_KEYS + ["semantic_query"]})

        hits, n_pass = search.filtered_search(bb, applied, method=method, k=TOP_K_POSTINGS,
                                              alpha=alpha, rerank=use_rerank)
        r_tab, m_tab, c_tab = st.tabs(["Results", "Job map", "Compare search engines"])

        with r_tab:
            if not hits:
                st.info("No internship matches every filter. Use **Edit filters** to broaden your search.")
            else:
                ids = [bb.postings[i]["posting_id"] for i, _ in hits]
                if st.session_state.get("sel_job") not in ids:
                    st.session_state.sel_job = ids[0]
                saved = st.session_state.setdefault("saved", set())
                left, right = st.columns([0.42, 0.58])
                with left:
                    st.markdown(f'<div class="muted">{len(hits)} top results · {n_pass} of {len(bb.postings)} '
                                f'postings match your filters · ★ {len(saved)} saved</div>', unsafe_allow_html=True)
                    score_name = "AI relevance" if use_rerank else {"hybrid": "Match", "vector": "Similarity",
                                                                     "bm25": "Keyword score"}[method]
                    for i, s in hits:
                        p = bb.postings[i]
                        sel = p["posting_id"] == st.session_state.sel_job
                        with st.container(key=f"card{'sel' if sel else ''}_job_{p['posting_id']}_{k}"):
                            star = " ★" if p["posting_id"] in saved else ""
                            st.markdown(ui.job_header(p, extra=star) + '<div style="margin-top:6px">'
                                        + ui.chips_html(ui.posting_chips(p)) + "</div>", unsafe_allow_html=True)
                            b1, b2 = st.columns([0.55, 0.45], vertical_alignment="center")
                            b1.markdown(f'<span class="muted">{score_name} {s:.2f}</span>', unsafe_allow_html=True)
                            b2.button("View details", key=f"view_{p['posting_id']}_{k}", width="stretch",
                                      type="primary" if sel else "secondary",
                                      on_click=_select_job, args=(p["posting_id"],))
                with right:
                    p = bb.by_pid[st.session_state.sel_job]
                    pi = bb.postings.index(p)
                    with st.container(key="card_detail"):
                        st.markdown(ui.job_header(p, big=True), unsafe_allow_html=True)
                        st.markdown(ui.chips_html(ui.posting_chips(p)), unsafe_allow_html=True)
                        a1, a2 = st.columns(2)
                        is_saved = p["posting_id"] in saved
                        a1.button("★ Saved" if is_saved else "☆ Save job", key=f"save_{p['posting_id']}",
                                  type="primary", width="stretch", on_click=_toggle_save, args=(p["posting_id"],))
                        a2.markdown(f'<div class="muted" style="text-align:right">Posting ID {p["posting_id"]}</div>',
                                    unsafe_allow_html=True)
                        st.markdown('<div class="kicker" style="margin-top:10px">Why this matched</div>',
                                    unsafe_allow_html=True)
                        q_sem = applied["semantic_query"]
                        rank_of = {}
                        for m in ("bm25", "vector", "hybrid"):
                            order = [idx for idx, _ in bb.posting_search(q_sem, m)]
                            rank_of[m] = order.index(pi) + 1 if pi in order else None
                        st.markdown(ui.chips_html([
                            (f"Meaning rank #{rank_of['vector']}", "gray"),
                            (f"Keyword rank #{rank_of['bm25']}" if rank_of["bm25"] else "No keyword overlap", "gray"),
                            (f"Smart rank #{rank_of['hybrid']}", "gray")]), unsafe_allow_html=True)
                        st.markdown('<div class="kicker" style="margin-top:12px">Skills</div>', unsafe_allow_html=True)
                        st.markdown(ui.chips_html([(s_, "") for s_ in p["skills"]]), unsafe_allow_html=True)
                        st.markdown('<div class="kicker" style="margin-top:12px">About the job</div>',
                                    unsafe_allow_html=True)
                        st.write(ui.nomath(p["description"]))
                    with st.container(key="card_similar"):
                        st.markdown('<div class="section-h">Similar internships</div>', unsafe_allow_html=True)
                        sims = bb.posting_emb @ bb.posting_emb[pi]
                        for j in [j for j in sims.argsort()[::-1] if j != pi][:3]:
                            q_ = bb.postings[j]
                            cc1, cc2 = st.columns([0.72, 0.28], vertical_alignment="center")
                            cc1.markdown(ui.job_header(q_), unsafe_allow_html=True)
                            cc2.button("View", key=f"sim_{q_['posting_id']}_{k}", width="stretch",
                                       on_click=_select_job, args=(q_["posting_id"],))

        with m_tab:
            with st.container(key="card_map"):
                ui.section("Where your search lands among all internships",
                           "Part A semantic-space view: postings colored by category, your query as a star, "
                           "your results circled.")
                semantic_map([bb.embed([applied["semantic_query"]])], ["your search"],
                             highlight_ids=[bb.postings[i]["posting_id"] for i, _ in hits])

        with c_tab:
            with st.container(key="card_compare"):
                ui.section("Keyword vs. meaning vs. smart search, side by side",
                           "Ch4 lab Part 4: meaning wins on paraphrases, keywords win on rare exact terms, and the "
                           "hybrid trades a little peak accuracy for robustness. ✅ = the posting the benchmark expects.")
                bench = SCEN["search_benchmarks"]
                bq = st.selectbox("Benchmark query", range(len(bench)),
                                  format_func=lambda i: f"{bench[i]['query']}   ({bench[i]['type']})")
                b = bench[bq]
                st.markdown(f'<div class="muted">{ui.esc(b["why"])}</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                for col, m, name in zip(cols, ["bm25", "vector", "hybrid"],
                                        ["Keywords (BM25)", "Meaning (vector)", "Smart (hybrid 50/50)"]):
                    all_hits = bb.posting_search(b["query"], m)
                    order = [bb.postings[i]["posting_id"] for i, _ in all_hits]
                    with col:
                        st.markdown(f"**{name}**")
                        rows = [{"#": r, "": "✅" if bb.postings[i]["posting_id"] in b["gold"] else "",
                                 "posting": f"{bb.postings[i]['title']} ({bb.postings[i]['posting_id']})",
                                 "score": round(s, 3)} for r, (i, s) in enumerate(all_hits[:5], start=1)]
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                        st.caption("Expected: " + ", ".join(
                            f"{g} at #{order.index(g) + 1}" if g in order else f"{g} not returned" for g in b["gold"]))
                rec = pd.DataFrame(evaluation.search_recall_at_k(bb, bench))
                avg = rec[["bm25", "vector", "hybrid"]].mean()
                ui.kpi_row([(f"{avg['bm25']:.2f}", "Keywords: avg recall@5"),
                            (f"{avg['vector']:.2f}", "Meaning: avg recall@5"),
                            (f"{avg['hybrid']:.2f}", "Smart: avg recall@5")])
                with st.expander("Recall@5 for every benchmark query"):
                    st.dataframe(rec.rename(columns={"bm25": "Keywords", "vector": "Meaning", "hybrid": "Smart"}),
                                 hide_index=True, width="stretch")

# ============================================================================= FOR YOU
with tabs[1]:
    profiles = SCEN["profiles"]
    left, right = st.columns([0.34, 0.66])
    with left:
        with st.container(key="card_profile"):
            st.markdown('<div class="row"><div class="logo" style="width:56px;height:56px;border-radius:50%;'
                        'background:#0a66c2;font-size:24px">🎓</div><div><div class="section-h">Your profile</div>'
                        '<div class="muted">Your profile is the search query: we embed it and find the '
                        'closest internships.</div></div></div>', unsafe_allow_html=True)
            pidx = st.selectbox("Example student", range(len(profiles)), format_func=lambda i: profiles[i]["label"])
            profile_text = st.text_area("About you (paste a resume summary)", profiles[pidx]["text"], height=190,
                                        key=f"profile_{pidx}")
            st.markdown('<div class="kicker">Must-haves</div>', unsafe_allow_html=True)
            txt = profile_text.lower()
            mh = {"needs_sponsorship": st.toggle("I need CPT/OPT support", value="f-1" in txt or "cpt" in txt,
                                                 key=f"mh1_{pidx}"),
                  "paid_only": st.toggle("Paid roles only", value="paid" in txt, key=f"mh2_{pidx}"),
                  "remote_only": st.toggle("Remote only", value="ideally remote" in txt, key=f"mh3_{pidx}")}
            hide_conflicts = st.toggle("Hide jobs that break a must-have", key=f"hide_{pidx}")
            with st.expander("Advanced: explanation prompt"):
                version = st.radio("Prompt version", ["v2", "v1"], horizontal=True,
                                   format_func={"v2": "v2 grounded + honest", "v1": "v1 naive first draft"}.get)

    recs = recommend.recommend(bb, profile_text, k=len(bb.postings))
    if hide_conflicts:
        recs = [(i, s) for i, s in recs if not recommend.must_have_conflicts(bb.postings[i], mh)]
    recs = recs[:5]
    scores = []
    with right:
        ui.section("Top job picks for you",
                   "Ranked by how close each posting is to your profile. Every explanation must quote you and "
                   "the job, and the quotes are checked word for word.")
        for r, (i, sim) in enumerate(recs, start=1):
            p = bb.postings[i]
            conflicts = recommend.must_have_conflicts(p, mh)
            with st.container(key=f"card_rec_{p['posting_id']}_{pidx}"):
                try:
                    ex = recommend.explain_fit(profile_text, p, version)
                except llm.NotCached:
                    ex = None
                h1, h2 = st.columns([0.74, 0.26])
                h1.markdown(ui.job_header(p) + '<div style="margin-top:6px">'
                            + ui.chips_html(ui.posting_chips(p) + [(f"⚠ {c}", "red") for c in conflicts])
                            + "</div>", unsafe_allow_html=True)
                if ex is None:
                    h2.markdown(f'<div class="muted">Profile similarity {sim:.2f}</div>', unsafe_allow_html=True)
                    st.info(NOT_CACHED_MSG)
                    continue
                scores.append(ex.get("fit_score", 0))
                h2.markdown(ui.match_meter(ex.get("fit_score"), ex.get("fit_label")), unsafe_allow_html=True)
                checks = " ".join(
                    f'<span class="{"check-yes" if has_skill(s_, profile_text) else "check-no"}">'
                    f'{"✓" if has_skill(s_, profile_text) else "○"} {ui.esc(s_)}</span>&nbsp;&nbsp;'
                    for s_ in p["skills"])
                n_have = sum(has_skill(s_, profile_text) for s_ in p["skills"])
                st.markdown(f'<div class="kicker" style="margin-top:8px">How you match: {n_have} of '
                            f'{len(p["skills"])} skills on your profile</div><div>{checks}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div style="margin-top:8px">{ui.esc(ex.get("explanation", ""))}</div>',
                            unsafe_allow_html=True)
                q1, q2 = st.columns(2)
                for col, title, qs in [(q1, "From your profile", ex["profile_quote_check"]),
                                       (q2, "From the job posting", ex["posting_quote_check"])]:
                    lines = "".join(
                        f'<div class="{"quote-ok" if ok else "quote-bad"}">{"✓" if ok else "✕ not found:"} '
                        f'“{ui.esc(q)}”</div>' for q, ok in qs) or '<div class="quote-bad">no quote given</div>'
                    col.markdown(f'<div class="kicker">{title}</div>{lines}', unsafe_allow_html=True)
                if ex.get("gaps"):
                    st.markdown('<div class="kicker" style="margin-top:8px">What\'s missing</div>'
                                + "".join(f"<div class='muted'>• {ui.esc(g)}</div>" for g in ex["gaps"]),
                                unsafe_allow_html=True)
                with st.expander(f"See how the AI scored this ({ex['quotes_verified']}/{ex['quotes_total']} "
                                 f"quotes verified · profile similarity {sim:.2f})"):
                    st.write(ui.nomath(ex.get("reasoning", "(no reasoning returned)")))

    st.write("")
    t_map, t_iter, t_honest = st.tabs(["Your profile on the job map", "Prompt iteration: v1 vs. v2",
                                       "Honesty test"])
    with t_map:
        with st.container(key="card_fy_map"):
            semantic_map([bb.embed([profile_text])], ["you"],
                         highlight_ids=[bb.postings[i]["posting_id"] for i, _ in recs], height=400)
    with t_iter:
        with st.container(key="card_iter"):
            ui.section("Same postings, two prompts",
                       "v1 asks 'why is this a great fit?' with no quote or honesty rule. "
                       "v2 must reason step by step, quote both texts, and call weak matches weak.")
            rows = []
            for i, _ in recs:
                row = {"posting": f"{bb.postings[i]['title']} ({bb.postings[i]['posting_id']})"}
                for v in ("v1", "v2"):
                    try:
                        e = recommend.explain_fit(profile_text, bb.postings[i], v)
                        row[f"{v} fit"] = e.get("fit_score")
                        row[f"{v} quotes verified"] = f"{e['quotes_verified']}/{e['quotes_total']}"
                    except llm.NotCached:
                        row[f"{v} fit"] = None
                rows.append(row)
            iter_df = pd.DataFrame(rows)
            st.dataframe(iter_df, hide_index=True, width="stretch")
            if {"v1 fit", "v2 fit"} <= set(iter_df) and iter_df[["v1 fit", "v2 fit"]].notna().all().all():
                ui.kpi_row([(f"{iter_df['v1 fit'].mean():.0f}", "v1 average fit"),
                            (f"{iter_df['v2 fit'].mean():.0f}", "v2 average fit")])
    with t_honest:
        with st.container(key="card_honest"):
            ui.section("Honesty test: a culinary student",
                       "There are no culinary internships in the corpus, so retrieval still returns *something*. "
                       "A trustworthy explainer must say these are weak matches.")
            mism = profiles[3]["text"]
            rows = []
            for i, sim in recommend.recommend(bb, mism, k=5):
                row = {"posting": f"{bb.postings[i]['title']} ({bb.postings[i]['posting_id']})",
                       "similarity": round(sim, 3)}
                for v in ("v1", "v2"):
                    try:
                        e = recommend.explain_fit(mism, bb.postings[i], v)
                        row[f"{v} fit"], row[f"{v} label"] = e.get("fit_score"), e.get("fit_label")
                    except llm.NotCached:
                        row[f"{v} fit"] = None
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ============================================================================= COACH
def _ask_from_pill():
    if st.session_state.get("coach_pill"):
        st.session_state.pending_q = st.session_state.coach_pill
        st.session_state.coach_pill = None


def _ask_from_more():
    if st.session_state.get("coach_pill_more"):
        st.session_state.pending_q = st.session_state.coach_pill_more
        st.session_state.coach_pill_more = None


def _ask_from_box():
    q = st.session_state.get("coach_input", "").strip()
    if q:
        st.session_state.pending_q = q
    st.session_state.coach_input = ""


def _clear_chat():
    st.session_state.chat = []


with tabs[2]:
    st.session_state.setdefault("chat", [])
    top_l, top_r = st.columns([0.8, 0.2], vertical_alignment="center")
    top_l.markdown('<div class="row"><div class="logo" style="width:52px;height:52px;border-radius:50%;'
                   'background:#057642;font-size:24px">🧭</div><div><div class="section-h">Juno · Career Coach</div>'
                   '<div class="muted">Answers only from CareerCompass career guides, cites every source, and hands '
                   'you to a human advisor when the guides don\'t cover your question.</div></div></div>',
                   unsafe_allow_html=True)
    with top_r.popover("⚙ Coach settings", width="stretch"):
        pipeline = st.selectbox("Retrieval pipeline", list(advanced.PIPELINES), index=1,
                                format_func=advanced.PIPELINES.get)
        excl = st.toggle("Ignore archived guides", help="Pre-retrieval fix: drops guides marked archived.")
        stress = st.toggle("Stress test the fact-checker",
                           help="Injects a hallucinated first draft to show the fact-checker and fallback.")
        st.button("Clear conversation", on_click=_clear_chat, width="stretch")

    if st.session_state.get("pending_q"):
        q = st.session_state.pop("pending_q")
        try:
            res = coach.ask(bb, q, pipeline=pipeline, exclude_archived=excl, stress_test=stress)
        except llm.NotCached:
            res = None
        st.session_state.chat.append({"q": q, "res": res, "pipeline": pipeline})

    chat_col, src_col = st.columns([0.6, 0.4])
    with chat_col:
        with st.container(key="card_chat"):
            with st.chat_message("assistant", avatar="🧭"):
                st.markdown("Hi, I'm **Juno**! Ask me about resumes, interviews, networking, offers, "
                            "internship etiquette, or work authorization. I'll show you exactly which guide "
                            "each answer comes from.")
            for turn in st.session_state.chat:
                with st.chat_message("user", avatar="🧑‍🎓"):
                    st.write(ui.nomath(turn["q"]))
                with st.chat_message("assistant", avatar="🧭"):
                    res = turn["res"]
                    if res is None:
                        st.info(NOT_CACHED_MSG)
                        continue
                    st.markdown(ui.badge(res["badge"]), unsafe_allow_html=True)
                    st.markdown(ui.render_citations(res["answer"]), unsafe_allow_html=True)
                    if res["status"] == "refused":
                        st.caption(f"Stopped by the {res['refusal_layer']} (closest guide similarity "
                                   f"{res['top_similarity']:.2f}, gate {SCOPE_GATE_MIN_COSINE}).")
                    with st.expander("How Juno fact-checked this answer"):
                        if not res["attempts"]:
                            st.write("No draft was written: the scope gate refused before calling the AI.")
                        for n, att in enumerate(res["attempts"], start=1):
                            st.markdown(f"**Draft {n}**")
                            st.caption(ui.nomath(att["draft"][:700]))
                            v = att["verdict"]
                            if v is None:
                                st.write("Juno answered OUT_OF_SCOPE, so the question was routed to an advisor.")
                                continue
                            ok = v.get("verdict") == "pass"
                            cc = v["citation_code_check"]
                            ui.chips([(f"Verdict: {v.get('verdict')}", "green" if ok else "red"),
                                      (f"Faithfulness: {v.get('faithfulness')}", "gray"),
                                      (f"Citations: {v.get('citations')}", "gray"),
                                      (f"Safety: {v.get('safety')}", "gray"),
                                      (f"Code citation check: {'ok' if cc['ok'] else 'failed'}",
                                       "green" if cc["ok"] else "red")])
                            for c in v.get("unsupported_claims", []):
                                st.markdown(f"- Unsupported: {ui.nomath(c)}")
                            for c in v.get("reasons", []):
                                st.markdown(f"- Reason: {ui.nomath(c)}")
                    with st.expander("What retrieval did (before/after ranking)"):
                        gold = next((x["gold"] for x in SCEN["coach_eval"] if x["q"] == turn["q"]), None)
                        st.caption(advanced.PIPELINES[turn["pipeline"]])
                        st.dataframe(pd.DataFrame(advanced.rank_table(bb, res["hits"], res["trace"], gold)),
                                     hide_index=True, width="stretch")
                        if res["trace"].get("hypothetical_answer"):
                            st.markdown("**HyDE hypothetical answer** (used only for searching):")
                            st.caption(ui.nomath(res["trace"]["hypothetical_answer"]))
                        if res["trace"].get("rewrites"):
                            st.markdown("**RAG-fusion rewrites:** " + ui.nomath(" | ".join(res["trace"]["rewrites"])))
            in_scope = [x["q"] for x in SCEN["coach_eval"]]
            featured = [in_scope[i] for i in (6, 7, 2, 11)] + SCEN["coach_out_of_scope"][:2]
            more = [q_ for q_ in in_scope + SCEN["coach_out_of_scope"] if q_ not in featured]
            sq1, sq2 = st.columns([0.8, 0.2], vertical_alignment="bottom")
            sq1.pills("Try asking", featured, key="coach_pill", on_change=_ask_from_pill,
                      format_func=ui.nomath)
            with sq2.popover("More examples", width="stretch"):
                st.caption("The last questions are outside the guides, so Juno should hand them to an advisor.")
                st.pills("More questions", more, key="coach_pill_more", on_change=_ask_from_more,
                         format_func=ui.nomath, label_visibility="collapsed")
            with st.form("coach_form", border=False):
                f1, f2 = st.columns([0.8, 0.2], vertical_alignment="bottom")
                f1.text_input("Ask Juno", key="coach_input", placeholder="Ask a career question...",
                              label_visibility="collapsed")
                f2.form_submit_button("Ask", type="primary", width="stretch", on_click=_ask_from_box)
    with src_col:
        with st.container(key="card_sources"):
            ui.section("Sources", "Guide excerpts Juno retrieved for the latest answer")
            last = next((t["res"] for t in reversed(st.session_state.chat) if t["res"]), None)
            if not last:
                st.caption("Ask a question to see the guide excerpts behind the answer.")
            else:
                for idx, score in last["hits"]:
                    c = bb.chunks[idx]
                    st.markdown(f'<span class="cite">{c["doc_id"]}</span> <b>{ui.esc(c["title"])}</b><br>'
                                + ui.chips_html([(c["status"], "green" if c["status"] == "current" else "red"),
                                                 (f"updated {c['last_updated']}", "gray"),
                                                 (f"score {score:.2f}", "gray")]), unsafe_allow_html=True)
                    if c["disclaimer"]:
                        st.warning(c["disclaimer"])
                    st.caption(ui.nomath(c["text"]))
                    st.divider()

# ============================================================================= EMPLOYERS
with tabs[3]:
    with st.container(key="card_emp_intro"):
        st.markdown('<div class="kicker">CareerCompass for Employers · Reach Copilot</div>'
                    '<div class="section-h" style="font-size:1.35rem">See which students your posting excludes, '
                    'before you publish it</div><div class="muted">Paste a draft internship posting. The copilot '
                    f'uses it as a search query over {len(bb.candidates)} student profiles and counts how many '
                    'qualified students each requirement excludes. It flags exclusionary wording (each flag is '
                    'reviewed by an AI judge) and re-measures your reach after a suggested rewrite. You only ever '
                    'see aggregate counts, never individual students.</div>', unsafe_allow_html=True)

    drafts = SCEN["employer_drafts"]
    d1, d2 = st.columns([0.55, 0.45])
    with d1:
        with st.container(key="card_emp_draft"):
            ui.section("1 · Your draft posting")
            didx = st.selectbox("Example draft", range(len(drafts)), format_func=lambda i: drafts[i]["label"])
            draft = st.text_area("Posting text", drafts[didx]["text"], height=210, key=f"draft_{didx}")
    cons, csrc = employer.extract_constraints(draft, VOCAB)
    before = employer.reach(bb, draft, cons, REACH_MIN_COSINE)
    with d2:
        with st.container(key="card_emp_cons"):
            ui.section("What the AI read as requirements", "Few-shot extraction (Ch3)" +
                       (" · keyword fallback, please check" if csrc == "rule-based fallback" else ""))
            labels = {"category": "Category", "required_skills": "Required skills",
                      "min_experience_years": "Min. experience (years)", "work_authorization": "Work authorization",
                      "work_mode": "Work mode", "paid": "Paid", "graduation_years": "Graduation years"}
            for key_, lab in labels.items():
                val = cons.get(key_)
                shown = "—" if val in (None, []) else (", ".join(map(str, val)) if isinstance(val, list) else str(val))
                st.markdown(f'<div class="row" style="justify-content:space-between"><span class="muted">{lab}'
                            f'</span><b>{ui.esc(shown)}</b></div>', unsafe_allow_html=True)
            if cons.get("authorization_legally_required"):
                st.info("Your draft states a legal basis for its work-authorization restriction. "
                        "The copilot keeps it and never suggests removing it. Verify with counsel.")

    kpi_slot = st.container()

    s2a, s2b = st.columns([0.55, 0.45])
    with s2a:
        with st.container(key="card_emp_funnel"):
            ui.section("2 · Your reach funnel", "Computed in code from student profile fields, not guessed by the AI")
            fig = go.Figure(go.Funnel(y=[f"{r['step'].capitalize()}: {r['rule']}" for r in before["funnel"]],
                                      x=[r["remaining"] for r in before["funnel"]], textinfo="value",
                                      marker=dict(color="#0a66c2")))
            fig.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
    with s2b:
        with st.container(key="card_emp_costs"):
            ui.section("Cost of each requirement",
                       f"Of the {before['relevant']} relevant students, how many each requirement rules out on its own")
            costs = sorted(before["costs"], key=lambda c: -c["excluded_on_its_own"])
            if not costs:
                st.caption("No hard requirements were found.")
            for c in costs:
                n, back = c["excluded_on_its_own"], c["regained_if_dropped"]
                kind = "red" if n >= 5 else ("amber" if n > 0 else "gray")
                st.markdown(f'<div class="row" style="justify-content:space-between;margin:4px 0">'
                            f'<span>{ui.esc(c["rule"])}</span><span>{ui.chip(f"excludes {n}", kind)}'
                            f'{ui.chip(f"+{back} if dropped alone", "gray")}</span></div>',
                            unsafe_allow_html=True)
            if costs and costs[0]["excluded_on_its_own"] > 0:
                blockers = [c["rule"] for c in costs if c["excluded_on_its_own"] == before["relevant"]]
                if len(blockers) > 1:
                    st.caption(f"{len(blockers)} requirements each exclude every relevant student "
                               f"({'; '.join(blockers)}), so dropping any one alone regains no one: all of "
                               "them have to change.")
                else:
                    st.caption(f"Most expensive requirement: **{costs[0]['rule']}**.")

    with st.container(key="card_emp_flags"):
        ui.section("3 · Suggested fixes", "Flagged by the AI or by a code check, then scored by an AI judge. "
                   "You decide which to apply.")
        try:
            flag_list = employer.all_flags(draft, cons)
            reviews = {r.get("index"): r for r in employer.judge_flags(draft, flag_list)}
        except llm.NotCached:
            flag_list, reviews = None, {}
            st.info(NOT_CACHED_MSG)
        kept = []
        if flag_list is not None and not flag_list:
            st.success("No exclusionary or inflated language found. Nice posting!")
        for i, f in enumerate(flag_list or []):
            rv = reviews.get(i, {})
            in_draft = employer.quote_in_draft(f.get("quote", ""), draft)
            fa, fb = st.columns([0.74, 0.26])
            fa.markdown(f'<b>“{ui.esc(f.get("quote", ""))}”</b> &nbsp;{ui.chip(f.get("issue", ""), "amber")}'
                        f'<div class="muted">{ui.esc(f.get("why", ""))}</div>'
                        f'<div>Suggestion: <i>{ui.esc(f.get("suggestion", ""))}</i></div>', unsafe_allow_html=True)
            ok = fb.toggle("Apply fix", value=bool(rv.get("keep", True)) and in_draft, key=f"flag{didx}_{i}")
            fb.markdown(ui.chips_html([(f"judge {rv.get('justification_score', '?')}/5",
                                        "green" if rv.get("keep") else "red"),
                                       (f.get("source", "LLM"), "gray")]), unsafe_allow_html=True)
            if ok:
                kept.append(f)
            st.divider()

    after, new_text = None, None
    if flag_list:
        try:
            new_text = employer.rewrite(draft, kept)
            new_cons, _ = employer.extract_constraints(new_text, VOCAB)
            # Same relevant pool as the original draft, so before/after isolates the requirements.
            after = employer.reach(bb, draft, new_cons, REACH_MIN_COSINE)
        except llm.NotCached:
            st.info(NOT_CACHED_MSG + " (A different set of applied fixes is a new request.)")
    if after:
        with st.container(key="card_emp_after"):
            ui.section("4 · Improved posting", "Same job duties and the same relevant students; only the "
                       "requirements changed")
            r1, r2 = st.columns([0.58, 0.42])
            r1.text_area("Rewritten posting (copy and edit before publishing)", new_text, height=260)
            with r2:
                st.dataframe(pd.DataFrame(after["funnel"]).rename(columns={"remaining": "students"}),
                             hide_index=True, width="stretch")
                if after["programs"]:
                    st.caption("Programs in your new reach (aggregate): " + ", ".join(
                        f"{k_} {v}" for k_, v in sorted(after["programs"].items(), key=lambda x: -x[1])))

    with kpi_slot:
        items = [(before["relevant"], "Students relevant to this role"),
                 (before["qualified"], "Qualified reach with your draft")]
        if after:
            delta = after["qualified"] - before["qualified"]
            items.append((after["qualified"], "Qualified reach after fixes", f"{delta:+d} students"))
        items.append((len(flag_list or []), "Issues flagged"))
        ui.kpi_row(items)
        st.write("")

    with st.expander("Risks and safeguards"):
        st.markdown("- **Risk:** optimizing for reach could pressure an employer to drop a requirement that is "
                    "genuinely needed, or the AI could misjudge a legal restriction (export control, clearances).\n"
                    "- **Safeguards:** the copilot never edits a live posting, and you approve each fix. Every flag "
                    "shows the judge's score. Restrictions with a stated legal basis are kept. The funnel is "
                    "computed in code, not by the AI. Only aggregate counts are shown, never individual students. "
                    "Candidate profiles contain no name, age, gender, or race.")

# ============================================================================= INSIGHTS
with tabs[4]:
    results = evaluation.load_results()
    ui.section("Insights: did the AI upgrades actually help?",
               "Part E: Chapter 4 advanced RAG techniques measured with lab-style RAGAS metrics")
    if llm.is_live() and st.button("Re-run the full evaluation (~250 AI calls, a few cents)"):
        bar = st.progress(0.0)
        results = evaluation.run_evaluation(bb, SCEN, progress=lambda f, t: bar.progress(f, text=t))
        st.success("Evaluation finished and saved to cache/eval_results.json")
    if not results:
        st.warning("No evaluation results yet. Run `python build_cache.py` with OPENAI_API_KEY set.")
    else:
        summ = pd.DataFrame(results["summary"]).T.reset_index().rename(columns={"index": "pipeline"})
        summ["description"] = summ["pipeline"].map(results["pipelines"])
        base = results["summary"]["baseline"]
        best = summ.sort_values(["context_recall", "answer_relevance"], ascending=False).iloc[0]
        rec_df = pd.DataFrame(results["search_recall_at_5"])
        ui.kpi_row([
            (f"{base['context_recall']:.2f}", "Coach context recall: baseline"),
            (f"{best['context_recall']:.2f}", f"Best: {best['pipeline']}",
             f"{best['context_recall'] - base['context_recall']:+.2f} vs. baseline"),
            (f"{results['refusal_accuracy']:.0%}", "Out-of-scope questions refused"),
            (f"{rec_df['hybrid'].mean():.2f}", "Job search recall@5 (smart)",
             f"{rec_df['hybrid'].mean() - rec_df['bm25'].mean():+.2f} vs. keywords"),
        ])
        st.write("")
        with st.container(key="card_eval_chart"):
            ui.section(f"Coach: baseline vs. each upgrade ({len(SCEN['coach_eval'])} questions)")
            long = summ.melt(id_vars="pipeline", value_vars=["context_recall", "faithfulness", "answer_relevance"],
                             var_name="metric", value_name="score")
            fig = px.bar(long, x="metric", y="score", color="pipeline", barmode="group", height=330,
                         range_y=[0, 1.05], color_discrete_sequence=["#a8a29e", "#0a66c2", "#0d9488", "#7c3aed", "#e8590c"])
            fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(title=None))
            st.plotly_chart(fig, width="stretch")
            for m in ("context_recall", "faithfulness", "answer_relevance"):
                summ[f"Δ {m}"] = summ[m] - base[m]
            st.dataframe(summ[["pipeline", "description", "context_recall", "faithfulness", "answer_relevance",
                               "answered", "Δ context_recall", "Δ answer_relevance"]].style.format(precision=3),
                         hide_index=True, width="stretch")
            st.caption("Method (labeled fallback): " + results["method"])

        with st.container(key="card_eval_ba"):
            ui.section("Before / after ranking", "What an upgrade changes for one question (✅ gold = the guide "
                       "that should be found)")
            e1, e2 = st.columns(2)
            qpick = e1.selectbox("Question", [x["q"] for x in SCEN["coach_eval"]], key="evalq")
            ppick = e2.selectbox("Upgrade", [p for p in advanced.PIPELINES if p != "baseline"],
                                 format_func=advanced.PIPELINES.get, key="evalp")
            gold = next(x["gold"] for x in SCEN["coach_eval"] if x["q"] == qpick)
            try:
                h, tr = advanced.retrieve(bb, qpick, ppick)
                b1, b2 = st.columns(2)
                b1.markdown("**Before:** vector top-3")
                b1.dataframe(pd.DataFrame(advanced.rank_table(bb, tr["baseline_pool"][:3], tr, gold)),
                             hide_index=True, width="stretch")
                b2.markdown(f"**After:** {advanced.PIPELINES[ppick]}")
                b2.dataframe(pd.DataFrame(advanced.rank_table(bb, h, tr, gold)), hide_index=True, width="stretch")
            except llm.NotCached:
                st.info(NOT_CACHED_MSG)

        rows_df = pd.DataFrame(results["rows"])
        with st.container(key="card_eval_fail"):
            ui.section("Observed failures", "Rows where the gold guide was missed, or the answer was refused or routed")
            fails = rows_df[(rows_df["context_recall"] < 1) | (rows_df["status"] != "answered")]
            st.dataframe(fails[["pipeline", "question", "gold", "retrieved", "status", "context_recall",
                                "answer_relevance"]], hide_index=True, width="stretch")
            st.markdown("<div class='stage'><b>Pre-retrieval:</b> the query or the corpus is the problem "
                        "(vocabulary mismatch, outdated documents, query drift). Fixes: query rewriting, metadata "
                        "filters, refreshing documents.</div><div class='stage'><b>Retrieval:</b> the right chunk "
                        "exists but isn't ranked high enough, or the top-k lacks diversity. Fixes: hybrid search, "
                        "re-ranking, a better embedding model.</div><div class='stage'><b>Post-retrieval:</b> the "
                        "right chunk was retrieved but the answer is wrong, conflicted, or unfaithful. Fixes: "
                        "conflict rules in the prompt, contextual compression, a judge.</div>",
                        unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="card_eval_ref"):
                ui.section("Refusals on out-of-scope questions")
                st.dataframe(pd.DataFrame(results["refusals"]), hide_index=True, width="stretch")
        with c2:
            with st.container(key="card_eval_search"):
                ui.section("Job search recall@5 by engine")
                st.dataframe(rec_df.rename(columns={"bm25": "Keywords", "vector": "Meaning", "hybrid": "Smart"}),
                             hide_index=True, width="stretch")
        with st.expander("Per-question scores for every pipeline"):
            st.dataframe(rows_df[["pipeline", "question", "context_recall", "faithfulness", "answer_relevance",
                                  "judge_badge", "faithfulness_note"]], hide_index=True, width="stretch")
        st.caption(f"Results recorded {results['created']}.")

# ============================================================================= HOW IT WORKS
with tabs[5]:
    with st.container(key="card_arch"):
        ui.section("One retrieval backbone, four products",
                   "Search, recommendations, the coach, and the employer copilot all use the same "
                   "embed → index → retrieve step. Only the query changes.")
        st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#e8f1fb", color="#b9d4f1", fontname="Helvetica", fontsize=10];
  A [label="Corpus A\\n44 postings\\n(1 posting = 1 chunk)"]; B [label="Corpus B\\n14 guides -> 56 chunks\\n(110 words, 30 overlap)"];
  C [label="Candidate pool\\n80 synthetic profiles"];
  E [label="all-MiniLM-L6-v2\\nembeddings (384-d)", fillcolor="#e6f4ea", color="#9bd3ae"];
  F [label="FAISS IndexFlatIP\\n+ BM25", fillcolor="#e6f4ea", color="#9bd3ae"];
  A -> E; B -> E; C -> E; E -> F;
  S [label="Jobs\\nquery = text\\nfew-shot filters + hybrid", fillcolor="#fff4e5", color="#f5c26b"];
  R [label="For You\\nquery = profile\\ngrounded explanations", fillcolor="#fff4e5", color="#f5c26b"];
  Q [label="Career Coach\\nquery = question\\nrerank/HyDE/fusion + RAG", fillcolor="#fff4e5", color="#f5c26b"];
  P [label="For Employers\\nquery = draft posting\\nreach funnel + flags", fillcolor="#fff4e5", color="#f5c26b"];
  F -> S; F -> R; F -> Q; F -> P;
  J [label="AI judge\\n(guardrail)", fillcolor="#fdecea", color="#f3b3ae"]; Q -> J; P -> J;
  V [label="Insights\\ncontext recall, faithfulness,\\nanswer relevance", fillcolor="#e6f4ea", color="#9bd3ae"]; Q -> V;
}""")
    with st.container(key="card_chapters"):
        ui.section("Course concepts used")
        st.markdown("""
| Feature | Course reference |
|---|---|
| Embeddings, FAISS, BM25, hybrid blend, cross-encoder re-rank, HyDE, RAGAS-style metrics | Ch4 slides + `04_rag_lab.ipynb` Parts 2-8 |
| Profile as query; new records count with no retraining | Ch4 "RAG meets the platform" + `04_credit_scoring_lab.ipynb` |
| Few-shot JSON extraction, persona system prompt, chain-of-thought scoring, LLM-as-judge | Ch3 slides + `03_prompt_engineering.ipynb` Steps 2, 4, 5, 6 |
| One `chat()` gateway with a cached/mock mode | Ch3/Ch4 notebooks' helper pattern |
| RAG fusion (reciprocal rank fusion), outdated-document check, chunk size and overlap | Ch4 "Advanced RAG" and "How RAG can go wrong" slides |
""")
        st.markdown("**Demo mode.** All retrieval (embeddings, FAISS, BM25, cross-encoder, PCA, reach funnel) runs "
                    "live on your machine. Only the gpt-4o-mini outputs are replayed from `cache/llm_cache.json`, "
                    "and each one is keyed to the exact prompt that produced it.")
