// Builds report/CareerCompass_Report.docx following the HW2 Report Template, sections 1-8 in order.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, PageBreak, Footer, PageNumber, BorderStyle,
} = require("docx");

const ROOT = process.argv[2];                      // CareerCompass project folder
const FIG = path.join(ROOT, "report", "figures");
const OUT = path.join(ROOT, "report", "CareerCompass_Report.docx");

const FONT = "Calibri";
const W = 9360;                                    // text width in DXA (8.5in - 2 x 1in)
let counted = [];                                  // words that count toward the limit

// ---------- helpers ----------
const runs = (text, opts = {}) => {
  // **bold** and *italic* inline markers
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
};
const plain = (s) => s.replace(/\*\*|\*/g, "");
const P = (text, o = {}) => {
  if (o.count !== false) counted.push(plain(text));
  return new Paragraph({ children: runs(text), spacing: { after: 120, line: 264 }, ...o.para });
};
const H1 = (text) => { counted.push(text); return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); };
const bullet = (text) => { counted.push(plain(text)); return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(text), spacing: { after: 60, line: 264 } }); };
const num = (text, ref) => { counted.push(plain(text)); return new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 60, line: 264 } }); };

// Highlighted box for the student's own analysis (not counted: the student's words replace it).
const yourTurn = (title, prompt) => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  rows: [new TableRow({ cantSplit: true, children: [new TableCell({
    width: { size: W, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "FFF4CE" },
    margins: { top: 120, bottom: 120, left: 160, right: 160 },
    borders: { top: { style: BorderStyle.SINGLE, size: 6, color: "E0B000" }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "E0B000" },
               left: { style: BorderStyle.SINGLE, size: 24, color: "E0B000" }, right: { style: BorderStyle.SINGLE, size: 6, color: "E0B000" } },
    children: [
      new Paragraph({ children: [new TextRun({ text: title, bold: true, color: "7A5A00" })], spacing: { after: 60 } }),
      new Paragraph({ children: [new TextRun({ text: prompt, italics: true, color: "5C4A00" })] }),
    ],
  })] })],
});

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "C8C8C8" };
const table = (header, rows, widths) => {
  rows.forEach(r => counted.push(r.join(" ")));
  counted.push(header.join(" "));
  const mk = (txt, head, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8F1FB" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    borders: { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder },
    children: [new Paragraph({ children: runs(txt, { size: 19, bold: head || undefined }) })],
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, true, widths[i])) }),
           ...rows.map(r => new TableRow({ children: r.map((c, i) => mk(c, false, widths[i])) }))],
  });
};

const pngSize = (file) => { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
const figure = (file, widthIn, caption) => {
  const f = path.join(FIG, file);
  const [w, h] = pngSize(f);
  const wPx = widthIn * 96, hPx = Math.round(wPx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: wPx, height: hPx },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }),
  ];
};
const spacer = () => new Paragraph({ spacing: { after: 60 }, children: [] });

// ---------- content ----------
const body = [];

// Title block
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "CareerCompass: One Retrieval Backbone, Three AI Products", bold: true, size: 34, color: "0A66C2" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "MIS 552 Homework 2 — Design and Evaluation Report", size: 24 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "[Your name]", highlight: "yellow", size: 22 }), new TextRun({ text: "  ·  October 2026", size: 22 })] }));

// 1. Architecture and data
body.push(H1("1. Architecture and data"));
body.push(P("CareerCompass runs four products on one retrieval backbone: all-MiniLM-L6-v2 embeddings, a FAISS index and BM25 over three corpora. Only the query changes: typed text (search), a profile (recommendations), a question (the coach) or a draft posting (the employer copilot). Coach and copilot drafts pass an LLM judge before they are shown."));
body.push(...figure("fig_pipeline.png", 6.3, "Figure 1. CareerCompass pipeline: 3 corpora, 1 backbone, 4 products"));
body.push(P("**Corpus A: 44 internship postings** across 8 categories (17 onsite, 13 hybrid, 14 remote; 3 unpaid; 25 support CPT/OPT). Each has 120+ words, with text that matches its flags. Retrieval challenges were planted deliberately: near-duplicates that differ in one detail (P001/P002 differ only in remote work and sponsorship; P021/P022 have the same text at two companies, paid vs. unpaid; P028/P029); rare exact terms in only one or two postings (PostgreSQL, SAP Ariba, OR-Tools); two paraphrase targets with zero keyword overlap with their queries, checked by script (P007, P027); and two legally required citizenship restrictions (P018, P039)."));
body.push(P("**Corpus B: 14 career guides** (320–349 words each). Two conflicts were planted: G01 vs. G02 on resume length, and the *archived* 2021 virtual-fair guide G12 vs. the current 2026 playbook G11. G10 (work authorization) is labeled simplified course material, not legal guidance. **Candidates:** 80 seeded synthetic profiles for the employer copilot, with no name, age, gender, race or address."));
body.push(table(["Decision", "Choice", "Justification (Chapter 4 trade-offs)"], [
  ["Posting chunks", "1 posting = 1 chunk", "It is the unit a student applies to; splitting would separate pay and sponsorship from the duties. 200–254 tokens, under MiniLM's 256-token limit."],
  ["Guide chunks", "Sentence-aware, ~110 words, 30-word overlap, title prefixed (56 chunks)", "Precise matches without truncation (max 236 tokens); the overlap keeps ideas that cross a boundary whole; the title gives each chunk context."],
  ["Embeddings + index", "all-MiniLM-L6-v2 (384-d), FAISS IndexFlatIP", "Free and local; inner product = cosine on normalized vectors; exact search at this scale."],
  ["Keyword retrieval", "BM25 (rank-bm25)", "Catches rare exact terms that embeddings blur."],
  ["Re-ranker", "ms-marco cross-encoder", "Reads the query and the chunk together; used on a top-10 pool."],
  ["LLM", "gpt-4o-mini, one cached gateway", "Every output is recorded, so the app runs without a key in demo mode."],
], [1700, 2900, 4760]));

// 2. Retrieval comparison
body.push(H1("2. Retrieval comparison"));
body.push(P("Across 7 benchmark queries, average recall@5 was 0.71 for BM25, 0.79 for vector search and 0.86 for the 50/50 hybrid. Each method failed on a different kind of query (ranks out of 44 postings, before filters):"));
body.push(table(["Benchmark query", "Type", "Expected", "BM25", "Vector", "Hybrid"], [
  ["PostgreSQL", "exact term", "P003, P012", "1, 2", "1, 2", "1, 2"],
  ["SAP Ariba", "exact term", "P043", "1", "6", "1"],
  ["OR-Tools", "exact term", "P044", "1", "5", "1"],
  ["Figma prototyping", "exact term", "P025/28/29", "3, 2, 1", "3, 1, 2", "3, 2, 1"],
  ["help a hospital understand why patients miss appointments", "paraphrase", "P007", "43", "1", "7"],
  ["make websites usable for people who are blind", "paraphrase", "P027", "21", "1", "1"],
  ["remote data analytics internship at a startup that sponsors international students", "mixed", "P001, P005", "2, 1", "1, 11", "1, 2"],
], [3960, 1200, 1200, 1000, 1000, 1000]));
body.push(spacer());
body.push(P("In the app, few-shot filter extraction turns the last query into Data Analytics + remote + supports CPT/OPT, which leaves exactly P001 and P005. Semantic-map facts: for 25 of 44 postings the nearest neighbor is in a *different* category, often the same company's posting (Quillstone's BI and UX Research roles, cosine 0.79). Cybersecurity is the most distinct category (within 0.64 vs. between 0.47), Product Management the least (0.59 vs. 0.52), and P027 is the most isolated posting (nearest neighbor 0.54). PCA keeps 18% of the variance."));
body.push(...figure("fig_semantic_map.png", 5.6, "Figure 2. Semantic-space map: postings by category, the query as a star, results circled"));
body.push(yourTurn("YOUR ANALYSIS (about 120–180 words)",
  "When does each method win, and why (Chapter 4: rare tokens vs. meaning)? Why does the hybrid lose the hospital paraphrase (rank 7) while BM25 noise doesn't hurt the screen-reader query? Which method would you deploy and what would you tune? Give one insight from the map in your own words."));

// 3. Recommendation honesty
body.push(H1("3. Recommendation honesty"));
body.push(P("The deliberately mismatched profile, a culinary student, was scored 0–2 out of 100 by both prompt versions on all five postings. The naive first prompt (v1) instead oversold *partial* matches: its average fit was 68–80 vs. 34–51 for v2, for example a nonprofit full-stack role at 85 vs. 40, and a role with no CPT/OPT support at 60 vs. 0 for the international student. v1 gave almost no quotes; v2's quotes were verified word for word in code (102 of 104 found). The v2 change added reasoning before the score, a rubric, mandatory verbatim quotes from both texts, a rule to call weak matches weak, and a gaps field. Must-have conflicts are shown as warnings, never hidden."));
body.push(yourTurn("YOUR ANALYSIS (about 70–100 words)",
  "Did the explanations stay honest on the mismatched profile? Why might v1 refuse to oversell an obvious mismatch but inflate partial ones? Which v2 change mattered most, and what would you still change?"));

// 4. Evaluation and failure diagnosis
body.push(H1("4. Evaluation and failure diagnosis"));
body.push(P("Re-ranking raised context recall from 0.90 to 0.97, but answer relevance fell from 0.99 to 0.93 because one in-scope question was routed to a human. RAG fusion was the only upgrade with lower recall than the baseline."));
body.push(table(["Pipeline (15 in-scope questions)", "Context recall", "Faithfulness", "Answer relevance", "Answered"], [
  ["Baseline: vector top-3", "0.90", "1.00", "0.99", "15"],
  ["Re-rank: top-10 → cross-encoder top-3", "0.97", "1.00", "0.93", "14"],
  ["HyDE", "0.93", "1.00", "0.93", "14"],
  ["RAG fusion: 3 rewrites + RRF", "0.87", "1.00", "0.93", "14"],
  ["HyDE + re-rank (upgraded)", "0.97", "1.00", "0.93", "14"],
], [3760, 1400, 1400, 1600, 1200]));
body.push(spacer());
body.push(P("Method (labeled RAGAS fallback): exact context recall against gold guide ids, plus faithfulness and answer relevance scored by a gpt-4o-mini judge, as in Chapter 4 lab Part 8. All 4 out-of-scope questions were refused (3 by the coach prompt, 1 by the similarity gate). Observed failures:"));
body.push(num("*Expo on Zoom?* The baseline retrieved the **archived** 2021 guide (G12) and answered \"virtual\". The judge passed it, because it was faithful to stale context. With archived guides ignored, it was answered correctly from G11.", "fails"));
body.push(num("The same question under re-rank, HyDE and the upgraded pipeline retrieved both G11 and G12. The judge failed the conflicted draft twice, and it was routed to a human (relevance 0).", "fails"));
body.push(num("*Printed resumes at the career fair?* HyDE and fusion drifted to resume guides (G01, G02) and missed G11 (recall 0).", "fails"));
body.push(num("*Two offers, $28 vs. $32:* three pipelines returned only G07 chunks and missed G06 (recall 0.5).", "fails"));
body.push(num("Faithfulness was 1.00 everywhere: the judge never caught failure 1.", "fails"));
body.push(yourTurn("YOUR ANALYSIS (about 220–280 words)",
  "Interpret the table: which upgrade would you ship, and why? Report and diagnose each metric that got worse (fusion's recall, and the relevance drop) with your hypothesis. Place at least one failure above in the Chapter 4 framework (pre-retrieval, retrieval or post-retrieval) and propose a fix. Is faithfulness 1.00 trustworthy, given failure 5?"));

// 5. Innovation feature rationale
body.push(H1("5. Innovation feature rationale"));
body.push(P("**Employer Reach Copilot.** Before publishing, an employer sees how many qualified students each requirement in a draft posting excludes. On the over-restricted demo draft, the suggested fixes raised qualified reach from 0 to 14 of 26 relevant students."));
body.push(P("**Stakeholder and decision.** The stakeholder is the employer, and through them the platform, whose marketplace works only when postings reach qualified students. The decision supported: which requirements are worth their cost in reach? Employers often paste full-time requirements (\"3+ years\", \"recent graduates only\") into internship postings and never see who they lost."));
body.push(P("**How it works.** Few-shot extraction (Chapter 3) turns the draft into constraints. The draft is then the *query* over 80 candidate profiles (Chapter 4; cosine ≥ 0.45 = relevant), and a reach funnel is computed in code. An LLM flags exclusionary wording, a code check flags any work-authorization restriction with no legal basis, and an LLM judge scores every flag. Finally, the rewrite's reach is re-measured against the same relevant pool."));
body.push(table(["Requirement in the demo draft", "Relevant students it excludes on its own (of 26)"], [
  ["3+ years of professional experience", "26"], ["Recent graduates only (Class of 2025)", "26"], ["Unpaid", "21"],
  ["Onsite 5 days a week", "11"], ["U.S. citizens only (no legal reason given)", "8"], ["SQL or Tableau", "2"],
], [5360, 4000]));
body.push(spacer());
body.push(P("**Risk and mitigation.** The risk: optimizing for reach could pressure an employer to drop a genuinely needed requirement, or the model could misjudge a lawful restriction (export control, clearances). The mitigation: restrictions with a stated legal basis are always kept; the employer approves each fix with the judge's rationale shown; nothing is published automatically; only aggregate counts are shown; and profiles carry no protected attributes."));

// 6. Ethics note
body.push(H1("6. Ethics note"));
body.push(yourTurn("YOUR NOTE (2–3 sentences)",
  "Pick at least one issue in your own words: fairness in matching (shared company and program vocabulary can favor some backgrounds); visa-status handling (CPT/OPT is shown as a warning, never used to silently hide jobs); or the harm of hallucinated advice (failure 1: a faithful-but-outdated answer passed the judge)."));

// 7. AI-use disclosure
body.push(H1("7. AI-use disclosure"));
body.push(table(["Tool", "Tasks", "How the output was verified"], [
  ["Claude Code (Claude Opus 5.5)", "Wrote the Streamlit app and the retrieval, recommendation, coach, evaluation and employer modules, following the course notebooks (Chapters 3–4).", "Headless tests of every tab; a script that replays every preset with no API key; review in the browser."],
  ["Claude Code", "Drafted both corpora from the specification in prompts.md; wrote the seeded candidate generator.", "Scripts checked word counts, flag/text consistency and zero keyword overlap for the paraphrase targets."],
  ["Claude Code", "Calibrated thresholds by measurement; found and fixed defects (a missed citizenship flag, an uninformative cost metric); drafted the factual parts of this report.", "Before/after numbers compared in the app and in cache/eval_results.json."],
  ["gpt-4o-mini (OpenAI API)", "All in-app generation and judging (filters, explanations, coach answers, fact-checks, HyDE, rewrites, RAGAS-style scores).", "Code-side checks: quote verification, citation checks, gold-set context recall."],
], [2100, 4060, 3200]));
body.push(spacer());
body.push(yourTurn("ADD YOUR PART (1–2 sentences)",
  "How you verified the work yourself (what you ran, read or changed), and a statement that the analysis in sections 2, 3, 4 and 6 is your own."));

// 8. Appendix
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Appendix: live-mode screenshots")] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Not counted in the word limit. Captured with OPENAI_API_KEY set (the top bar shows “Live AI”); each request below was new, so it was answered live by gpt-4o-mini rather than replayed from the cache.", italics: true, color: "555555" })] }));
body.push(...figure("live_1_jobs.png", 6.4, "Screenshot 1. Jobs: a new query, with live few-shot filter extraction and filtered hybrid results"));
body.push(...figure("live_2_coach.png", 6.4, "Screenshot 2. Career Coach: a new question answered live with inline citations, a fact-check badge and sources"));
body.push(...figure("live_3_employer.png", 6.4, "Screenshot 3. For Employers: a new draft, with live constraint extraction, reach numbers and judged fixes"));

// ---------- document ----------
const doc = new Document({
  creator: "CareerCompass", title: "CareerCompass Design and Evaluation Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "0A66C2" },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "fails", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "CareerCompass · MIS 552 HW2 · page ", size: 16, color: "888888" }),
                 new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  const words = counted.join(" ").split(/\s+/).filter(Boolean).length;
  console.log("wrote", OUT);
  console.log("counted words (sections 1-7, excluding your-analysis boxes and appendix):", words);
});
