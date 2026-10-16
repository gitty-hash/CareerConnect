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
const H1 = (text) => { counted.push(text); return new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, keepLines: true, children: [new TextRun(text)] }); };
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
const sub = (text) => { counted.push(text); return new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
  children: [new TextRun({ text, bold: true, color: "004182" })] }); };

// Title block
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "CareerCompass: One Retrieval Backbone, Three AI Products", bold: true, size: 34, color: "0A66C2" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "MIS 552 Homework 2 — Design and Evaluation Report", size: 24 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "[Your name]", highlight: "yellow", size: 22 }), new TextRun({ text: "  ·  October 2026", size: 22 })] }));

// 1. Architecture and data
body.push(H1("1. Architecture and data"));
body.push(P("CareerCompass runs four products on one retrieval backbone: all-MiniLM-L6-v2 embeddings, FAISS and BM25. Only the query changes: typed text (search), a profile (recommendations), a question (the coach) or a draft posting (the employer copilot). Coach answers and copilot flags pass an LLM judge before they are shown."));
body.push(...figure("fig_pipeline.png", 6.1, "Figure 1. CareerCompass pipeline: 3 corpora, 1 backbone, 4 products"));
body.push(P("**The data was designed to make retrieval fail in specific, visible ways.** Corpus A has 44 postings across 8 categories (17 onsite, 13 hybrid, 14 remote; 3 unpaid; 25 open to CPT/OPT). Corpus B has 14 career guides of 320–349 words. Each trap tests one stage of the pipeline, and the demo handles each one on screen:"));
body.push(bullet("**Near-duplicates test the filters, not the embeddings.** P001 and P002 are the same Saguaro Analytics role (cosine 0.90), one remote with CPT/OPT and one onsite without sponsorship. Vector search ranks them #1 and #2 for the brief's own query; only the few-shot filters (remote + CPT/OPT) separate them, and the Jobs tab shows exactly that."));
body.push(bullet("**Rare exact terms and zero-overlap paraphrases test the hybrid** (PostgreSQL, SAP Ariba and OR-Tools appear in one or two postings; P007 and P027 share no keyword with their queries; see Section 2)."));
body.push(bullet("**Conflicting guides test the coach.** G01 (one page for undergraduates) and G02 (two pages for researchers) are retrieved together for both resume questions; Juno answered the undergraduate “one page [G01]” and the PhD student “two pages [G02]”. G12, an **archived** 2021 virtual-fair guide, contradicts the current G11 playbook; this trap produced the most important failure in Section 4."));
body.push(P("Corpus B also labels G10 (work authorization) as simplified course material. 80 seeded candidate profiles, with no name, age, gender or race, power the employer copilot."));
body.push(table(["Decision", "Choice", "Justification (Chapter 4 trade-offs)"], [
  ["Posting chunks", "1 posting = 1 chunk", "The unit a student applies to; splitting would separate pay and sponsorship from the duties. 200–254 tokens, under MiniLM's 256-token limit, with structured facts first."],
  ["Guide chunks", "Sentence-aware, ~110 words, 30-word overlap, title prefix (56 chunks)", "Small enough for precise matches without truncation (max 236 tokens); the overlap keeps ideas that cross a boundary whole; the title gives each chunk context."],
  ["Embeddings + index", "all-MiniLM-L6-v2, FAISS IndexFlatIP", "Free and local; inner product = cosine; exact search at this scale."],
  ["Keyword + re-rank", "BM25; ms-marco cross-encoder", "BM25 catches rare terms; the cross-encoder reads query and chunk together."],
], [1700, 2900, 4760]));

// 2. Retrieval comparison
body.push(H1("2. Retrieval comparison"));
body.push(P("**Keywords win when the query contains a rare, exact token; meaning wins when the query shares no words with the posting; and a 50/50 hybrid fails when its keyword side matches only filler words.** Ranks of the expected posting (out of 44):"));
body.push(table(["Benchmark query", "Type", "BM25", "Vector", "Hybrid", "Hybrid, stopwords removed"], [
  ["SAP Ariba", "exact term", "1", "6", "1", "1"],
  ["OR-Tools", "exact term", "1", "5", "1", "2"],
  ["PostgreSQL (2 postings)", "exact term", "1, 2", "1, 2", "1, 2", "1, 2"],
  ["help a hospital understand why patients miss appointments", "paraphrase", "43", "1", "7", "1"],
  ["make websites usable for people who are blind", "paraphrase", "21", "1", "1", "2"],
  ["remote data analytics internship at a startup that sponsors international students (2 postings)", "mixed", "2, 1", "1, 11", "1, 2", "1, 2"],
  ["Average recall@5 over all 7 queries (incl. Figma prototyping)", "", "0.71", "0.79", "0.86", "1.00"],
], [3860, 1100, 900, 900, 900, 1700]));
body.push(spacer());
body.push(P("BM25 wins on product names because a token that appears in one posting has a very high inverse document frequency, so it is decisive. MiniLM breaks “SAP Ariba” into common sub-words that mean roughly “enterprise software”, so the right posting sinks to rank 5–6 among similar roles. The paraphrase queries reverse this: BM25 ranks P007 at 43 because “no-shows” and “booked visits” never match “miss appointments”, while vector search ranks it first."));
body.push(P("The hybrid's failure on the hospital query is the most useful result. BM25's top match (P009, a fintech role) shared only {a, help, understand}, but min-max normalization rescales the best BM25 score to 1.0, however weak it is, so filler-word noise counted as much as a strong semantic match and pushed P007 to rank 7. I tested the fix: removing stopwords before BM25 lifted the hybrid to recall@5 1.00 and kept every exact-term win, while rank fusion (RRF) without stopword removal scored only 0.71. **The rule: blend keywords and meaning, but never let the keyword side vote on words that carry no meaning.** Stopword-aware BM25 is the first change I would ship."));
body.push(...figure("fig_semantic_map.png", 5.0, "Figure 2. Semantic-space map (PCA): postings by category, query as a star"));
body.push(P("**Map insight.** Categories are not clean clusters: for 25 of 44 postings the nearest neighbor is in another category, usually a posting from the same company (Quillstone's BI and UX Research roles, cosine 0.79). The embedding is driven by the company story in the first paragraph, not by the role. This is why category filters matter, and why posting text should lead with the role and skills. Cybersecurity and UX/UI Design are the tightest clusters (within-category similarity 0.64 and 0.63, against about 0.47 to other categories)."));

// 3. Recommendation honesty
body.push(H1("3. Recommendation honesty"));
body.push(P("The mismatched culinary profile stayed honest under both prompts (fit 0–2 of 100 on all five postings): when a match is obviously absent, even a naive prompt will not invent one. The dishonesty appeared in *partial* matches. Asked “why is this a great fit?”, the v1 prompt scored the international student 60 for P004 while its own explanation said the role does not support CPT/OPT, “which is essential”; the framing pulled the score up even though the model saw the dealbreaker. Across all profiles v1 averaged 68–80 against 34–51 for v2 and gave almost no evidence. I changed the prompt to reason before scoring (requirements, then evidence, then dealbreakers), added a rubric, and required verbatim quotes from both texts that code verifies: v2 scored P004 at 0, quoting “we do not offer visa sponsorship”, and 102 of 104 quotes were found word for word. One miss quoted “supports” where the posting says “support”, so next I would have the model cite numbered sentences instead of retyping quotes."));

// 4. Evaluation and failure diagnosis
body.push(H1("4. Evaluation and failure diagnosis"));
body.push(P("Method (labeled RAGAS fallback): context recall computed exactly against each question's gold guides, plus faithfulness and answer relevance scored by a gpt-4o-mini judge, as in Chapter 4 lab Part 8; 15 in-scope questions. All 4 out-of-scope questions were refused (3 by the coach prompt, 1 by the similarity gate)."));
body.push(table(["Pipeline", "Context recall", "Faithfulness", "Answer relevance", "Answered"], [
  ["Baseline: vector top-3", "0.90", "1.00", "0.99", "15"],
  ["Re-rank: top-10 → cross-encoder top-3", "0.97", "1.00", "0.93", "14"],
  ["HyDE", "0.93", "1.00", "0.93", "14"],
  ["RAG fusion: 3 rewrites + RRF", "0.87", "1.00", "0.93", "14"],
  ["HyDE + re-rank (upgraded)", "0.97", "1.00", "0.93", "14"],
], [3760, 1400, 1400, 1600, 1200]));
body.push(spacer());
body.push(P("**What I would ship: re-ranking, plus the two fixes below.** Re-ranking reaches the best recall (0.97) with no extra LLM call; HyDE + re-rank ties it but pays for a generated answer on every question. As in the lab, evaluation is what lets you buy the cheaper upgrade."));
body.push(sub("Two metrics got worse"));
body.push(bullet("**RAG fusion's recall fell to 0.87.** For “Should I bring printed copies of my resume to the career fair?”, the rewrites became “hard copies of my CV for the career expo” and “print my resume for the employment event”. Every rewrite repeated “resume/CV” and none kept “career fair”, and reciprocal rank fusion rewards documents that appear in many lists, so the resume guides outvoted G11. HyDE failed the same question for the same reason: its hypothetical answer was about resumes. More queries amplify whatever the rewrites over-emphasize."));
body.push(bullet("**Answer relevance fell from 0.99 to 0.93, and the drop is the safety layer working.** All of it comes from one question, “Is the fall recruiting expo on Zoom?”. The upgraded pipelines retrieved both the current G11 and the archived G12; the judge rejected the contradictory draft twice and routed the student to an advisor. The baseline scored higher only because it confidently answered “the expo is virtual” from the archived guide. The metric rewarded the wrong answer."));
body.push(sub("Failure diagnosis"));
body.push(P("**Failure: the outdated “virtual expo” answer. Stage: pre-retrieval** — the corpus contained a stale document and nothing before retrieval removed it, so better retrieval and a stricter judge could only route around it. **Fix (tested):** filter out guides marked archived before retrieval. The question was then answered correctly from G11 and passed the judge, and context recall rose from 0.90 to 0.97 for the baseline and from 0.87 to 0.93 for fusion. A second, retrieval-stage failure: for the two-offer question all three chunks came from the negotiation guide (G07), missing G06 on comparing offers. Capping retrieval at one chunk per guide brought both baseline and re-rank to **recall 1.00** in my retrieval test."));
body.push(P("**Faithfulness of 1.00 is not trustworthy.** The judge checks only whether claims are supported by the retrieved text, so a faithful answer built on an outdated guide passes; it is also the same model family as the generator. Next I would add a document-freshness check, a small human-labeled spot check of judge verdicts, and a judge prompt that sees each source's date and status."));

// 5. Innovation feature rationale
body.push(H1("5. Innovation feature rationale"));
body.push(P("**Employer Reach Copilot.** The stakeholder is the employer, and through them the platform, whose marketplace works only when postings reach qualified students. The decision it supports: which requirements in my draft are worth their cost in reach? The draft becomes the query over 80 candidate profiles (Chapter 4); few-shot extraction (Chapter 3) turns it into constraints; a funnel computed in code counts the students left after each requirement; an LLM flags exclusionary wording, with a code check for citizenship rules that give no legal reason; a judge scores each flag; and the rewrite's reach is re-measured against the same relevant pool."));
body.push(P("On the over-restricted demo draft, reach rose from **0 to 14 of 26** relevant students. The first version of the cost table showed “0 regained” for every requirement, because two requirements (“3+ years” and “Class of 2025 only”) each excluded all 26 students, so dropping either one alone changed nothing. Employers would read that as “nothing matters”, so the copilot now reports what each requirement excludes on its own (26, 26, 21 unpaid, 11 onsite, 8 citizens-only) and says when several must change together. On a new live draft, reach rose from 1 to 5 of 8."));
body.push(P("**Risk:** optimizing for reach could pressure an employer to drop a requirement that is genuinely needed, or the model could misjudge a lawful restriction such as export control. **Mitigation:** restrictions with a stated legal basis are always kept and never flagged for removal (P018 and P039 test this), the employer approves every fix with the judge's rationale, nothing is published automatically, and only aggregate counts are shown, never an individual student."));

// 6. Ethics note
body.push(H1("6. Ethics note"));
body.push(P("The most serious harm I observed was confident, outdated advice that passed a fact-checker: a student told the expo is virtual would miss an in-person recruiting event, which is why freshness has to be enforced before retrieval rather than trusted to a judge. Work authorization is the other high-stakes attribute: CareerCompass shows CPT/OPT conflicts as visible warnings instead of silently filtering jobs out, labels its visa guide as simplified course material, and the employer copilot challenges citizenship requirements only when no legal basis is given."));

// 7. AI-use disclosure
body.push(H1("7. AI-use disclosure"));
body.push(table(["Tool", "Tasks", "How the output was verified"], [
  ["Claude Code (Claude Opus 5.5)", "Wrote the Streamlit app and all modules following the course notebooks; drafted the synthetic corpora; calibrated thresholds; ran the retrieval experiments behind Sections 2–4; drafted this report, including its interpretations.", "Headless tests of every tab; tools/verify_cached.py; scripted data checks; every number traced to cache/eval_results.json or a re-runnable experiment; the author reviewed and edited the report."],
  ["gpt-4o-mini (OpenAI API)", "All in-app generation and judging: filters, explanations, coach answers, fact-checks, HyDE, rewrites, RAGAS-style scores.", "Code-side checks: quote verification, citation checks, gold-set context recall, and the failure analysis above."],
], [2100, 4060, 3200]));

// 8. Appendix
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Appendix: live-mode screenshots")] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Not counted in the word limit. Captured with OPENAI_API_KEY set; each request was new, so gpt-4o-mini answered it live rather than replaying the cache.", italics: true, color: "555555" })] }));
body.push(...figure("live_1_jobs.png", 6.4, "Screenshot 1. Jobs: a new query, with live few-shot filter extraction and filtered hybrid results"));
body.push(...figure("live_2_coach.png", 6.4, "Screenshot 2. Career Coach: a new question answered live with inline citations, a fact-check badge and sources (archived G12 flagged)"));
body.push(...figure("live_3_employer.png", 6.4, "Screenshot 3. For Employers: a new draft, with live constraint extraction and reach rising from 1 to 5 students"));

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
