"""Generate the synthetic student candidate pool used by the Employer Reach Copilot (Part F).

Seeded and deterministic, so the pool is reproducible. Following the Chapter 4
credit-scoring lab (Part 6, "What the data leaves out"), profiles contain NO name,
age, gender, race, photo, or address - only what an employer may legitimately
screen on: program, skills, experience, work-mode preference, and whether the
student needs employer support for work authorization.

Run:  python data/generate_candidates.py   ->  writes data/candidates.json
"""

import json
import random
from pathlib import Path

random.seed(552)

PROGRAMS = {
    "Business Analytics": ["SQL", "Excel", "Tableau", "Python", "statistics", "Power BI", "A/B testing"],
    "Information Systems": ["SQL", "Python", "Power BI", "systems analysis", "Excel", "project management"],
    "Computer Science": ["Python", "Java", "React", "Git", "C++", "SQL", "Docker", "algorithms"],
    "Data Science": ["Python", "pandas", "scikit-learn", "SQL", "statistics", "PyTorch", "Tableau"],
    "Marketing": ["social media", "copywriting", "Canva", "Google Analytics", "SEO", "content writing"],
    "Finance": ["Excel", "financial modeling", "valuation", "PowerPoint", "accounting"],
    "Industrial Engineering": ["process improvement", "Lean", "Excel", "linear programming", "Python"],
    "Supply Chain Management": ["Excel", "inventory planning", "SQL", "Tableau", "vendor management"],
    "Cybersecurity": ["networking", "Linux", "Splunk", "incident response", "Python"],
    "Human-Computer Interaction": ["Figma", "user interviews", "prototyping", "usability testing"],
}

LEVELS = [("sophomore", 2029), ("junior", 2028), ("senior", 2027), ("master's student", 2027)]

INTERESTS = {
    "Business Analytics": "data analyst or business intelligence internships",
    "Information Systems": "business intelligence or IT consulting internships",
    "Computer Science": "software engineering internships",
    "Data Science": "data science or machine learning internships",
    "Marketing": "digital and social media marketing internships",
    "Finance": "financial analyst internships",
    "Industrial Engineering": "operations and process improvement internships",
    "Supply Chain Management": "supply chain analyst internships",
    "Cybersecurity": "security operations internships",
    "Human-Computer Interaction": "UX design and research internships",
}

EXPERIENCE_PHRASES = {
    0.0: "No prior internship yet; experience comes from class projects and student clubs.",
    0.5: "One part-time campus job using these skills for about six months.",
    1.0: "One previous summer internship.",
    2.0: "Two previous internships plus a year of part-time work.",
}

candidates = []
cid = 1
for program, skills in PROGRAMS.items():
    for _ in range(8):  # 10 programs x 8 = 80 candidates
        level, grad_year = random.choice(LEVELS)
        exp = random.choices([0.0, 0.5, 1.0, 2.0], weights=[4, 3, 2, 1])[0]
        if level == "sophomore":
            exp = min(exp, 0.5)
        # Roughly one in three students in the pool needs CPT/OPT support (international students).
        needs_sponsorship = random.random() < 0.35
        modes = random.choice([["remote", "hybrid", "onsite"], ["remote", "hybrid"], ["remote"],
                               ["hybrid", "onsite"], ["remote", "hybrid", "onsite"]])
        paid_only = random.random() < 0.7          # most students cannot afford unpaid work
        my_skills = random.sample(skills, k=min(len(skills), random.randint(3, 5)))
        grad_class = grad_year if level != "master's student" else grad_year
        summary = (f"{level.capitalize()} in {program} (class of {grad_class}). "
                   f"Skills: {', '.join(my_skills)}. {EXPERIENCE_PHRASES[exp]} "
                   f"Interested in {INTERESTS[program]}; open to {', '.join(modes)} work.")
        candidates.append({
            "candidate_id": f"C{cid:03d}",
            "program": program,
            "level": level,
            "grad_year": grad_class,
            "skills": my_skills,
            "experience_years": exp,
            "needs_sponsorship": needs_sponsorship,
            "work_modes": modes,
            "paid_only": paid_only,
            "summary": summary,
        })
        cid += 1

out = Path(__file__).with_name("candidates.json")
out.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
print(f"wrote {len(candidates)} candidates -> {out}")
