import streamlit as st
import os
import nltk
nltk.data.path.append(r"C:\Users\moazz\AppData\Roaming\nltk_data")

from utils import extract_text_from_pdf, clean_text
from model import calculate_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="🔥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --surface2:  #18181f;
    --border:    #2a2a38;
    --ember1:    #ff4d00;
    --ember2:    #ff8c00;
    --ember3:    #ffb347;
    --text:      #f0ede8;
    --muted:     #7a7890;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 760px; padding: 3rem 2rem 5rem !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ember2);
    border: 1px solid var(--ember1);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.4rem, 6vw, 3.6rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, var(--ember3) 0%, var(--ember1) 55%, #c2410c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    margin-top: 1rem;
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    line-height: 1.7;
}

/* ── Glow divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--ember1) 40%, var(--ember2) 60%, transparent 100%);
    margin: 2.2rem 0;
    opacity: 0.55;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ember2);
    margin-bottom: 0.6rem;
}

/* ── Upload box ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.25s;
    padding: 0.5rem 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--ember1) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stBaseButton-secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
    transition: border-color 0.25s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--ember1) !important;
    box-shadow: 0 0 0 3px rgba(255, 77, 0, 0.12) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ── Analyze button ── */
[data-testid="stBaseButton-primary"] button,
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--ember1), #c2410c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(255, 77, 0, 0.3) !important;
    margin-top: 0.5rem !important;
}
div.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(255, 77, 0, 0.45) !important;
}
div.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Score card ── */
.score-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.5rem 2rem 2rem;
    margin-top: 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--ember1), var(--ember2), var(--ember3));
}
.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, var(--ember3), var(--ember1));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.score-label {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* ── Progress bar ── */
.progress-wrap {
    background: var(--surface2);
    border-radius: 999px;
    height: 8px;
    margin: 1.6rem 0 1.2rem;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ember1), var(--ember3));
    transition: width 1s ease;
    box-shadow: 0 0 12px rgba(255, 140, 0, 0.6);
}

/* ── Verdict chip ── */
.verdict {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 1.2rem;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    margin-top: 0.8rem;
}
.verdict-excellent { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35); color: #4ade80; }
.verdict-moderate  { background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
.verdict-low       { background: rgba(255,77,0,0.10);   border: 1px solid rgba(255,77,0,0.3);   color: var(--ember2); }

/* ── Tips box ── */
.tips-box {
    background: var(--surface2);
    border-left: 3px solid var(--ember1);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1.4rem;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.75;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--ember1) !important; }

/* ── Misc ── */
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Instant Analysis</div>
    <h1>Resume Analyzer</h1>
    <p class="hero-sub">
        Drop your resume. Paste the job description.<br>
        Get a precision match score in seconds.
    </p>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

# ── Resume upload ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Your Resume</div>', unsafe_allow_html=True)
resume_file = st.file_uploader(
    "Upload resume",
    type=["pdf"],
    label_visibility="collapsed",
)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Job description ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Job Description</div>', unsafe_allow_html=True)
job_desc = st.text_area(
    "Job description",
    height=180,
    placeholder="Paste the full job description here — requirements, responsibilities, preferred skills…",
    label_visibility="collapsed",
)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
analyze = st.button("⚡  Analyze Match", use_container_width=True)

# ── Logic ─────────────────────────────────────────────────────────────────────
if analyze:
    if resume_file is None:
        st.error("Please upload a resume PDF to continue.")
    elif job_desc.strip() == "":
        st.error("Please paste a job description to continue.")
    else:
        with st.spinner("Analyzing your resume…"):
            temp_path = "temp_resume.pdf"
            with open(temp_path, "wb") as f:
                f.write(resume_file.getbuffer())

            resume_text = extract_text_from_pdf(temp_path)
            score = calculate_similarity(resume_text, job_desc)

        # ── Score card ──
        if score > 80:
            verdict_class = "verdict-excellent"
            verdict_icon  = "✦"
            verdict_text  = "Excellent Match"
            tip = "Your resume strongly aligns with this role. Ensure your achievements are quantified and your top keywords appear near the top of the document."
        elif score > 50:
            verdict_class = "verdict-moderate"
            verdict_icon  = "◈"
            verdict_text  = "Moderate Match"
            tip = "You're in the running, but consider adding missing skills from the job description, mirroring key phrases used by the employer, and tightening your summary section."
        else:
            verdict_class = "verdict-low"
            verdict_icon  = "◇"
            verdict_text  = "Low Match"
            tip = "Significant gaps detected. Highlight transferable skills, incorporate exact keywords from the posting, and tailor your work experience bullets to match the role."

        progress_width = min(score, 100)

        st.markdown(f"""
        <div class="score-card">
            <div class="score-number">{score}<span style="font-size:2rem; opacity:.5">%</span></div>
            <div class="score-label">Match Score</div>
            <div class="progress-wrap">
                <div class="progress-fill" style="width:{progress_width}%"></div>
            </div>
            <div class="verdict {verdict_class}">{verdict_icon}&nbsp;&nbsp;{verdict_text}</div>
            <div class="tips-box">
                <strong style="color: var(--ember2); font-size:0.7rem; letter-spacing:.1em; text-transform:uppercase;">
                    Recommendation
                </strong><br>{tip}
            </div>
        </div>
        """, unsafe_allow_html=True)