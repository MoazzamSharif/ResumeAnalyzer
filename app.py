# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import nltk

nltk.data.path.append(r"C:\Users\moazz\AppData\Roaming\nltk_data")

from utils import extract_text_from_pdf
from model import calculate_similarity, extract_skill_keywords, extract_meaningful_keywords

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --ink:      #1c1917;
    --ink2:     #44403c;
    --ink3:     #78716c;
    --bg:       #262220;
    --surface:  #2e2a28;
    --surface2: #363230;
    --border:   #3d3935;
    --border2:  #504c49;
    --cream:    #e8e2d9;
    --cream2:   #d6cfc5;
    --accent:   #a3b899;
    --accent2:  #7a9e6e;
    --radius:   5px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--cream) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer, #MainMenu { display: none !important; visibility: hidden !important; }

.block-container {
    max-width: 680px !important;
    padding: 4rem 2rem 6rem !important;
}

/* Header */
.site-header {
    border-bottom: 1px solid var(--border2);
    padding-bottom: 2.25rem;
    margin-bottom: 3rem;
}
.site-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 0.85rem;
}
.site-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.2rem, 5vw, 3.2rem) !important;
    font-weight: 400 !important;
    line-height: 1.1 !important;
    color: var(--cream) !important;
    margin: 0 0 0.85rem !important;
    letter-spacing: -0.02em;
}
.site-desc {
    font-size: 0.875rem;
    color: var(--ink3);
    line-height: 1.75;
    font-weight: 300;
}

/* Labels */
.field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 0.5rem;
    display: block;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 0.25rem 0.75rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--border2) !important; }
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--ink3) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stBaseButton-secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--cream2) !important;
    border-radius: var(--radius) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* Textarea */
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--cream) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
    resize: vertical !important;
    transition: border-color 0.2s;
    caret-color: var(--cream);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--border2) !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--ink3) !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* Button */
div.stButton > button {
    width: 100% !important;
    background: var(--cream) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.15s !important;
    margin-top: 0.25rem !important;
}
div.stButton > button:hover {
    background: var(--cream2) !important;
    transform: translateY(-1px) !important;
}

/* Result */
.result-wrap {
    margin-top: 2.5rem;
    border-top: 1px solid var(--border2);
    padding-top: 2rem;
}
.result-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 1.25rem;
}
.score-row {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.score-big {
    font-family: 'Playfair Display', serif;
    font-size: 6rem;
    font-weight: 400;
    line-height: 1;
    color: var(--cream);
    letter-spacing: -0.03em;
}
.score-denom {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    color: var(--ink3);
    font-weight: 300;
}
.score-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
    font-style: italic;
    color: var(--ink3);
    margin-bottom: 1.75rem;
}
.bar-track {
    background: var(--surface2);
    border-radius: 2px;
    height: 2px;
    margin-bottom: 2.25rem;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--cream2);
}

/* Skills */
.skills-heading {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 2rem 0 0.75rem;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.75rem; }
.tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.22rem 0.55rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.tag-hit  { background: rgba(122,158,110,0.18); color: #a3c494; border: 1px solid rgba(122,158,110,0.3); }
.tag-miss { background: var(--surface2); color: var(--ink3); border: 1px solid var(--border); }
.skills-tally {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    color: var(--ink3);
    margin-top: 0.4rem;
}

/* Note */
.result-note {
    font-size: 0.82rem;
    font-weight: 300;
    color: var(--ink3);
    line-height: 1.8;
    margin-top: 1.75rem;
    padding: 1.25rem 1.4rem;
    background: var(--surface);
    border-left: 2px solid var(--border2);
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* Debug */
.debug-wrap {
    margin-top: 2rem;
    border-top: 1px solid var(--border);
    padding-top: 1.25rem;
}
.debug-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 0.5rem;
}
.debug-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink3);
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--ink3) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--ink3) !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown("""
<div class="site-header">
    <div class="site-kicker">Career Tools</div>
    <h1 class="site-title">Resume Analyzer</h1>
    <p class="site-desc">Upload your resume and paste a job description.<br>
    We score how well your experience matches the role.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------
st.markdown('<span class="field-label">Resume &mdash; PDF only</span>', unsafe_allow_html=True)
resume_file = st.file_uploader("Upload resume", type=["pdf"], label_visibility="collapsed")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

st.markdown('<span class="field-label">Job Description</span>', unsafe_allow_html=True)
job_desc_input = st.text_area(
    "Job description",
    height=200,
    placeholder="Paste the full job description here...",
    label_visibility="collapsed",
)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    analyze = st.button("Analyze", use_container_width=True)
with col2:
    show_debug = st.checkbox("Debug", value=False)

# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------
if analyze:
    if resume_file is None:
        st.error("Please upload a resume PDF.")
    elif not job_desc_input.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Scoring..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name

            try:
                resume_text = extract_text_from_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)

            job_text = job_desc_input.strip()

            resume_skills = extract_skill_keywords(resume_text)
            job_skills    = extract_skill_keywords(job_text)
            resume_kw     = extract_meaningful_keywords(resume_text)
            job_kw        = extract_meaningful_keywords(job_text)

            score = calculate_similarity(resume_text, job_text)

            matched_skills = sorted(resume_skills & job_skills)
            missing_skills = sorted(job_skills - resume_skills)

        # Verdict
        if score >= 70:
            verdict = "Strong match."
            note = ("Your resume covers the core requirements well. Before applying, make sure "
                    "your most relevant experience appears in the top third of the document and "
                    "that key terms mirror the exact phrasing used in the posting.")
        elif score >= 40:
            verdict = "Partial match."
            note = ("You meet some requirements but gaps remain. Review missing skills below "
                    "and where honest, incorporate them. Tailoring your summary to reflect the "
                    "role's language will help your application stand out.")
        else:
            verdict = "Weak match."
            note = ("Your resume does not closely align with this role. Consider whether this "
                    "position suits your background, or do a targeted rewrite addressing the "
                    "specific skills and responsibilities listed in the description.")

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-meta">Analysis complete &nbsp;&mdash;&nbsp; {resume_file.name}</div>
            <div class="score-row">
                <span class="score-big">{int(score)}</span>
                <span class="score-denom">/ 100</span>
            </div>
            <div class="score-verdict">{verdict}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{min(int(score),100)}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if job_skills:
            st.markdown('<div class="skills-heading">Skills from job description</div>', unsafe_allow_html=True)
            tags_html = '<div class="tag-row">'
            for s in sorted(job_skills):
                cls = "tag-hit" if s in resume_skills else "tag-miss"
                tags_html += f'<span class="tag {cls}">{s}</span>'
            tags_html += '</div>'
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown(
                f'<p class="skills-tally">{len(matched_skills)} matched &nbsp;/&nbsp; '
                f'{len(missing_skills)} missing out of {len(job_skills)} detected</p>',
                unsafe_allow_html=True
            )

        st.markdown(f'<div class="result-note">{note}</div>', unsafe_allow_html=True)

        # Debug panel
        if show_debug:
            st.markdown('<div class="debug-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="debug-label">Debug &mdash; Resume text (first 500 chars)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debug-box">{resume_text[:500]}</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="debug-label">Debug &mdash; Skills detected in resume</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debug-box">{sorted(resume_skills)}</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="debug-label">Debug &mdash; Skills detected in job description</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debug-box">{sorted(job_skills)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)