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
@import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --dark-green: #2d7d5e;
    --maroon:     #c94c5e;
    --dark-bg:    #0f1419;
    --surface:    #1a2027;
    --surface2:   #252d36;
    --border:     #3a444f;
    --border2:    #4a5560;
    --text:       #e8ecf1;
    --text-light: #a8b3bf;
    --accent-green: #3fa370;
    --accent-maroon: #e85570;
    --radius:     8px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1419 0%, #1a2230 100%) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
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
    background: linear-gradient(135deg, var(--dark-green) 0%, #1f5940 100%);
    border-radius: 12px;
    padding: 3.5rem 2.5rem;
    margin-bottom: 3.5rem;
    box-shadow: 0 12px 48px rgba(45, 125, 94, 0.25);
}
.site-kicker {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
    margin-bottom: 0.75rem;
    font-weight: 600;
}
.site-title {
    font-family: 'Merriweather', serif !important;
    font-size: clamp(2.2rem, 5vw, 3.2rem) !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    color: #ffffff !important;
    margin: 0 0 0.75rem !important;
    letter-spacing: -0.01em;
}
.site-desc {
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.85);
    line-height: 1.6;
    font-weight: 300;
}

/* Labels */
.field-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 0.65rem;
    display: block;
    font-weight: 600;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover { 
    border-color: var(--dark-green) !important;
    box-shadow: 0 4px 16px rgba(45, 125, 94, 0.25) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--text-light) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
}
[data-testid="stBaseButton-secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-light) !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--dark-green) !important;
    color: var(--dark-green) !important;
}

/* Textarea */
[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 2px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    transition: all 0.3s ease !important;
    caret-color: var(--accent-green);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--dark-green) !important;
    box-shadow: 0 0 0 3px rgba(45, 125, 94, 0.2) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #aaa !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* Button */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--dark-green) 0%, var(--accent-green) 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 1rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 8px 24px rgba(45, 125, 94, 0.3) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, var(--accent-green) 0%, #2d7d5e 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 36px rgba(45, 125, 94, 0.4) !important;
}

/* Result */
.result-wrap {
    margin-top: 3rem;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem;
    box-shadow: 0 8px 32px rgba(45, 125, 94, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.result-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 1.5rem;
    font-weight: 600;
}
.score-row {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.score-big {
    font-family: 'Merriweather', serif;
    font-size: 5.5rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(135deg, var(--dark-green), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.score-denom {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    color: var(--text-light);
    font-weight: 500;
}
.score-verdict {
    font-family: 'Merriweather', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-maroon);
    margin-bottom: 2rem;
}
.bar-track {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    margin-bottom: 2.5rem;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green) 0%, var(--dark-green) 100%);
    animation: slideIn 1.2s ease-out forwards;
    box-shadow: 0 0 12px rgba(63, 163, 112, 0.4);
}
@keyframes slideIn {
    from { width: 0; opacity: 0; }
    to { width: 100%; opacity: 1; }
}

/* Skills */
.skills-heading {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-light);
    margin: 2.5rem 0 1.25rem;
    border-top: 1px solid var(--border);
    padding-top: 2rem;
    font-weight: 600;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.tag {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.02em;
    font-weight: 500;
    transition: all 0.2s ease;
}
.tag-hit  { 
    background: rgba(63, 163, 112, 0.2); 
    color: var(--accent-green); 
    border: 1px solid var(--accent-green);
}
.tag-hit:hover {
    background: rgba(63, 163, 112, 0.3);
    box-shadow: 0 0 12px rgba(63, 163, 112, 0.2);
}
.tag-miss { 
    background: var(--border2); 
    color: var(--text-light); 
    border: 1px solid var(--border);
}
.tag-miss:hover {
    background: rgba(232, 85, 112, 0.15);
    border-color: var(--accent-maroon);
    color: var(--accent-maroon);
}
.skills-tally {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-light);
    margin-top: 0.75rem;
    font-weight: 500;
}

/* Note */
.result-note {
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.8;
    margin-top: 2rem;
    padding: 1.5rem 1.75rem;
    background: rgba(45, 125, 94, 0.12);
    border-left: 4px solid var(--dark-green);
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* Debug */
.debug-wrap {
    margin-top: 2.5rem;
    border-top: 1px solid var(--border);
    padding-top: 2rem;
}
.debug-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 0.75rem;
    font-weight: 600;
}
.debug-box {
    background: var(--border);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    color: var(--accent-green);
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(201, 76, 94, 0.15) !important;
    border: 1px solid rgba(232, 85, 112, 0.4) !important;
    border-radius: var(--radius) !important;
    color: var(--accent-maroon) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stSpinner"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
    font-weight: 500 !important;
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
