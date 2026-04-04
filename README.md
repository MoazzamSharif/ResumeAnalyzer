# Resume Analyzer

A local, privacy-first resume scoring tool built with Streamlit. Upload a resume PDF and paste a job description — get an instant match score with skill gap analysis.

---

## Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-link.streamlit.app](https://resumeanalyzer-8uxwzkbwef73ymexjzqpob.streamlit.app/))

> Upload your resume → Paste the JD → Click Analyze

![Resume Analyzer Screenshot](screenshot.png)

The app scores your resume out of 100 and shows which skills from the job description you have and which you're missing.

---

## Features

- **Domain-agnostic scoring** — works for tech, marketing, finance, healthcare, legal, and more
- **Skill gap breakdown** — see exactly which keywords you're missing from the JD
- **No data sent anywhere** — runs fully offline on your machine
- **Clean UI** — dark editorial interface built with Streamlit

---

## How It Works

Scoring uses a weighted combination of four signals:

| Signal | Weight | What it measures |
|---|---|---|
| Skill keyword F1 | 40% | Precision + recall of matched skill terms |
| Skill keyword Jaccard | 25% | Softer overlap that handles large skill sets |
| Meaningful keyword F1 | 20% | General vocabulary overlap |
| TF-IDF cosine similarity | 15% | Overall semantic proximity of both texts |

The raw score is then linearly stretched to a 0–100 scale with a base floor so no resume scores below 10.

**Verdict bands:**
- 65 and above → Strong match
- 35–64 → Partial match
- Below 35 → Weak match

---

## Project Structure

```
ResumeAnalyzer/
├── app.py          # Streamlit UI
├── model.py        # Scoring logic
├── utils.py        # PDF extraction and text cleaning
├── sample_data/
│   ├── sample_resume.pdf
│   └── sample_job.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/your-username/ResumeAnalyzer.git
cd ResumeAnalyzer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run the app
streamlit run app.py
```

---

## Requirements

Create a `requirements.txt` with:

```
streamlit
pdfplumber
scikit-learn
nltk
```

---

## Limitations

- Scoring is keyword and TF-IDF based — it does not understand context the way an LLM would
- Very short resumes (under 150 words) may produce less reliable scores
- Scanned PDFs (image-based) will extract no text — use a text-layer PDF

---

## Built With

- [Streamlit](https://streamlit.io) — UI framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction
- [scikit-learn](https://scikit-learn.org) — TF-IDF and cosine similarity
- [NLTK](https://www.nltk.org) — Text tokenization and stopword removal

---

## Author

Built by [@moazz](https://github.com/moazz)
