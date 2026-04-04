# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text
import re

# ---------------------------------------------------------------------------
# Domain-agnostic skill / keyword patterns
# ---------------------------------------------------------------------------

# Hard skills: broad across tech, finance, healthcare, marketing, legal, etc.
HARD_SKILL_PATTERN = re.compile(
    r'\b('
    # --- Tech / Software ---
    r'python|java|javascript|typescript|ruby|golang|rust|swift|kotlin|scala|php|matlab|bash|sql|nosql|'
    r'react|angular|vue|django|flask|fastapi|spring|nodejs|express|tensorflow|pytorch|keras|'
    r'scikit|sklearn|pandas|numpy|scipy|xgboost|lightgbm|'
    r'aws|azure|gcp|docker|kubernetes|terraform|jenkins|devops|mlops|airflow|mlflow|kubeflow|sagemaker|'
    r'spark|hadoop|tableau|powerbi|etl|'
    r'postgresql|postgres|mysql|mongodb|redis|elasticsearch|cassandra|sqlite|oracle|'
    r'bert|transformers|spacy|nltk|'
    r'api|rest|graphql|microservices|git|linux|cloud|backend|frontend|fullstack|'
    r'regression|classification|clustering|ensemble|'
    r'nlp|machinelearning|deeplearning|datascience|computervision|'
    # --- Finance / Accounting ---
    r'accounting|bookkeeping|auditing|taxation|budgeting|forecasting|valuation|'
    r'ifrs|gaap|cpa|cfa|cma|acca|'
    r'excel|quickbooks|sap|oracle|xero|'
    r'equity|derivatives|portfolio|underwriting|actuarial|'
    r'reconciliation|payroll|accounts|procurement|'
    # --- Marketing / Sales ---
    r'seo|sem|ppc|cro|analytics|'
    r'branding|copywriting|advertising|pr|'
    r'crm|salesforce|hubspot|marketo|'
    r'socialmedia|contentmarketing|emailmarketing|'
    r'b2b|b2c|gtm|leadgeneration|'
    # --- Healthcare / Medical ---
    r'nursing|pharmacy|radiology|diagnosis|surgery|'
    r'ehr|emr|hipaa|cpt|icd|'
    r'clinical|patient|healthcare|medical|pharmaceutical|'
    r'epidemiology|pathology|oncology|pediatrics|'
    # --- Legal ---
    r'litigation|arbitration|compliance|regulatory|'
    r'contracts|drafting|negotiation|ip|'
    r'paralegal|legalresearch|discovery|'
    # --- HR / Management ---
    r'recruiting|onboarding|talentacquisition|'
    r'performancemanagement|compensation|benefits|'
    r'hris|workday|successfactors|'
    r'training|coaching|mentoring|'
    # --- Engineering / Manufacturing ---
    r'autocad|solidworks|catia|matlab|simulink|'
    r'manufacturing|production|qualitycontrol|leanmanufacturing|'
    r'sixsigma|kaizen|iso|'
    r'electrical|mechanical|civil|structural|'
    r'cad|cam|plc|scada|'
    # --- Education / Research ---
    r'curriculum|pedagogy|assessment|elearning|'
    r'research|analysis|statistics|spss|stata|'
    r'writing|editing|publishing|'
    # --- Soft skills (universal) ---
    r'leadership|management|agile|scrum|communication|collaboration|'
    r'analytical|problemsolving|critical|creative|'
    r'presentation|reporting|stakeholder|'
    r')\b',
    re.IGNORECASE
)

# Multi-word phrases - domain-agnostic
PHRASE_PATTERN = re.compile(
    r'machine\s+learning|deep\s+learning|data\s+science|computer\s+vision|'
    r'natural\s+language|power\s+bi|scikit[\-\s]learn|ci[\-/]?cd|hugging\s*face|'
    r'project\s+management|product\s+management|product\s+development|'
    r'business\s+development|business\s+analysis|business\s+intelligence|'
    r'financial\s+modeling|financial\s+analysis|financial\s+reporting|'
    r'supply\s+chain|risk\s+management|change\s+management|'
    r'public\s+health|mental\s+health|social\s+work|'
    r'graphic\s+design|ux[\s/]?ui|user\s+experience|user\s+research|'
    r'content\s+strategy|digital\s+marketing|growth\s+hacking|'
    r'customer\s+success|customer\s+service|client\s+relations|'
    r'data\s+analysis|data\s+visualization|data\s+engineering|'
    r'software\s+engineering|software\s+development|systems\s+design|'
    r'quality\s+assurance|technical\s+writing|grant\s+writing|'
    r'lean\s+manufacturing|six\s+sigma|process\s+improvement|'
    r'human\s+resources|talent\s+acquisition|performance\s+management|'
    r'mergers\s+(?:and|&)\s+acquisitions|venture\s+capital|private\s+equity',
    re.IGNORECASE
)

STOPWORDS = {
    'the','and','for','are','you','will','with','this','that','have','from',
    'our','your','we','be','to','of','in','a','an','on','at','by','as','is',
    'it','or','not','but','was','all','they','their','who','what','how',
    'also','can','into','over','more','about','than','which','has','its',
    'been','some','may','would','should','each','both','through','during',
    'well','per','etc','strong','proven','excellent','good','key','new',
    'within','across','using','based','including','such','other','these',
    'work','role','team','join','help','make','build','use','used','like',
    'experience','years','ability','skills','knowledge','understanding',
    'looking','seeking','required','preferred','must','nice','plus',
    'great','high','level','highly','year','day','time','able','want',
    'need','get','give','take','keep','let','set','see','come','go',
    'many','very','just','even','back','here','there','then','than',
    'when','where','while','after','before','between','under','without',
}

def normalize(s):
    return re.sub(r'[\s\-_/\.\+#]+', '', s.lower())

def extract_skill_keywords(text):
    found = set()
    for m in PHRASE_PATTERN.finditer(text):
        found.add(normalize(m.group()))
    for m in HARD_SKILL_PATTERN.finditer(text):
        found.add(normalize(m.group()))
    return found

def extract_meaningful_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = set(text.split())
    return {w for w in words if len(w) > 3 and w not in STOPWORDS and not w.isdigit()}

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def f1_overlap(set_a, set_b):
    """Balanced F1 between two keyword sets."""
    if not set_a or not set_b:
        return 0.0
    matched   = set_a & set_b
    precision = len(matched) / len(set_a)
    recall    = len(matched) / len(set_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def jaccard_overlap(set_a, set_b):
    """Jaccard similarity - punishes large mismatches less harshly than F1."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def tfidf_score(resume_text, job_text):
    cleaned_resume = clean_text(resume_text)
    cleaned_job    = clean_text(job_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])
        score  = cosine_similarity(matrix[0:1], matrix[1:2])
        return float(score[0][0])
    except Exception:
        return 0.0

def _stretch(x, low=0.0, high=0.72):
    """
    Linearly stretch a raw combined score (typically 0.05-0.60) to 0-100.
    Clamps outside the range so extreme values don't go negative or over 100.
    """
    x = max(low, min(high, x))
    return (x - low) / (high - low) * 100

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_similarity(resume_text, job_text):
    """
    Returns a score 0-100 that reflects how well the resume matches the job.

    Breakdown:
      40% skill keyword F1  (domain-agnostic regex patterns)
      25% skill keyword Jaccard (softer overlap)
      20% meaningful keyword F1 (general vocabulary overlap)
      15% TF-IDF cosine on cleaned text (semantic proximity)

    The raw weighted sum is then linearly stretched so that
    a realistic "strong" match lands around 70-85 and a
    poor match lands around 5-20.
    """
    resume_skills = extract_skill_keywords(resume_text)
    job_skills    = extract_skill_keywords(job_text)
    resume_kw     = extract_meaningful_keywords(resume_text)
    job_kw        = extract_meaningful_keywords(job_text)
    tf            = tfidf_score(resume_text, job_text)

    skill_f1  = f1_overlap(resume_skills, job_skills)    if job_skills else 0.0
    skill_jac = jaccard_overlap(resume_skills, job_skills) if job_skills else 0.0
    kw_f1     = f1_overlap(resume_kw, job_kw)

    # Weighted raw score (all components are 0-1)
    if job_skills:
        raw = (skill_f1  * 0.40 +
               skill_jac * 0.25 +
               kw_f1     * 0.20 +
               tf        * 0.15)
    else:
        # No skills detected in JD - fall back to keyword + TF-IDF only
        raw = (kw_f1 * 0.70 +
               tf    * 0.30)

    stretched = _stretch(raw, low=0.0, high=0.72)
    final = 10 + (stretched * 0.90)
    return round(min(max(final, 0), 100), 2)