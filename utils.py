# -*- coding: utf-8 -*-
import pdfplumber
import re
import nltk
nltk.data.path.append(r"C:\Users\moazz\AppData\Roaming\nltk_data")
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try normal extraction first
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    text += page_text + " "
                else:
                    # Fallback: extract words individually
                    words = page.extract_words()
                    if words:
                        text += " ".join(w['text'] for w in words) + " "
    except Exception as e:
        print(f"[utils] PDF extraction error: {e}")
        return ""

    if not text.strip():
        print("[utils] WARNING: No text extracted from PDF.")
        return ""

    print(f"[utils] Extracted {len(text)} chars from PDF.")
    print(f"[utils] First 300 chars: {text[:300]}")
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)