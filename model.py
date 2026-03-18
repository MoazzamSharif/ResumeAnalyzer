from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

def calculate_similarity(resume_text, job_text):
    documents = [clean_text(resume_text), clean_text(job_text)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0]*100, 2)   # return % match