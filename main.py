import os
from utils import extract_text_from_pdf
from model import calculate_similarity

# Get the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full paths
resume_path = os.path.join(BASE_DIR, 'sample_data', 'sample_resume.pdf')
job_desc_path = os.path.join(BASE_DIR, 'sample_data', 'sample_job.txt')

# Load files
resume_text = extract_text_from_pdf(resume_path)
job_desc = open(job_desc_path, 'r', encoding='utf-8').read()

# Calculate similarity
score = calculate_similarity(resume_text, job_desc)
print(f"Match Score: {score}%")