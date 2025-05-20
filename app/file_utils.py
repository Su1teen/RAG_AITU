import os, re
import numpy as np
import PyPDF2
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

# Извлечение текста (для duplicate-check)

def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        return open(filepath, encoding='utf-8', errors='ignore').read()
    if ext == '.docx':
        doc = DocxDocument(filepath)
        return '\n'.join(p.text for p in doc.paragraphs)
    if ext == '.pdf':
        text = ''
        reader = PyPDF2.PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    return ''

# cosine similarity

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any(): return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# нормализация

def normalize_text(text: str) -> str:
    t = text.lower().strip()
    return re.sub(r'\s+', ' ', t)

# поиск похожих

def find_similar_files(uploaded_text: str, folder: str, threshold: float = 0.7):
    model = SentenceTransformer('all-MiniLM-L12-v2')
    u_norm = normalize_text(uploaded_text)
    u_emb  = model.encode([u_norm], convert_to_numpy=True)[0]
    sims = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(('.txt','.docx','.pdf')): continue
        path = os.path.join(folder, fn)
        txt  = extract_text_from_file(path)
        ntxt = normalize_text(txt)
        emb  = model.encode([ntxt], convert_to_numpy=True)[0]
        sim  = cosine_similarity(u_emb, emb)
        if sim >= threshold:
            sims.append({'file': fn, 'similarity': round(sim*100, 2)})
    sims.sort(key=lambda x: x['similarity'], reverse=True)
    if any(s['similarity']==100 for s in sims):
        return [s for s in sims if s['similarity']==100]
    return sims[:3]