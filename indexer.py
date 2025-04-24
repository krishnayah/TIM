import os
import json
import hashlib
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Configuration
DATABASE_FILE = 'data.json'
MODEL_NAME = 'all-MiniLM-L6-v2'
THRESHOLD = 0.2 # sentence similarity threshold for segmenting

# Initialize the sentence-transformers model
model = SentenceTransformer(MODEL_NAME)


def get_embeddings(file_path, force_regenerate=False):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    # Hash the file to detect changes
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # Load existing database or initialize
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Return cached segments if available
    if file_hash in data and not force_regenerate:
        return data[file_hash]['segments']

    # Otherwise, extract text and segment
    text = get_text_from_pdf(file_path)
    segments = segment_and_generate_embeddings(text, THRESHOLD)

    # Save to database
    data[file_hash] = {
        'file_path': file_path,
        'file_type': 'pdf',
        'hash': file_hash,
        'text': text,
        'segments': segments
    }
    with open(DATABASE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    return segments


def get_text_from_pdf(file_path):
    # This gets raw text from a pdf
    raw_pages = []
    with open(file_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            raw_pages.append(page.extract_text() or "")

    # Combine and clean paragraphs
    raw = "\n\n".join(raw_pages)
    paragraphs = raw.split("\n\n")
    cleaned = [" ".join(p.splitlines()) for p in paragraphs]
    return "\n\n".join(cleaned)


def segment_and_generate_embeddings(text, threshold):
    # Segments text into semantically similar chunks.
    segments = []
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

    for sentence in sentences:
        embedding = model.encode(sentence, convert_to_numpy=True)
        if segments:
            prev_emb = np.array(segments[-1]['embedding'], dtype=np.float32)
            emb1 = prev_emb.reshape(1, -1)
            emb2 = np.array(embedding, dtype=np.float32).reshape(1, -1)
            sim = util.cos_sim(emb1, emb2).item()
            if sim > threshold:
                segments[-1]['text'] += ' ' + sentence
                merged_emb = model.encode(segments[-1]['text'], convert_to_numpy=True)
                segments[-1]['embedding'] = merged_emb.tolist()
            else:
                segments.append({'text': sentence, 'embedding': embedding.tolist()})
        else:
            segments.append({'text': sentence, 'embedding': embedding.tolist()})

    return segments


def get_relevant_segments(query):

    rankings = {}

    query_emb = model.encode(query, convert_to_tensor=True).float().cpu()
    query_emb = query_emb / query_emb.norm(p=2)

    # Load database
    if not os.path.exists(DATABASE_FILE):
        return rankings
    with open(DATABASE_FILE, 'r') as f:
        data = json.load(f)

    # Iterate through files and segments
    for file_hash, info in data.items():
        file_name = info.get('file_path', '')
        for seg in info.get('segments', []):
            emb = torch.tensor(seg['embedding'], dtype=torch.float32).unsqueeze(0)

            emb = emb.to(query_emb.device)
            emb = emb / emb.norm()
            # Calculate dot similarity
            score = util.dot_score(query_emb, emb)[0][0].item()
            rankings[seg['text']] = {
                'similarity': score,
                'file_name': file_name
            }

    return rankings


if __name__ == '__main__':
    file_path = 'example.pdf'
    get_embeddings(file_path)
    query = 'Your search query here'
    rankings = get_relevant_segments(query)
    for text, meta in rankings.items():
        print(f"{text} -> Score: {meta['similarity']:.4f}, File: {meta['file_name']}")
