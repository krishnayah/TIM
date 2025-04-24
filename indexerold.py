import os
import json
from openai import APIError, OpenAI
import hashlib
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity




database_file = 'data.json'
api_key = ""

client = OpenAI(api_key=api_key)

# file schema:
# {
#     "file_path": {
#         "file_type": "pdf",
#         "hash": "hash_of_file",
#         "text": "extracted_text",
#         "embedding": {},
#         "segments": [
#             {
#                 "text": "text_segment",
#                 "embedding": {},
#                 },
#             {
#                 "text": "text_segment",
#                 "embedding": {},
#             },
#             ...
#
#     }



def get_embeddings(file_path, force_regenerate=False):
    # check if file exists, hash file, if it exists in the database, return the embeddings
    # if it doesn't exist, read the file, extract text, get embeddings, and save to the database
    # check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    # hash the file
    file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    # check if file is in the database, if so, return its segments & embeddings
    if os.path.exists(database_file) :
        with open(database_file, "r") as f:
            data = json.load(f)
            if file_hash in data and not force_regenerate:
                return data[file_hash]["segments"]
            else:
                # Student text from the file
                text = get_text_from_pdf(file_path)
                segments = segment_and_generate_embeddings(text, 0.50, regenerate_embeddings=True)
                # save the file to the database
                data[file_hash] = {
                    "file_path": file_path,
                    "file_type": "pdf",
                    "hash": file_hash,
                    "text": text,
                    "embedding": {},
                    "segments": segments
                }
                with open(database_file, "w") as f:
                    json.dump(data, f)
                return segments



def get_text_from_pdf(file_path):
    raw_pages = []
    with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text() or ""
            raw_pages.append(text)
    # join pages with a blank line
    raw = "\n\n".join(raw_pages)
    # split on paragraph separators

    cleaned_pars = [" ".join(p.splitlines()) for p in paragraphs]
    clean = "".join(cleaned_pars)
    return clean


def get_text_from_image_pdf(file_path):
    pass

def get_embeddings_from_text(text):
    pass


def segment_and_generate_embeddings(text, threshold, regenerate_embeddings=True):
    # use OpenAI to segment text
    segments = []
    #{"text": "text goes here", embeddings: "embeddings go here"}
    # read text sentence by sentence, either by a period delimiter, whichever comes first. two words sentences or less are not counted
    # as segments and are included in the previous segment
    # use OpenAI to get embeddings for a buffer, and as the buffer is built up, check to see if the sentence has
    # a certain number of similarity to the current buffer. if so, add it, else, push the buffer to the segments list
    # and start a new buffer
    total_sentences = len(text.split("."))
    cur = 0

    for sentence in text.split("."):
        cur += 1

        print(f"Processing sentence {cur} of {total_sentences}")
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        # get embedding for the sentence
        try:
            embedding = client.embeddings.create(input=sentence, model="text-embedding-3-large").data[0].embedding

        except APIError as e:
            print(f"API error: {e}")
            continue
        # check similarity to the current buffer
        if len(segments) > 0:
            similarity = cosine_similarity([segments[-1]['embedding']], [embedding])[0][0]
            print(f"Similarity: {similarity}")
            if similarity > threshold:
                segments[-1]['text'] += " " + sentence
                # instead of setting the new embedding to just the latest sentence, regenerate embedding for all of the text
                # in the segment

                if regenerate_embeddings:
                    segments[-1]['embedding'] = client.embeddings.create(input=segments[-1]['text'], model="text-embedding-3-large").data[0].embedding
                else:
                    segments[-1]['embedding'] = embedding
            else:
                segments.append({'text': sentence, 'embedding': embedding})
        else:
            segments.append({'text': sentence, 'embedding': embedding})


    return segments




# get the embeddings, and then find the most similar segments to the query
def get_relevant_segments(query):
    # get the embeddings for the query
    rankings = {}
    query_embedding = client.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
    # load the database
    if os.path.exists(database_file):
        with open(database_file, "r") as f:
            data = json.load(f)
            # iterate through the files in the database
            for file_hash in data:
                file_name = data[file_hash]["file_path"]
                segments = data[file_hash]["segments"]
                # iterate through the segments and find the most similar one to the query
                for segment in segments:
                    similarity = cosine_similarity([segment['embedding']], [query_embedding])[0][0]
                    # add similarity and filename to rankings
                    rankings[segment['text']] = {"similarity": similarity, "file_name": file_name}
    return rankings


