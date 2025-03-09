from pdfParser import cleanText, extractText
import nltk
import os

# Set the NLTK data path to the directory used during the Docker build process
nltk.data.path.append(os.getenv('NLTK_DATA', '/usr/local/share/nltk_data'))

from nltk.tokenize import sent_tokenize

def chunk_text(text):
    sentences = sent_tokenize(text)
    chunk = ""
    chunks = []
    MAX_CHUNK_LENGTH = 512

    for sentence in sentences:
        if len(sentence) + len(chunk) >= MAX_CHUNK_LENGTH:
            chunks.append(chunk)
            chunk = sentence
        else:
            chunk += " " + sentence

    return chunks