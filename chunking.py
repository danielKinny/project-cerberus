from pdfParser import cleanText, extractText
import re
import nltk

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

def chunk_text(text):
    sentences = sent_tokenize(text)
    chunk = ""
    chunks = []
    MAX_CHUNK_LENGTH = 512

    for sentence in sentences:
        if len(sentence) + len(chunk) >= MAX_CHUNK_LENGTH:
            chunks.append(chunk)
            print( len(chunk) )
            chunk = sentence
        else:
            chunk += " " + sentence

    return chunks