from pdfParser import cleanText, extractText
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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