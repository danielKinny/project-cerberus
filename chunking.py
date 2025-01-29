from pdfParser import cleanText, extractText
import re

def chunk_text(text):
    split_text = text.split(".")
    chunk = ""
    chunks = []
    MAX_CHUNK_LENGTH = 512

    for sentence in split_text:
        if len(sentence) + len(chunk) >= MAX_CHUNK_LENGTH:
            chunks.append(chunk)
            print( len(chunk) )
            chunk = sentence
        else:
            chunk += sentence

    return chunks