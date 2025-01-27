from transformers import pipeline
import re

summary = pipeline('summarization')

def cleanText(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    return cleaned_text

with open("proof_of_concept\\sample.txt","r", encoding="utf-8") as f:
    text = f.read().strip()

text = text.replace('.','.<eos>')
text = text.replace('!','!<eos>')
text = text.replace('?', '?<eos>').strip()
sentences = text.split('<eos')

maxChunk = 500
currentChunk = 0
chunks = []

for sentence in sentences:
    if len(chunks) == currentChunk+1:
        if len( chunks[currentChunk]) + len(sentence.split(' ')) <= maxChunk:
            chunks[currentChunk].extend(sentence.split(' '))
        else:
            currentChunk += 1
            chunks.append(sentence.split(' '))
    else:
        print(currentChunk)
        chunks.append(sentence.split(' '))

for chunk_id in range( len(chunks)):
    chunks[chunk_id] = ' '.join(chunks[chunk_id] )

res = summary(chunks, max_length=120, min_length=30, do_sample=False )

entireSummary = cleanText( ' '.join([summ['summary_text'] for summ in res]) )

print(entireSummary)