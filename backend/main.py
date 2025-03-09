import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from pdfParser import cleanText, extractText
from chunking import chunk_text
import torch
import logging
from typing import List, Tuple
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

question_generation_model = "mrm8488/t5-base-finetuned-question-generation-ap"
answering_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
tokenizer = AutoTokenizer.from_pretrained(question_generation_model)
model = AutoModelWithLMHead.from_pretrained(question_generation_model)

logging.basicConfig(level=logging.INFO)

def save_uploaded_file(file: UploadFile) -> str:
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    return file_location

def process_chunks(chunks: List[str]) -> List[Tuple[str, str]]:
    questions = []
    for i, chunk in enumerate(chunks):
        context_window = " ".join(chunks[max(0, i-1):min(i+2, len(chunks))])
        formatted_text = chunk.strip().replace('\n', ' ')
        chunk_tensor = tokenizer(f"answer:{formatted_text} context:{context_window}", max_length=512, truncation=True, padding=True, return_tensors="pt")

        outputs = model.generate(
            chunk_tensor['input_ids'],
            max_length=128,
            num_return_sequences=3,
            do_sample=True,
            top_k=30,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2
        )

        logging.info(f"Chunk {i+1} has been processed out of {len(chunks)}...")

        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            answer = answering_pipeline(
                question=question,
                context=context_window,
                max_answer_len=120,
                handle_impossible_answers=True,
                top_k=30,
            )

            for ans in answer:
                if ans['score'] > 0.5:
                    questions.append((question, ans['answer']))
    return questions

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    try:
        file_location = save_uploaded_file(file)
        extracted_text = extractText(file_location)
        if not extracted_text:
            raise ValueError("Failed to extract text from the uploaded file.")
        
        cleaned_text = " ".join(cleanText(page) for page in extracted_text)
        if not cleaned_text.strip():
            raise ValueError("The extracted text is empty after cleaning.")
        
        chunks = chunk_text(cleaned_text)
        questions = process_chunks(chunks)

        return JSONResponse({"questions": questions})

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    
    finally:
        # Remove the uploaded file after processing
        if os.path.exists(file_location):
            os.remove(file_location)
            logging.info(f"File {file_location} has been removed.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
