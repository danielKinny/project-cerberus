import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from pdfParser import cleanText, extractText
from chunking import chunk_text
import torch

app = FastAPI()

app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

question_generation_model = "valhalla/t5-base-qg-hl"
answering_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
tokenizer = T5Tokenizer.from_pretrained(question_generation_model)
model = T5ForConditionalGeneration.from_pretrained(question_generation_model)

@app.post("/upload")

async def upload_file(file: UploadFile = File(...)):
        questions= []

        try:

            file_location = f"temp/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())

            extracted_text = extractText(file_location)
            cleaned_text = " ".join(cleanText(page) for page in extracted_text)
            chunks = chunk_text(cleaned_text)     

            for i,chunk in enumerate(chunks):
                chunk_tensor = tokenizer(f"context:{chunk}", max_length=512, truncation=True, padding=True, return_tensors="pt")

                outputs = model.generate(
                    chunk_tensor['input_ids'],
                    max_length=64,
                    num_return_sequences=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=1, 
                    )
                
                print(f"Chunk {i} has been processed out of {len(chunks) }...")

                for output in outputs:
                    question = tokenizer.decode(output, skip_special_tokens=True).replace('"\\"','""')
                    answer = answering_pipeline(question=question, context=chunk)
                    questions.append([question, answer['answer']])
                
            return JSONResponse({"questions":questions})

        except Exception as e:
            print(e)
            return JSONResponse({"error":str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
