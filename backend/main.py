import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
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

question_generation_model = "mrm8488/t5-base-finetuned-question-generation-ap"
answering_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
tokenizer = AutoTokenizer.from_pretrained(question_generation_model)
model = AutoModelWithLMHead.from_pretrained(question_generation_model)

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

                context_window = " ".join(chunks[ max(0, i-1) : min(i+2, len(chunks) ) ] )
                formatted_text = chunk.strip().replace('\n',' ')
                chunk_tensor = tokenizer(f"answer:{formatted_text} context:{context_window}", max_length=512, truncation=True, padding=True, return_tensors="pt")

                outputs = model.generate(
                    chunk_tensor['input_ids'],
                    max_length=128,
                    num_return_sequences=3,
                    do_sample=True,
                    top_k=30,
                    top_p=0.95,
                    temperature=0.7,
                    no_repeat_ngram_size = 2
                    )
                
                print(f"Chunk {i+1} has been processed out of {len(chunks) }...")

                for output in outputs:

                    question = tokenizer.decode(output, skip_special_tokens=True)
                    
                    answer = answering_pipeline(
                        question=question,
                        context=context_window,
                        max_answer_len=120,
                        handle_impossible_answers = True,
                        top_k=30,
                        )
                    
                    for ans in answer:
                        if ans['score'] > 0.5:
                            questions.append([question, ans['answer']])
                    
            return JSONResponse({"questions":questions})

        except Exception as e:
            print(e)
            return JSONResponse({"error":str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
