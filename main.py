import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pdfParser import cleanText, extractText
app = FastAPI()

tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

CHUNK_SIZE = 512

@app.post("/upload")

async def upload_file(file: UploadFile = File(...)):
        questions= []
        try:

            file_location = f"temp/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())

            extracted_text = extractText(file_location)
            cleaned_text = " ".join(cleanText(page) for page in extracted_text)
            print( cleaned_text )
            input_text = f"context: {cleaned_text}"

            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].squeeze(0)

            chunks = [input_ids[i : i + CHUNK_SIZE] for i in range(0, len(input_ids), CHUNK_SIZE)]

            for chunk in chunks:
                inputs = {"input_ids":chunk.unsqueeze(0)}


                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=64,
                    num_return_sequences=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=1, 
                    )

                for output in outputs:
                    question = tokenizer.decode(output, skip_special_tokens=True).replace('"\\"','""')
                    questions.append(question)
                
            return JSONResponse({"questions":questions})

        except Exception as e:
            return JSONResponse({"error":str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
