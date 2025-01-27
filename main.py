import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

tokenizer = T5Tokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = T5ForConditionalGeneration.from_pretrained("iarfmoose/t5-base-question-generator")

CHUNK_SIZE = 512

@app.post("/upload")

async def upload_file(file: UploadFile = File(...)):
        questions= []
        try:

            context = await file.read()
            text = context.decode("utf-8")
            input_text = f"context: {text}"

            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].squeeze(0)

            chunks = [input_ids[i : i + CHUNK_SIZE] for i in range(0, len(input_ids), CHUNK_SIZE)]

            for chunk in chunks:
                inputs = {"input_ids":chunk.unsqueeze(0)}


                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=64,
                    num_return_sequences=5,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7, 
                    )

                for output in outputs:
                    question = tokenizer.decode(output, skip_special_tokens=True)
                    questions.append(question)
                
            return JSONResponse({"questions":questions})

        except Exception as e:
            return JSONResponse({"error":str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
