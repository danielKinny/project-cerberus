from transformers import T5Tokenizer, T5ForConditionalGeneration

CHUNK_SIZE = 512
tokenizer = T5Tokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = T5ForConditionalGeneration.from_pretrained("iarfmoose/t5-base-question-generator")
questions=[]

with open(r"proof_of_concept\sample.txt", "r", encoding="utf-8") as f:
    context = f.read()

input_text = f"context: {context}"

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

    for idx, output in enumerate(outputs):
        question = tokenizer.decode(output, skip_special_tokens=True)
        questions.append(question)
        print(f"Question {idx + 1}: {question}")