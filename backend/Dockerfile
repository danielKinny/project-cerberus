FROM python:3.12.8-slim

WORKDIR /app

# Copy requirements first to leverage cache
COPY piprequirements.txt .
RUN pip install --no-cache-dir -r piprequirements.txt

# Copy only necessary files
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]