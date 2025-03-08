FROM python:3.12.8-slim

WORKDIR /app

# Copy only necessary files
COPY . .

CMD ["python", "-m", "http.server", "8080"]