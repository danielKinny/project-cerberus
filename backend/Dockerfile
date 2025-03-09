FROM python:3.12.8-slim

WORKDIR /app

# Copy requirements first to leverage cache
COPY piprequirements.txt .

RUN pip install --no-cache-dir --upgrade -r piprequirements.txt

# Download NLTK data to a specific directory
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

# Copy only necessary files
COPY . .

# Set the NLTK_DATA environment variable
ENV NLTK_DATA=/usr/local/share/nltk_data

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]