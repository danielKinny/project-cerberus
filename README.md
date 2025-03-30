# PDF to Flashcards Web Application

## Introduction
This is a web application that converts PDF files into flashcards using Natural Language Processing (NLP).

### Backend
The backend is primarily coded in Python using the FastAPI library to handle API endpoints. It utilizes Hugging Face API models to process text. As of the last update to this README file, the following models are used:

- **Question Generation Model**: `mrm8488/t5-base-finetuned-question-generation-ap`
- **Answer Generation Model**: `deepset/roberta-large-squad2`

### Frontend
The frontend is built with basic HTML and CSS, with JavaScript handling:
- POST and GET requests
- Functionality for flipping, editing, and deleting flashcards

## How It Works

### `pdfParser.py`
- Uses `PyPDF2` to extract text directly from a provided PDF file.
- Uses Python's `regex` library to format text for better processing by the question generation model.
- Imported into `main.py` and utilized in the initial asynchronous function handling PDF input.

### `chunking.py`
- Uses the `NLTK` library to accurately tokenize and split text to prevent sentence fragmentation.
- Intelligently chunks text to stay within the **512-token limit** of the NLP models.

### `main.py`
- Initializes the question and answer generation models.
- Uses `chunking.py` and `pdfParser.py` to process the extracted text.
- Iterates through the list of text chunks and generates **three questions per chunk**, along with answers.
- Only answers with a **confidence score of 0.5 or higher** are included in the final output.
- Returns a JSON object containing the generated questions and answers to the frontend.

## How to Run It

### Using Docker
This application has a Docker setup, making it easy to run with **Docker Desktop**. Follow these steps:

1. Navigate to the project directory.
2. Run the following command in the terminal:
   ```sh
   docker-compose up --build
   ```
3. The setup will take around **5 minutes** to download dependencies and initialize.
4. Once properly initialized, access the frontend via:
   ```
   http://localhost:8080
   ```

