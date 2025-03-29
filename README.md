hello! this is a web application that can convert pdf files into flashcards using NLP ( natural language processing )

**INTRODUCTION**:
The backend is primarily coded in python, using the FastAPI library to handle API endpoints, as of the last update to this readme file, the backend uses HuggingFaceAPI models to process the text, in particular, the question generator model is mrm8488/t5-base-finetuned-question-generation-ap, and the answer generation model is deepset/roberta-large-squad2.

The frontend is a simple design coded in basic html and css, the scripting is using JS to handle post and get requests and also add functionality to flip, edit and delete flashcards.

**HOW IT WORKS**:

pdfParser.py uses PyPDF2 to extract text directly from the pdf file provided, it also uses the regex library in python to slightly format the text in a way that can be understood by the question generation model later in main.py, it is imported into main.py and uses in the initial asynchronous function that handles the pdf.

chunking.py uses the NLTK library to accurately tokenize and split up text to ensure that sentences arent split in half, thus maintaining the integrity of the text provided, it is then intelligently chunked to ensure that each piece doesnt exceed the 512 token limit.

main.py initalises the question generating and answering models, uses chunking.py and pdfParser.py as utilities to recieve a list of chunks that make up the entire text, the list is iterated through and 3 questions are generated per chunk, as well as answers, only answers with a confidence score of 0.5 or higher are appended to the final questions array to ensure proper questions and answers. the array containing the questions and answers are then return as JSON object to the front end.

**HOW TO RUN IT**:

This app has a docker set-up, so it can be accessed if you have docker desktop, by just navigating to the directory and then using 
docker-compose up --build
in your terminal.

it will take around 5 minutes to properly start up due to the dependencies being downloaded, once it's properly initialised the frontend can be accessed through http://localhost:8080





