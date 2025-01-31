hello! this is a web application that can convert pdf files into flashcards using NLP ( natural language processing )

**INTRODUCTION**:
The backend is primarily coded in python, using the FastAPI library to handle API endpoints, as of the last update to this readme file, the backend uses HuggingFaceAPI models to process the text, in particular, the question generator model is mrm8488/t5-base-finetuned-question-generation-ap, and the answer generation model is deepset/roberta-large-squad2.

The frontend is a simple design coded in basic html and css, the scripting is using JS to handle post and get requests and also add functionality to flip, edit and delete flashcards.

**HOW IT WORKS**:

pdfParser.py uses PyPDF2 to extract text directly from the pdf file provided, it also uses the regex library in python to slightly format the text in a way that can be understood by the question generation model later in main.py, it is imported into main.py and uses in the initial asynchronous function that handles the pdf.

chunking.py uses the NLTK library to accurately tokenize and split up text to ensure that sentences arent split in half, thus maintaining the integrity of the text provided, it is then intelligently chunked to ensure that each piece doesnt exceed the 512 token limit.

main.py initalises the question generating and answering models, uses chunking.py and pdfParser.py as utilities to recieve a list of chunks that make up the entire text, the list is iterated through and 3 questions are generated per chunk, as well as answers, only answers with a confidence score of 0.5 or higher are appended to the final questions array to ensure proper questions and answers. the array containing the questions and answers are then return as JSON object to the front end.

**HOW TO RUN IT**:

As of now, there is no file that can directly install the file onto a computer, thus it must be done manually.

To clone this repository, git must be installed onto your computer, and it can be cloned by typing git clone <url to this repo>

To run it, your computer must have python 3.12.8, and a corresponding pip module that is also up to date, any newer versions of python wont work because the torch module that is essential to the transformers library is not updated for later versions as of the writing of this document.

after the repo has been cloned or downloaded, it is recommended to first initialise a virtual environment to ensure that the there is no conflict between the pip modules required for the program and the already existing pip modules present on your computer.

to initialise a virtual environment, run these commands on your terminal

python -m venv venv 
or
python3 -m venv venv

this should create a .venv folder in your directory.

the actual VM can then be initialised using the command

**on windows**:
venv/Scripts/activate

**on macos**:
(source) venv/bin/activate

after the VM is initialised, run

pip install -r requirements.txt
or
pip3 install -r requirements.txt

this will install the required modules for the program to run.
this program is a locally hosted program, so it is needed to manually initialise both the backend and the frontend

open two terminal windows.
in both terminal windows, type in:
cd project-cerberus
this will locate to the project's directory.

in one window, type in:
cd backend
uvicorn main:app --reload

in the second window, type in:
cd frontend
python3 -m http.server 8080
or python -m http.server 8080


congrats! you have locally hosted the web app. make sure that the backend terminal says that the application startup is complete, and go over to localhost:8080/ and try out the app.

-danny





