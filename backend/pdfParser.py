import PyPDF2
import re
from fastapi.responses import JSONResponse

def extractText(pdf_file: str):
    with open(pdf_file, 'rb') as pdf:
        reader = PyPDF2.PdfReader(pdf, strict=False)
        pdf_text=[]

        if len(reader.pages) > 25:
            raise ValueError("PDF has more than 25 pages.")
            
        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)
        
        return pdf_text

def cleanText(text):
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted special characters but keep necessary ones
    text = re.sub(r"[^a-zA-Z0-9.,!?'/()\-\s]", '', text)

    return text.strip()