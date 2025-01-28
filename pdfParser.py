import PyPDF2
import re

def extractText(pdf_file: str):
    with open(pdf_file, 'rb') as pdf:
        reader = PyPDF2.PdfReader(pdf, strict=False)
        pdf_text=[]

        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)

    return pdf_text

def cleanText(text):
    text = text.encode( 'utf-8', 'ignore').decode('utf-8')
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

if __name__ == '__main__':
    extractedText = extractText('concepts\\Hassett - Summary.pdf')
    
    with open("concepts//sample.txt","a") as s:
        
        for text in extractedText:
            s.write(cleanText(text) +" ")