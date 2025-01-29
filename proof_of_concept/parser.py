import PyPDF2
import re

def extractText(pdf_file: str):
    extracted_text = []
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        extracted_text.append(page.extract_text())
    return " ".join(extracted_text)

def cleanText(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    
    # Preserve paragraph breaks and clean excessive newlines
    text = re.sub(r'(\s*\n\s*){2,}', '\n\n', text)
    
    # Fix spacing issues around punctuation (e.g., 'i.e.' instead of 'i. e.')
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    # Handle hyphenation across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Keep parentheses intact and fix spacing around them
    text = re.sub(r'(\()([^\s])', r'\1 \2', text)
    text = re.sub(r'([^\s])(\))', r'\1 \2', text)
    
    # Remove unwanted special characters but retain basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"()-]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    with open("testing.txt","w") as f:
        f.write(text)

    return text

if __name__ == "__main__":
    print(cleanText(extractText("proof_of_concept\test_pdf.pdf")))