import sys

try:
    import fitz  # PyMuPDF
    doc = fitz.open("Dominate or Delete Decentralized Competing Bandits in Serial.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    with open("pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("PDF extracted successfully.")
except ImportError:
    try:
        import PyPDF2
        with open("Dominate or Delete Decentralized Competing Bandits in Serial.pdf", "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        with open("pdf_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("PDF extracted successfully using PyPDF2.")
    except ImportError:
        print("No PyMuPDF or PyPDF2 found.")
