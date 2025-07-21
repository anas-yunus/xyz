import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os

# 👇 Set Tesseract path (installed with pip or standalone)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\madhurpatil\AppData\Roaming\Python\Python313\Scripts\tesseract.exe"

# 👇 Set Poppler path (needed by pdf2image on Windows)
POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin"  # Change this to your actual Poppler bin path

def extract_text_from_pdf(pdf_path):
    """Extract text from normal (selectable) PDFs."""
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def extract_text_from_scanned_pdf(pdf_path):
    """OCR for scanned PDFs using pdf2image + pytesseract."""
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return "\n".join([pytesseract.image_to_string(img) for img in images])

def extract_text_from_image(image_path):
    """OCR for standalone image files."""
    return pytesseract.image_to_string(Image.open(image_path))

def smart_pdf_ocr(pdf_path):
    """Try native PDF extraction first, fallback to OCR if text is too short."""
    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) < 100:
        return extract_text_from_scanned_pdf(pdf_path)
    return text
