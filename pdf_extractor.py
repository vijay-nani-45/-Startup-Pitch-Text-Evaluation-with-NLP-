# pdf_extractor.py - Enhanced with OCR and diagnostics
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import re
import io
import os
from typing import Dict, List, Tuple
import pandas as pd
from config import Config

class EnhancedPDFExtractor:
    def __init__(self):
        self.config = Config()
        self.section_keywords = {
            'problem': ['problem', 'challenge', 'pain point', 'issue', 'difficulty'],
            'solution': ['solution', 'product', 'service', 'approach', 'technology'],
            'market': ['market', 'tam', 'sam', 'som', 'addressable market', 'opportunity'],
            'traction': ['traction', 'growth', 'revenue', 'users', 'customers', 'metrics', 'kpi'],
            'team': ['team', 'founder', 'co-founder', 'ceo', 'cto', 'experience', 'background'],
            'business_model': ['business model', 'revenue model', 'monetization', 'pricing', 'subscription'],
            'ask': ['funding', 'investment', 'raise', 'capital', 'ask', 'use of funds', 'valuation']
        }
    
    def diagnose_pdf(self, pdf_path: str) -> Dict:
        """Diagnose PDF structure and content"""
        try:
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            
            if file_size_mb > self.config.MAX_PDF_SIZE_MB:
                return {
                    'status': 'warning',
                    'message': f'Large file ({file_size_mb:.1f}MB) - may be slow',
                    'can_process': True
                }
            
            doc = fitz.open(pdf_path)
            info = {
                'filename': os.path.basename(pdf_path),
                'pages': doc.page_count,
                'encrypted': doc.is_encrypted,
                'file_size_mb': file_size_mb,
                'has_text': False,
                'has_images': False,
                'text_sample': '',
                'status': 'ok',
                'can_process': True
            }
            
            if doc.is_encrypted:
                doc.close()
                return {
                    'status': 'error',
                    'message': 'PDF is password protected',
                    'can_process': False
                }
            
            # Check content in first few pages
            for page_num in range(min(3, doc.page_count)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                if text and len(text) > 50:
                    info['has_text'] = True
                    info['text_sample'] = text[:200] + "..."
                    break
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    info['has_images'] = True
            
            doc.close()
            
            if not info['has_text'] and not info['has_images']:
                info['status'] = 'warning'
                info['message'] = 'No readable content detected'
            
            return info
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Cannot open PDF: {str(e)}',
                'can_process': False
            }
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PyMuPDF extraction error: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")
            return ""
    
    # def extract_text_ocr(self, pdf_path: str) -> str:
    #     """Extract text using OCR for image-based PDFs"""
    #     if not self.config.OCR_ENABLED:
    #         return ""
        
    #     try:
    #         doc = fitz.open(pdf_path)
    #         text = ""
            
    #         print(f"   🔍 Using OCR extraction...")
            
    #         for page_num in range(min(10, doc.page_count)):  # Limit OCR to first 10 pages
    #             page = doc[page_num]
                
    #             # Convert page to image
    #             mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
    #             pix = page.get_pixmap(matrix=mat)
    #             img_data = pix.tobytes("png")
    #             img = Image.open(io.BytesIO(img_data))
                
    #             # Apply OCR
    #             page_text = pytesseract.image_to_string(
    #                 img, 
    #                 config='--psm 6 -l eng'  # Page segmentation mode for uniform text
    #             )
    #             text += page_text + "\n"
            
    #         doc.close()
    #         return text.strip()
            
    #     except Exception as e:
    #         print(f"OCR extraction error: {e}")
    #         return ""
    
    
    def extract_text_ocr(self, pdf_path: str) -> str:
    
        if not self.config.OCR_ENABLED:
            return ""

        try:
        # Explicit path to tesseract.exe
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            doc = fitz.open(pdf_path)
            text = ""

            print(f"   🔍 Using OCR extraction...")

            for page_num in range(min(10, doc.page_count)):  # Limit OCR to first 10 pages
                page = doc[page_num]

            # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

            # Apply OCR
                page_text = pytesseract.image_to_string(
                    img,
                    config='--psm 6 -l eng'  # Page segmentation mode for uniform text
                )
                text += page_text + "\n"

            doc.close()
            return text.strip()

        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""

    
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\$\%\-\(\)\/\:]', ' ', text)
        
        # Remove very short lines (artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Keep lines with substantial content
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def categorize_content(self, text: str) -> Dict[str, str]:
        """Categorize text into pitch deck sections"""
        sections = {key: "" for key in self.section_keywords.keys()}
        
        if not text:
            return sections
        
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            
            # Check each section's keywords
            for section, keywords in self.section_keywords.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in sentence_lower)
                
                if keyword_matches > 0:
                    sections[section] += sentence + ". "
        
        return sections
    
    def extract_deck_data(self, pdf_path: str, deck_name: str) -> Dict:
        """Main extraction method with comprehensive fallbacks"""
        
        # First, diagnose the PDF
        diagnosis = self.diagnose_pdf(pdf_path)
        
        if not diagnosis.get('can_process', True):
            return {
                "error": f"{diagnosis.get('message', 'Cannot process PDF')}",
                "deck_name": deck_name
            }
        
        print(f"   📋 {diagnosis['pages']} pages, {diagnosis['file_size_mb']:.1f}MB")
        
        # Try extraction methods in order
        text = ""
        extraction_method = "none"
        
        # Method 1: PyMuPDF
        text = self.extract_text_pymupdf(pdf_path)
        if text and len(text) >= self.config.MIN_TEXT_LENGTH:
            extraction_method = "PyMuPDF"
        else:
            # Method 2: pdfplumber
            text = self.extract_text_pdfplumber(pdf_path)
            if text and len(text) >= self.config.MIN_TEXT_LENGTH:
                extraction_method = "pdfplumber"
            else:
                # Method 3: OCR (if enabled and images detected)
                if diagnosis.get('has_images', False):
                    text = self.extract_text_ocr(pdf_path)
                    if text and len(text) >= self.config.MIN_TEXT_LENGTH:
                        extraction_method = "OCR"
        
        if not text or len(text) < self.config.MIN_TEXT_LENGTH:
            return {
                "error": f"Could not extract sufficient text (got {len(text)} chars, need {self.config.MIN_TEXT_LENGTH}+)",
                "deck_name": deck_name,
                "diagnosis": diagnosis
            }
        
        print(f"   ✅ Extracted {len(text)} characters using {extraction_method}")
        
        # Clean and categorize text
        cleaned_text = self.clean_text(text)
        categorized_content = self.categorize_content(cleaned_text)
        
        return {
            "deck_name": deck_name,
            "full_text": cleaned_text,
            "word_count": len(cleaned_text.split()),
            "extraction_method": extraction_method,
            "diagnosis": diagnosis,
            **categorized_content
        }
