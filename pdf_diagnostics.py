# pdf_diagnostics.py - New file for troubleshooting
import fitz
import os

def diagnose_pdf(pdf_path: str) -> Dict:
    """Diagnose PDF extraction issues"""
    try:
        doc = fitz.open(pdf_path)
        info = {
            'filename': os.path.basename(pdf_path),
            'pages': doc.page_count,
            'encrypted': doc.is_encrypted,
            'metadata': doc.metadata,
            'has_text': False,
            'has_images': False,
            'text_sample': ''
        }
        
        # Check first few pages for content
        for page_num in range(min(3, doc.page_count)):
            page = doc[page_num]
            text = page.get_text()
            if text and len(text.strip()) > 50:
                info['has_text'] = True
                info['text_sample'] = text[:200] + "..."
                break
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                info['has_images'] = True
        
        doc.close()
        return info
        
    except Exception as e:
        return {
            'filename': os.path.basename(pdf_path),
            'error': str(e)
        }

# Add to main.py for debugging
def diagnose_pdf_directory(pdf_directory: str):
    """Diagnose all PDFs in directory"""
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    print(f"\n🔍 PDF DIAGNOSTICS FOR {len(pdf_files)} FILES:")
    print("=" * 60)
    
    for pdf_path in pdf_files:
        info = diagnose_pdf(pdf_path)
        print(f"\n📄 {info['filename']}")
        
        if 'error' in info:
            print(f"   ❌ Error: {info['error']}")
        else:
            print(f"   📋 Pages: {info['pages']}")
            print(f"   🔒 Encrypted: {info['encrypted']}")
            print(f"   📝 Has Text: {info['has_text']}")
            print(f"   🖼️ Has Images: {info['has_images']}")
            if info['text_sample']:
                print(f"   📖 Sample: {info['text_sample'][:100]}...")
