# config.py - Updated model configuration
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Updated Gemini Models (current as of 2024/2025)
    GEMINI_MODELS = [
        'gemini-1.5-flash',      # Latest, fastest, most cost-effective
        'gemini-1.5-pro',       # More capable, better for complex tasks
        'gemini-1.0-pro',       # Legacy support
        'gemini-pro-latest',    # Always points to latest pro version
    ]
    
    # Rest of your config...
    SCORING_WEIGHTS = {
        'problem_clarity': 0.15,
        'market_potential': 0.2,
        'traction_strength': 0.25,
        'team_experience': 0.15,
        'business_model': 0.15,
        'vision_moat': 0.05,
        'overall_confidence': 0.05
    }
    
    OUTPUT_DIR = "output"
    REALTIME_RESULTS_DIR = "real_time_results"
    MAX_PDF_SIZE_MB = 50
    MIN_TEXT_LENGTH = 50
    OCR_ENABLED = True
