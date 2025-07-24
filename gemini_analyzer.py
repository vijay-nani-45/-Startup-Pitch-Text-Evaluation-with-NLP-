# gemini_analyzer.py - Updated with current model names
import google.generativeai as genai
import numpy as np
from typing import Dict, List
import re
import json
import time
import os
from config import Config

class GeminiPitchAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize with current Gemini API models"""
        self.config = Config()
        
        # Get API key
        api_key = api_key or self.config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Updated model names (as of 2024/2025)
        self.model_options = [
            'gemini-1.5-flash',      # Latest and fastest
            'gemini-1.5-pro',       # More capable
            'gemini-1.0-pro',       # Legacy fallback
            'gemini-pro-latest',    # Always latest pro
        ]
        
        # Try to initialize model with fallbacks
        self.model = self._initialize_model()
        
        if not self.model:
            raise ValueError("Could not initialize any Gemini model. Check your API key and network connection.")
        
        print("✅ Gemini API initialized successfully")
    
    def _initialize_model(self):
        """Try to initialize Gemini model with updated names"""
        
        for model_name in self.model_options:
            try:
                print(f"   Trying model: {model_name}")
                
                # Create model instance
                model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple request
                test_response = model.generate_content(
                    "Test: Reply with 'OK'",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=10,
                        temperature=0.1,
                    )
                )
                
                if test_response and test_response.text:
                    print(f"✅ Successfully connected to {model_name}")
                    return model
                    
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        # If all models fail, try to list available models
        try:
            print("🔍 Checking available models...")
            available_models = list(genai.list_models())
            print("Available models:")
            for model in available_models:
                if 'generateContent' in model.supported_generation_methods:
                    print(f"  - {model.name}")
            
            # Try the first available model that supports generateContent
            for model in available_models:
                if 'generateContent' in model.supported_generation_methods:
                    try:
                        model_instance = genai.GenerativeModel(model.name)
                        test_response = model_instance.generate_content("Test: Reply with 'OK'")
                        if test_response and test_response.text:
                            print(f"✅ Successfully connected to {model.name}")
                            return model_instance
                    except:
                        continue
        except Exception as e:
            print(f"Could not list available models: {e}")
        
        return None
    
    def _safe_generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """Safely generate content with retries and error handling"""
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.3,
                    )
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return ""
    
    # Keep all your existing methods (generate_deck_summary, classify_startup_category, etc.)
    # but make sure they use self._safe_generate_content() instead of direct model calls
    
    def generate_deck_summary(self, deck_data: Dict) -> str:
        """Generate 4-bullet point summary using Gemini API"""
        full_text = deck_data.get('full_text', '')[:3000]
        
        if len(full_text) < 100:
            return "• Insufficient content for comprehensive summary generation"
        
        prompt = f"""
        Analyze this startup pitch deck and create exactly 4 bullet points that summarize the key aspects:

        Pitch Content: {full_text}

        Create 4 bullet points covering:
        1. Problem & Solution
        2. Market Opportunity  
        3. Traction & Business Model
        4. Team & Ask

        Format: Start each line with "•" and keep each point under 80 characters.
        Be specific and data-driven where possible.
        """
        
        response = self._safe_generate_content(prompt)
        
        if response:
            # Clean up and format response
            lines = response.split('\n')
            bullets = []
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    clean_line = line.lstrip('•-* ').strip()
                    if clean_line:
                        bullets.append(f"• {clean_line}")
            
            if len(bullets) >= 4:
                return '\n'.join(bullets[:4])
            elif len(bullets) > 0:
                return '\n'.join(bullets)
        
        # Fallback
        return self._generate_fallback_summary(deck_data)
    
    def classify_startup_category(self, deck_data: Dict) -> str:
        """Classify startup into categories using Gemini API"""
        full_text = deck_data.get('full_text', '')[:2000]
        
        if len(full_text) < 50:
            return "Other"
        
        categories = [
            "Fintech", "HealthTech", "EdTech", "SaaS", "E-commerce", 
            "AI/ML", "IoT", "Blockchain", "Gaming", "Social Media",
            "Enterprise Software", "Consumer Apps", "Hardware", "Biotech"
        ]
        
        prompt = f"""
        Analyze this startup pitch and classify it into ONE of these categories:
        
        Categories: {', '.join(categories)}
        
        Pitch Content: {full_text}
        
        Instructions:
        - Choose the MOST appropriate single category
        - Respond with ONLY the category name
        - If unclear, choose "Other"
        """
        
        response = self._safe_generate_content(prompt)
        
        if response:
            category = response.strip().replace('"', '').replace("'", "")
            
            for valid_cat in categories + ["Other"]:
                if valid_cat.lower() in category.lower():
                    return valid_cat
        
        return self._classify_fallback(deck_data)
    
    def generate_investability_insight(self, scores: Dict, deck_data: Dict) -> str:
        """Generate one-line investability insight using Gemini API"""
        composite_score = scores.get('composite_score', 0)
        category = self.classify_startup_category(deck_data)
        
        weak_areas = [k for k, v in scores.items() if k != 'composite_score' and v < 5]
        strong_areas = [k for k, v in scores.items() if k != 'composite_score' and v >= 8]
        
        prompt = f"""
        As a venture capital analyst, provide a concise investment insight for this {category} startup.
        
        Overall Score: {composite_score:.1f}/10
        Strong Areas: {', '.join(strong_areas) if strong_areas else 'None'}
        Weak Areas: {', '.join(weak_areas) if weak_areas else 'None'}
        
        Brief Content: {deck_data.get('full_text', '')[:500]}
        
        Provide ONE professional sentence (under 100 characters) that captures:
        - Investment readiness level
        - Key strength or concern
        - Category-specific insight
        """
        
        response = self._safe_generate_content(prompt)
        
        if response:
            sentences = response.split('.')
            insight = sentences[0].strip()
            if len(insight) > 120:
                insight = insight[:117] + "..."
            return insight
        
        return self._generate_fallback_insight(composite_score, category)
    
    def analyze_competitive_landscape(self, deck_data: Dict) -> str:
        """Analyze competitive positioning"""
        full_text = deck_data.get('full_text', '')[:2000]
        
        if len(full_text) < 100:
            return "Insufficient content for competitive analysis"
        
        prompt = f"""
        Analyze the competitive landscape mentioned in this pitch deck:
        
        Content: {full_text}
        
        Identify:
        1. Direct competitors mentioned
        2. Competitive advantages claimed
        3. Market positioning
        
        Provide a brief analysis (2-3 sentences) of their competitive position.
        """
        
        response = self._safe_generate_content(prompt)
        return response if response else "Competitive analysis not available"
    
    # Add your fallback methods here...
    def _generate_fallback_summary(self, deck_data: Dict) -> str:
        """Fallback rule-based summary"""
        summary_points = []
        
        sections = [
            ('problem', 'Problem'),
            ('solution', 'Solution'), 
            ('market', 'Market'),
            ('traction', 'Traction')
        ]
        
        for section_key, section_name in sections:
            content = deck_data.get(section_key, '')
            if content and len(content.strip()) > 20:
                summary = content[:70] + "..." if len(content) > 70 else content
                summary_points.append(f"• {section_name}: {summary}")
        
        while len(summary_points) < 4:
            summary_points.append(f"• Section {len(summary_points) + 1}: Content analysis pending")
        
        return '\n'.join(summary_points[:4])
    
    def _classify_fallback(self, deck_data: Dict) -> str:
        """Fallback rule-based classification"""
        full_text = deck_data.get('full_text', '').lower()
        
        category_keywords = {
            'Fintech': ['payment', 'banking', 'finance', 'loan', 'credit', 'investment'],
            'HealthTech': ['health', 'medical', 'healthcare', 'patient', 'doctor', 'hospital'],
            'EdTech': ['education', 'learning', 'student', 'course', 'school', 'university'],
            'SaaS': ['software', 'saas', 'cloud', 'api', 'platform', 'service'],
            'E-commerce': ['ecommerce', 'marketplace', 'retail', 'shopping', 'store'],
            'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural']
        }
        
        best_category = "Other"
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category if best_score > 0 else "Other"
    
    def _generate_fallback_insight(self, composite_score: float, category: str) -> str:
        """Fallback insight generation"""
        if composite_score >= 8:
            return f"Strong {category} opportunity with excellent fundamentals"
        elif composite_score >= 6:
            return f"Promising {category} startup with room for improvement"
        elif composite_score >= 4:
            return f"Early-stage {category} startup needs significant development"
        else:
            return f"Underdeveloped {category} pitch requires major improvements"
