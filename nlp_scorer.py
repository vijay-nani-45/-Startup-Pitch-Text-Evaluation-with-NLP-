# nlp_scorer.py - Enhanced scoring system
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class EnhancedNLPScorer:
    def __init__(self):
        self.config = Config()
        self.analyzer = SentimentIntensityAnalyzer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
            print("Warning: NLTK stopwords not available")
    
    def score_problem_clarity(self, problem_text: str) -> float:
        """Score problem statement clarity (0-10)"""
        if not problem_text or len(problem_text.strip()) < 20:
            return 0.0
        
        # Enhanced problem indicators
        clarity_indicators = [
            'specifically', 'particular', 'exactly', 'precisely', 'currently',
            'existing', 'traditional', 'inefficient', 'difficult', 'expensive',
            'time-consuming', 'frustrating', 'broken', 'outdated'
        ]
        
        problem_lower = problem_text.lower()
        indicator_score = min(sum(2 for indicator in clarity_indicators if indicator in problem_lower), 10)
        
        # Look for quantified problems
        numbers = re.findall(r'\d+(?:\.\d+)?(?:\s*(?:million|billion|percent|%|\$))', problem_lower)
        quantification_score = min(len(numbers) * 3, 10)
        
        # Sentiment analysis - problems should have negative sentiment
        sentiment = self.analyzer.polarity_scores(problem_text)
        negativity_score = min(abs(sentiment['neg']) * 15, 10)
        
        # Length and specificity
        word_count = len(problem_text.split())
        length_score = min((word_count / 30) * 10, 10)
        
        # Combine scores
        final_score = (indicator_score * 0.3 + quantification_score * 0.2 + 
                      negativity_score * 0.3 + length_score * 0.2)
        
        return min(final_score, 10.0)
    
    def score_market_potential(self, market_text: str) -> float:
        """Score market potential mentions (0-10)"""
        if not market_text or len(market_text.strip()) < 10:
            return 0.0
        
        # Enhanced market indicators
        market_indicators = [
            'billion', 'million', 'tam', 'sam', 'som', 'market size', 'addressable',
            'growing', 'cagr', 'forecast', 'projected', 'expected', 'opportunity',
            'potential', 'segment', 'industry', 'sector'
        ]
        
        market_lower = market_text.lower()
        indicator_score = min(sum(1.5 for indicator in market_indicators if indicator in market_lower), 10)
        
        # Look for specific market size numbers
        market_numbers = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|million|b|m)', market_lower)
        size_score = min(len(market_numbers) * 4, 10)
        
        # Growth indicators
        growth_indicators = ['growth', 'growing', 'increase', 'expanding', 'rising', 'cagr']
        growth_score = min(sum(2 for word in growth_indicators if word in market_lower), 10)
        
        # Research/validation mentions
        validation_words = ['research', 'report', 'study', 'analysis', 'data', 'statistics']
        validation_score = min(sum(1 for word in validation_words if word in market_lower), 8)
        
        final_score = (indicator_score * 0.3 + size_score * 0.4 + 
                      growth_score * 0.2 + validation_score * 0.1)
        
        return min(final_score, 10.0)
    
    def score_traction_strength(self, traction_text: str) -> float:
        """Score traction and metrics (0-10)"""
        if not traction_text or len(traction_text.strip()) < 10:
            return 0.0
        
        # Traction indicators
        traction_indicators = [
            'revenue', 'customers', 'users', 'growth', 'monthly', 'quarterly',
            'annual', 'retention', 'churn', 'conversion', 'engagement',
            'active users', 'recurring', 'subscribers', 'downloads'
        ]
        
        traction_lower = traction_text.lower()
        indicator_score = min(sum(1.5 for indicator in traction_indicators if indicator in traction_lower), 10)
        
        # Growth percentages and metrics
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', traction_text)
        growth_numbers = [float(p) for p in percentages if float(p) > 5]  # Meaningful growth
        growth_score = min(len(growth_numbers) * 3, 10)
        
        # Revenue mentions with numbers
        revenue_patterns = [
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|m|million|thousand)',
            r'(\d+(?:,\d+)*)\s*(?:dollars|revenue|sales)'
        ]
        revenue_mentions = 0
        for pattern in revenue_patterns:
            revenue_mentions += len(re.findall(pattern, traction_lower))
        revenue_score = min(revenue_mentions * 4, 10)
        
        # Specific metrics (KPIs)
        kpi_patterns = [
            r'\d+(?:,\d+)*\s*(?:users|customers|downloads|signups)',
            r'(\d+(?:\.\d+)?)\s*(?:mrr|arr|ltv|cac)'
        ]
        kpi_count = 0
        for pattern in kpi_patterns:
            kpi_count += len(re.findall(pattern, traction_lower))
        kpi_score = min(kpi_count * 2, 10)
        
        final_score = (indicator_score * 0.25 + growth_score * 0.25 + 
                      revenue_score * 0.3 + kpi_score * 0.2)
        
        return min(final_score, 10.0)
    
    def score_team_experience(self, team_text: str) -> float:
        """Score team background and experience (0-10)"""
        if not team_text or len(team_text.strip()) < 10:
            return 0.0
        
        # Experience indicators
        experience_indicators = [
            'founded', 'co-founded', 'previous', 'former', 'ex-', 'years',
            'experience', 'background', 'worked at', 'led', 'managed',
            'built', 'scaled', 'grew', 'successful'
        ]
        
        team_lower = team_text.lower()
        exp_score = min(sum(1.5 for indicator in experience_indicators if indicator in team_lower), 10)
        
        # Educational background
        education_keywords = [
            'mba', 'phd', 'masters', 'degree', 'stanford', 'harvard', 'mit',
            'berkeley', 'university', 'college', 'graduate'
        ]
        edu_score = min(sum(2 for keyword in education_keywords if keyword in team_lower), 8)
        
        # Company experience (big tech/successful companies)
        company_indicators = [
            'google', 'apple', 'microsoft', 'amazon', 'facebook', 'meta',
            'netflix', 'tesla', 'uber', 'airbnb', 'stripe', 'square',
            'goldman sachs', 'mckinsey', 'bain', 'bcg'
        ]
        company_score = min(sum(3 for company in company_indicators if company in team_lower), 10)
        
        # Leadership titles
        titles = ['ceo', 'cto', 'cfo', 'founder', 'director', 'vp', 'head of', 'lead', 'senior']
        title_score = min(sum(1 for title in titles if title in team_lower), 8)
        
        final_score = (exp_score * 0.3 + edu_score * 0.2 + 
                      company_score * 0.35 + title_score * 0.15)
        
        return min(final_score, 10.0)
    
    def score_business_model(self, business_text: str) -> float:
        """Score business model clarity (0-10)"""
        if not business_text or len(business_text.strip()) < 10:
            return 0.0
        
        # Business model types
        model_types = [
            'subscription', 'saas', 'freemium', 'marketplace', 'commission',
            'transaction fee', 'licensing', 'advertising', 'per user',
            'enterprise', 'b2b', 'b2c', 'recurring', 'one-time'
        ]
        
        business_lower = business_text.lower()
        model_score = min(sum(2 for model in model_types if model in business_lower), 10)
        
        # Pricing mentions
        pricing_patterns = [
            r'\$\s*(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:month|user|seat)',
            r'(\d+(?:\.\d+)?)\s*(?:dollars|pricing|price|cost)'
        ]
        pricing_mentions = 0
        for pattern in pricing_patterns:
            pricing_mentions += len(re.findall(pattern, business_text))
        pricing_score = min(pricing_mentions * 3, 10)
        
        # Revenue clarity
        revenue_keywords = [
            'revenue streams', 'monetization', 'income', 'profit', 'margin',
            'ltv', 'cac', 'payback', 'unit economics', 'gross margin'
        ]
        revenue_score = min(sum(2 for keyword in revenue_keywords if keyword in business_lower), 10)
        
        final_score = (model_score * 0.4 + pricing_score * 0.3 + revenue_score * 0.3)
        
        return min(final_score, 10.0)
    
    def score_vision_moat(self, full_text: str) -> float:
        """Score vision and competitive moat (0-10)"""
        if not full_text or len(full_text.strip()) < 50:
            return 0.0
        
        # Vision indicators
        vision_indicators = [
            'vision', 'mission', 'future', 'transform', 'revolutionize',
            'disrupt', 'change', 'reimagine', 'redefine', 'innovate'
        ]
        
        text_lower = full_text.lower()
        vision_score = min(sum(1.5 for indicator in vision_indicators if indicator in text_lower), 8)
        
        # Competitive moat indicators
        moat_indicators = [
            'patent', 'intellectual property', 'proprietary', 'unique',
            'competitive advantage', 'moat', 'barrier', 'defensible',
            'exclusive', 'first mover', 'network effect'
        ]
        moat_score = min(sum(2 for indicator in moat_indicators if indicator in text_lower), 10)
        
        # Technology differentiation
        tech_indicators = [
            'ai', 'machine learning', 'blockchain', 'iot', 'cloud',
            'api', 'algorithm', 'data', 'analytics', 'automation'
        ]
        tech_score = min(sum(1 for indicator in tech_indicators if indicator in text_lower), 8)
        
        final_score = (vision_score * 0.3 + moat_score * 0.5 + tech_score * 0.2)
        
        return min(final_score, 10.0)
    
    def score_overall_confidence(self, full_text: str) -> float:
        """Score overall confidence and professional tone (0-10)"""
        if not full_text or len(full_text.strip()) < 50:
            return 0.0
        
        # Confidence words
        confidence_words = [
            'proven', 'validated', 'successful', 'achieved', 'accomplished',
            'demonstrated', 'experienced', 'expert', 'leading', 'best-in-class',
            'award-winning', 'recognized', 'established', 'track record'
        ]
        
        text_lower = full_text.lower()
        confidence_score = min(sum(1.5 for word in confidence_words if word in text_lower), 10)
        
        # Sentiment analysis
        sentiment = self.analyzer.polarity_scores(full_text)
        positivity_score = min(sentiment['pos'] * 12, 10)
        
        # Professional language indicators
        professional_words = [
            'strategy', 'execute', 'implement', 'optimize', 'leverage',
            'synergy', 'scalable', 'sustainable', 'efficient', 'effective'
        ]
        prof_score = min(sum(0.5 for word in professional_words if word in text_lower), 8)
        
        final_score = (confidence_score * 0.4 + positivity_score * 0.4 + prof_score * 0.2)
        
        return min(final_score, 10.0)
    
    def score_deck(self, deck_data: Dict) -> Dict[str, float]:
        """Score a complete pitch deck"""
        scores = {
            'problem_clarity': self.score_problem_clarity(deck_data.get('problem', '')),
            'market_potential': self.score_market_potential(deck_data.get('market', '')),
            'traction_strength': self.score_traction_strength(deck_data.get('traction', '')),
            'team_experience': self.score_team_experience(deck_data.get('team', '')),
            'business_model': self.score_business_model(deck_data.get('business_model', '')),
            'vision_moat': self.score_vision_moat(deck_data.get('full_text', '')),
            'overall_confidence': self.score_overall_confidence(deck_data.get('full_text', ''))
        }
        
        # Calculate composite score using configured weights
        composite_score = sum(scores[dim] * self.config.SCORING_WEIGHTS[dim] for dim in scores.keys())
        scores['composite_score'] = composite_score
        
        return scores
