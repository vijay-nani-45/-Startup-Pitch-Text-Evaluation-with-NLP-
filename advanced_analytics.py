# advanced_analytics.py - New module for ML and AI analytics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
import torch

class PitchSuccessPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = [
            'problem_clarity', 'market_potential', 'traction_strength',
            'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
        ]
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = df[self.feature_columns].copy()
        
        # Add derived features
        features['market_traction_ratio'] = df['market_potential'] / (df['traction_strength'] + 0.1)
        features['team_execution_score'] = (df['team_experience'] + df['traction_strength']) / 2
        features['business_clarity_score'] = (df['business_model'] + df['problem_clarity']) / 2
        features['overall_strength'] = df[self.feature_columns].mean(axis=1)
        features['score_variance'] = df[self.feature_columns].var(axis=1)
        
        # Encode categories if available
        if 'category' in df.columns:
            try:
                if not hasattr(self.category_encoder, 'classes_'):
                    category_encoded = self.category_encoder.fit_transform(df['category'])
                else:
                    category_encoded = self.category_encoder.transform(df['category'])
                features['category_encoded'] = category_encoded
            except ValueError:
                # Handle unseen categories
                features['category_encoded'] = 0
        
        return features
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train the success prediction model"""
        features = self.prepare_features(df)
        
        # Create synthetic success labels based on composite score
        # In production, this would come from actual investment outcomes
        success_threshold = 7.0
        labels = (df['composite_score'] >= success_threshold).astype(int)
        
        # Add some noise to make it more realistic
        noise_factor = 0.1
        composite_with_noise = df['composite_score'] + np.random.normal(0, noise_factor, len(df))
        continuous_labels = np.clip(composite_with_noise / 10, 0, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, continuous_labels, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate accuracy metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'feature_importance': importance,
            'train_score': train_score,
            'test_score': test_score,
            'model_trained': True
        }
    
    def predict_success_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Predict investment success probability"""
        if not self.is_trained:
            self.train_model(df)
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict(features_scaled)
        
        return np.clip(probabilities, 0, 1)
    
    def get_feature_importance_chart(self) -> go.Figure:
        """Create feature importance visualization"""
        if not hasattr(self, 'feature_importance'):
            return go.Figure()
        
        fig = px.bar(
            self.feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Key Success Factors (Feature Importance)'
        )
        fig.update_layout(height=400)
        return fig

class AdvancedTextAnalyzer:
    def __init__(self):
        # Initialize advanced NLP models with error handling
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except Exception:
            # Fallback to basic sentiment analysis
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.use_transformer_sentiment = False
        else:
            self.use_transformer_sentiment = True
    
    def analyze_pitch_language(self, deck_data: Dict) -> Dict:
        """Advanced language analysis of pitch content"""
        full_text = deck_data.get('full_text', '')
        
        if len(full_text) < 50:
            return self._get_default_language_analysis()
        
        # Sentiment analysis by section
        sections = ['problem', 'solution', 'market', 'traction', 'team']
        sentiment_scores = {}
        
        for section in sections:
            section_text = deck_data.get(section, '')
            if section_text and len(section_text) > 50:
                sentiment_scores[f'{section}_sentiment'] = self._analyze_section_sentiment(section_text)
        
        # Language confidence indicators
        confidence_indicators = [
            'proven', 'validated', 'successful', 'achieved', 'demonstrated',
            'experienced', 'expert', 'leading', 'industry-leading', 'proprietary',
            'patent-pending', 'breakthrough', 'revolutionary', 'first-to-market'
        ]
        
        confidence_score = sum(
            full_text.lower().count(indicator) for indicator in confidence_indicators
        ) / len(full_text.split()) * 1000  # Normalize per 1000 words
        
        # Technical sophistication analysis
        technical_terms = [
            'algorithm', 'machine learning', 'ai', 'blockchain', 'api',
            'scalable', 'cloud', 'data analytics', 'automation', 'optimization'
        ]
        
        tech_sophistication = sum(
            full_text.lower().count(term) for term in technical_terms
        ) / len(full_text.split()) * 1000
        
        # Calculate language complexity metrics
        words = full_text.split()
        unique_words = set(words)
        
        return {
            **sentiment_scores,
            'confidence_score': confidence_score,
            'tech_sophistication': tech_sophistication,
            'vocabulary_diversity': len(unique_words) / len(words) if words else 0,
            'avg_sentence_length': self._calculate_avg_sentence_length(full_text),
            'readability_score': self._calculate_readability_score(full_text)
        }
    
    def _analyze_section_sentiment(self, text: str) -> float:
        """Analyze sentiment of a text section"""
        if self.use_transformer_sentiment:
            try:
                result = self.sentiment_analyzer(text[:512])  # Limit length
                return result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
            except Exception:
                pass
        
        # Fallback to VADER
        if hasattr(self.sentiment_analyzer, 'polarity_scores'):
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        
        return 0.0
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 0]
        if not sentences:
            return 0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        return np.mean(sentence_lengths)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word length"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 0]
        words = text.split()
        
        if not sentences or not words:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability formula (lower is more readable)
        readability = (avg_sentence_length * 1.015) + (avg_word_length * 84.6) - 206.835
        
        # Normalize to 0-10 scale (higher is more readable)
        normalized_score = max(0, min(10, (100 - readability) / 10))
        
        return normalized_score
    
    def _get_default_language_analysis(self) -> Dict:
        """Return default analysis when text is insufficient"""
        return {
            'confidence_score': 0,
            'tech_sophistication': 0,
            'vocabulary_diversity': 0,
            'avg_sentence_length': 0,
            'readability_score': 5
        }

class IndustryBenchmarkAnalyzer:
    def __init__(self):
        # Industry benchmark data (would come from database in production)
        self.industry_benchmarks = {
            'Fintech': {
                'composite_score': 7.2, 'market_potential': 8.1, 'traction_strength': 6.8,
                'team_experience': 7.5, 'business_model': 7.8, 'problem_clarity': 7.0
            },
            'HealthTech': {
                'composite_score': 6.9, 'market_potential': 7.8, 'traction_strength': 6.2,
                'team_experience': 7.1, 'business_model': 6.5, 'problem_clarity': 7.3
            },
            'SaaS': {
                'composite_score': 7.5, 'market_potential': 7.9, 'traction_strength': 7.8,
                'team_experience': 7.6, 'business_model': 8.1, 'problem_clarity': 7.2
            },
            'AI/ML': {
                'composite_score': 7.8, 'market_potential': 8.3, 'traction_strength': 7.1,
                'team_experience': 8.0, 'business_model': 7.4, 'problem_clarity': 7.6
            },
            'E-commerce': {
                'composite_score': 6.8, 'market_potential': 7.5, 'traction_strength': 7.2,
                'team_experience': 6.9, 'business_model': 7.0, 'problem_clarity': 6.5
            },
            'EdTech': {
                'composite_score': 6.5, 'market_potential': 7.2, 'traction_strength': 5.8,
                'team_experience': 6.8, 'business_model': 6.3, 'problem_clarity': 6.9
            }
        }
    
    def create_benchmark_comparison(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compare current pitches against industry benchmarks"""
        
        benchmark_comparison = []
        
        for _, deck in results_df.iterrows():
            category = deck.get('category', 'Other')
            
            if category in self.industry_benchmarks:
                benchmark = self.industry_benchmarks[category]
                
                comparison = {
                    'deck_name': deck['deck_name'],
                    'category': category,
                    'composite_vs_benchmark': deck['composite_score'] - benchmark['composite_score'],
                    'market_vs_benchmark': deck.get('market_potential', 0) - benchmark['market_potential'],
                    'traction_vs_benchmark': deck.get('traction_strength', 0) - benchmark['traction_strength'],
                    'team_vs_benchmark': deck.get('team_experience', 0) - benchmark['team_experience'],
                    'percentile_rank': self._calculate_percentile_rank(deck['composite_score'], category),
                    'benchmark_category': category,
                    'overall_vs_industry': 'Above' if deck['composite_score'] > benchmark['composite_score'] else 'Below'
                }
                
                benchmark_comparison.append(comparison)
        
        return pd.DataFrame(benchmark_comparison)
    
    def _calculate_percentile_rank(self, score: float, category: str) -> float:
        """Calculate percentile rank within category"""
        # Simplified calculation - in production would use historical data
        if category in self.industry_benchmarks:
            benchmark_score = self.industry_benchmarks[category]['composite_score']
            # Assume normal distribution with std of 1.5
            from scipy import stats
            percentile = stats.norm.cdf(score, loc=benchmark_score, scale=1.5) * 100
            return min(99, max(1, percentile))
        return 50  # Default to median
    
    def create_benchmark_visualization(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Create benchmark comparison visualization"""
        
        if comparison_df.empty:
            return go.Figure()
        
        fig = px.scatter(
            comparison_df,
            x='composite_vs_benchmark',
            y='percentile_rank',
            color='category',
            size=abs(comparison_df['composite_vs_benchmark']),
            hover_name='deck_name',
            title="Performance vs Industry Benchmarks",
            labels={
                'composite_vs_benchmark': 'Score Difference vs Benchmark',
                'percentile_rank': 'Industry Percentile Rank'
            }
        )
        
        # Add reference lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                      annotation_text="Industry Median")
        fig.add_vline(x=0, line_dash="dash", line_color="gray", 
                      annotation_text="Benchmark Score")
        
        fig.update_layout(height=500)
        return fig

class AIInsightsGenerator:
    def __init__(self, gemini_analyzer=None):
        self.gemini = gemini_analyzer
        self.text_analyzer = AdvancedTextAnalyzer()
        self.benchmark_analyzer = IndustryBenchmarkAnalyzer()
    
    def generate_portfolio_insights(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive AI-powered portfolio insights"""
        
        insights = {
            'portfolio_summary': self._analyze_portfolio_composition(results_df),
            'risk_assessment': self._assess_portfolio_risk(results_df),
            'investment_recommendations': self._generate_investment_recommendations(results_df),
            'market_trends': self._identify_market_trends(results_df),
            'due_diligence_flags': self._identify_due_diligence_flags(results_df),
            'benchmark_analysis': self._create_benchmark_insights(results_df)
        }
        
        return insights
    
    def _analyze_portfolio_composition(self, df: pd.DataFrame) -> Dict:
        """Analyze overall portfolio composition"""
        total_decks = len(df)
        avg_score = df['composite_score'].mean()
        
        high_performers = len(df[df['composite_score'] >= 7])
        medium_performers = len(df[(df['composite_score'] >= 5) & (df['composite_score'] < 7)])
        low_performers = len(df[df['composite_score'] < 5])
        
        # Find strongest and weakest dimensions
        dimension_cols = ['problem_clarity', 'market_potential', 'traction_strength', 
                         'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
        existing_dims = [col for col in dimension_cols if col in df.columns]
        
        if existing_dims:
            dimension_means = df[existing_dims].mean()
            strongest_dim = dimension_means.idxmax()
            weakest_dim = dimension_means.idxmin()
        else:
            strongest_dim = "N/A"
            weakest_dim = "N/A"
        
        return {
            'total_analyzed': total_decks,
            'average_quality': avg_score,
            'quality_distribution': {
                'investment_ready': high_performers,
                'needs_development': medium_performers,
                'significant_work_needed': low_performers
            },
            'category_diversity': df['category'].nunique() if 'category' in df.columns else 0,
            'strongest_dimension': strongest_dim,
            'weakest_dimension': weakest_dim,
            'score_variance': df['composite_score'].std()
        }
    
    def _assess_portfolio_risk(self, df: pd.DataFrame) -> Dict:
        """Assess overall portfolio risk"""
        risk_metrics = {}
        
        # Concentration risk
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            max_category_pct = (category_counts.iloc[0] / len(df)) * 100
            risk_metrics['concentration_risk'] = 'High' if max_category_pct > 60 else 'Medium' if max_category_pct > 40 else 'Low'
        
        # Quality risk
        low_quality_pct = (len(df[df['composite_score'] < 5]) / len(df)) * 100
        risk_metrics['quality_risk'] = 'High' if low_quality_pct > 40 else 'Medium' if low_quality_pct > 20 else 'Low'
        
        # Execution risk (based on traction scores)
        if 'traction_strength' in df.columns:
            avg_traction = df['traction_strength'].mean()
            risk_metrics['execution_risk'] = 'High' if avg_traction < 4 else 'Medium' if avg_traction < 6 else 'Low'
        
        return risk_metrics
    
    def _generate_investment_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate specific investment recommendations"""
        recommendations = []
        
        # Top performers
        top_decks = df.nlargest(min(3, len(df)), 'composite_score')
        for _, deck in top_decks.iterrows():
            recommendations.append({
                'type': 'STRONG_BUY',
                'deck': deck['deck_name'],
                'score': deck['composite_score'],
                'category': deck.get('category', 'Unknown'),
                'reasoning': f"Exceptional {deck.get('category', 'startup')} opportunity with strong fundamentals across all dimensions",
                'confidence': 'HIGH',
                'priority': 1
            })
        
        # Hidden gems (high potential in specific areas)
        hidden_gems = df[
            (df['composite_score'] >= 6) & 
            (df['composite_score'] < 7) &
            ((df.get('market_potential', 0) >= 8) | (df.get('traction_strength', 0) >= 8))
        ]
        
        for _, deck in hidden_gems.iterrows():
            if deck['deck_name'] not in [r['deck'] for r in recommendations]:
                strong_area = 'market potential' if deck.get('market_potential', 0) >= 8 else 'traction'
                recommendations.append({
                    'type': 'CONSIDER',
                    'deck': deck['deck_name'],
                    'score': deck['composite_score'],
                    'category': deck.get('category', 'Unknown'),
                    'reasoning': f"Strong {strong_area} with solid overall fundamentals in {deck.get('category', 'startup')} space",
                    'confidence': 'MEDIUM',
                    'priority': 2
                })
        
        # Watch list (potential with development needed)
        watch_list = df[
            (df['composite_score'] >= 4) & 
            (df['composite_score'] < 6) &
            (df.get('team_experience', 0) >= 7)  # Strong team
        ]
        
        for _, deck in watch_list.iterrows():
            if deck['deck_name'] not in [r['deck'] for r in recommendations]:
                recommendations.append({
                    'type': 'WATCH',
                    'deck': deck['deck_name'],
                    'score': deck['composite_score'],
                    'category': deck.get('category', 'Unknown'),
                    'reasoning': f"Strong team but needs development in execution and market validation",
                    'confidence': 'LOW',
                    'priority': 3
                })
        
        return sorted(recommendations, key=lambda x: (x['priority'], -x['score']))
    
    def _identify_market_trends(self, df: pd.DataFrame) -> Dict:
        """Identify trends across the portfolio"""
        trends = {}
        
        if 'category' in df.columns and len(df) > 1:
            # Category performance trends
            category_performance = df.groupby('category')['composite_score'].mean().sort_values(ascending=False)
            trends['top_performing_categories'] = category_performance.head(3).to_dict()
            
            # Category with highest market potential
            if 'market_potential' in df.columns:
                market_leaders = df.groupby('category')['market_potential'].mean().sort_values(ascending=False)
                trends['highest_market_potential_categories'] = market_leaders.head(3).to_dict()
        
        # Overall trends
        avg_scores = {}
        dimension_cols = ['problem_clarity', 'market_potential', 'traction_strength', 
                         'team_experience', 'business_model']
        
        for col in dimension_cols:
            if col in df.columns:
                avg_scores[col] = df[col].mean()
        
        trends['dimension_strengths'] = avg_scores
        
        return trends
    
    def _identify_due_diligence_flags(self, df: pd.DataFrame) -> List[Dict]:
        """Identify potential red flags requiring due diligence"""
        flags = []
        
        for _, deck in df.iterrows():
            deck_flags = []
            
            # Low team experience with high market claims
            if (deck.get('team_experience', 0) < 4 and deck.get('market_potential', 0) > 8):
                deck_flags.append("Team experience doesn't match market opportunity claims")
            
            # High traction claims but low business model clarity
            if (deck.get('traction_strength', 0) > 7 and deck.get('business_model', 0) < 5):
                deck_flags.append("Strong traction claims but unclear monetization strategy")
            
            # Low problem clarity but high overall confidence
            if (deck.get('problem_clarity', 0) < 4 and deck.get('overall_confidence', 0) > 7):
                deck_flags.append("High confidence but poorly defined problem statement")
            
            # Composite score significantly different from individual scores
            individual_scores = [deck.get(col, 0) for col in ['problem_clarity', 'market_potential', 'traction_strength', 'team_experience', 'business_model'] if col in deck]
            if individual_scores:
                avg_individual = np.mean(individual_scores)
                if abs(deck['composite_score'] - avg_individual) > 2:
                    deck_flags.append("Inconsistent scoring pattern requires review")
            
            if deck_flags:
                flags.append({
                    'deck_name': deck['deck_name'],
                    'composite_score': deck['composite_score'],
                    'flags': deck_flags
                })
        
        return flags
    
    def _create_benchmark_insights(self, df: pd.DataFrame) -> Dict:
        """Create insights based on industry benchmarks"""
        benchmark_comparison = self.benchmark_analyzer.create_benchmark_comparison(df)
        
        if benchmark_comparison.empty:
            return {'message': 'No benchmark data available'}
        
        # Identify outperformers and underperformers
        outperformers = benchmark_comparison[benchmark_comparison['composite_vs_benchmark'] > 1]
        underperformers = benchmark_comparison[benchmark_comparison['composite_vs_benchmark'] < -1]
        
        return {
            'total_with_benchmarks': len(benchmark_comparison),
            'outperformers': len(outperformers),
            'underperformers': len(underperformers),
            'avg_vs_benchmark': benchmark_comparison['composite_vs_benchmark'].mean(),
            'best_vs_benchmark': benchmark_comparison.loc[benchmark_comparison['composite_vs_benchmark'].idxmax()] if not benchmark_comparison.empty else None,
            'categories_analyzed': benchmark_comparison['category'].unique().tolist()
        }
