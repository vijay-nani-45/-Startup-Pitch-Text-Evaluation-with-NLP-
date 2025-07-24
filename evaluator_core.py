# evaluator_core.py - Core evaluation logic without circular dependencies
import os
import pandas as pd
import numpy as np
from typing import List, Dict
import glob
from pitch_extractor import PitchDeckExtractor
from nlp_scorer import NLPPitchScorer
from llm_analyzer import LLMPitchAnalyzer
from dashboard import PitchDashboard
import warnings
warnings.filterwarnings('ignore')

class StartupPitchEvaluator:
    def __init__(self, api_key: str = None):
        self.extractor = PitchDeckExtractor()
        self.scorer = NLPPitchScorer()
        self.llm_analyzer = LLMPitchAnalyzer(api_key=api_key)
        self.dashboard = PitchDashboard()
        
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file"""
        deck_name = os.path.basename(pdf_path).replace('.pdf', '')
        print(f"🔄 Processing: {deck_name}")
        
        try:
            # Extract content
            deck_data = self.extractor.extract_deck_data(pdf_path, deck_name)
            
            if 'error' in deck_data:
                return {'error': deck_data['error'], 'deck_name': deck_name}
            
            # Score the deck
            scores = self.scorer.score_deck(deck_data)
            
            # LLM analysis using API
            summary = self.llm_analyzer.generate_deck_summary(deck_data)
            category = self.llm_analyzer.classify_startup_category(deck_data)
            insight = self.llm_analyzer.generate_investability_insight(scores, deck_data)
            
            # Combine results
            result = {
                'deck_name': deck_name,
                'word_count': deck_data.get('word_count', 0),
                'category': category,
                'summary': summary,
                'investability_insight': insight,
                'timestamp': pd.Timestamp.now(),
                **scores
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'deck_name': deck_name}
    
    def process_pitch_decks(self, pdf_directory: str) -> pd.DataFrame:
        """Process all pitch decks in directory"""
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return pd.DataFrame()
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        all_results = []
        
        for pdf_path in pdf_files:
            result = self.process_single_pdf(pdf_path)
            
            if 'error' not in result:
                all_results.append(result)
                print(f"✅ Completed: {result['deck_name']} (Score: {result['composite_score']:.2f})")
            else:
                print(f"❌ Failed: {result['deck_name']} - {result['error']}")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    def generate_comprehensive_analysis(self, results_df: pd.DataFrame, output_dir: str = "output"):
        """Generate comprehensive analysis and reports"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE PITCH DECK ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Create summary table
        summary_table = self.dashboard.create_scoring_summary_table(results_df)
        
        # Display top performers
        print(f"\n🏆 TOP 3 PERFORMING DECKS:")
        top_3 = summary_table.head(3)
        for _, deck in top_3.iterrows():
            print(f"{deck['rank']}. {deck['deck_name']} - Score: {deck['composite_score']}/10")
            if 'investability_insight' in deck:
                print(f"   💡 {deck['investability_insight']}")
        
        # Save detailed results
        summary_table.to_csv(os.path.join(output_dir, 'pitch_deck_scores.csv'), index=False)
        print(f"\n💾 Detailed scores saved to: {output_dir}/pitch_deck_scores.csv")
        
        # Create visualizations
        print(f"\n📊 Generating visualizations...")
        
        try:
            # Create various charts
            radar_fig = self.dashboard.create_radar_chart(results_df)
            radar_fig.write_html(os.path.join(output_dir, 'radar_chart.html'))
            
            ranking_fig = self.dashboard.create_ranking_chart(results_df)
            ranking_fig.write_html(os.path.join(output_dir, 'ranking_chart.html'))
            
            print(f"✅ All visualizations saved to: {output_dir}/")
        except Exception as e:
            print(f"⚠️ Some visualizations may not be available: {e}")
        
        return summary_table
