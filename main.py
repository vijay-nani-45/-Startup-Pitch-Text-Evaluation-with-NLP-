# main.py - Complete main application
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict
import glob
import time
from datetime import datetime
import argparse

# Import our modules
from config import Config
from pdf_extractor import EnhancedPDFExtractor
from nlp_scorer import EnhancedNLPScorer
from gemini_analyzer import GeminiPitchAnalyzer
from dashboard import EnhancedDashboard

# For real-time monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class StartupPitchEvaluator:
    def __init__(self, api_key: str = None):
        """Initialize the complete evaluation system"""
        self.config = Config()
        
        # Initialize components
        try:
            self.extractor = EnhancedPDFExtractor()
            self.scorer = EnhancedNLPScorer()
            self.gemini_analyzer = GeminiPitchAnalyzer(api_key=api_key)
            self.dashboard = EnhancedDashboard()
            
            print("✅ System initialized successfully")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def diagnose_pdf_directory(self, pdf_directory: str):
        """Diagnose all PDFs in directory before processing"""
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            return False
        
        print(f"\n🔍 DIAGNOSING {len(pdf_files)} PDF FILES:")
        print("=" * 70)
        
        processable_count = 0
        
        for pdf_path in pdf_files:
            diagnosis = self.extractor.diagnose_pdf(pdf_path)
            
            status_icon = "✅" if diagnosis['status'] == 'ok' else "⚠️" if diagnosis['status'] == 'warning' else "❌"
            print(f"\n{status_icon} {diagnosis['filename']}")
            
            if 'pages' in diagnosis:
                print(f"   📄 Pages: {diagnosis['pages']}")
                print(f"   💾 Size: {diagnosis['file_size_mb']:.1f} MB")
                print(f"   🔒 Encrypted: {diagnosis.get('encrypted', 'Unknown')}")
                print(f"   📝 Has Text: {diagnosis.get('has_text', False)}")
                print(f"   🖼️ Has Images: {diagnosis.get('has_images', False)}")
                
                if diagnosis.get('text_sample'):
                    print(f"   📖 Sample: {diagnosis['text_sample'][:80]}...")
            
            if 'message' in diagnosis:
                print(f"   💬 Note: {diagnosis['message']}")
            
            if diagnosis.get('can_process', False):
                processable_count += 1
        
        print(f"\n📊 Summary: {processable_count}/{len(pdf_files)} files can be processed")
        
        return processable_count > 0
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file with comprehensive analysis"""
        deck_name = os.path.basename(pdf_path).replace('.pdf', '')
        start_time = time.time()
        
        print(f"\n🔄 Processing: {deck_name}")
        
        try:
            # Step 1: Extract content
            deck_data = self.extractor.extract_deck_data(pdf_path, deck_name)
            
            if 'error' in deck_data:
                return {
                    'error': deck_data['error'], 
                    'deck_name': deck_name,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Score with NLP
            print(f"   🧠 Analyzing content...")
            scores = self.scorer.score_deck(deck_data)
            
            # Step 3: Gemini API analysis
            print(f"   🤖 Generating AI insights...")
            try:
                summary = self.gemini_analyzer.generate_deck_summary(deck_data)
                category = self.gemini_analyzer.classify_startup_category(deck_data)
                insight = self.gemini_analyzer.generate_investability_insight(scores, deck_data)
                
                # Additional analysis
                competitive_analysis = self.gemini_analyzer.analyze_competitive_landscape(deck_data)
                
            except Exception as e:
                print(f"   ⚠️ Gemini API error: {e}")
                summary = self.gemini_analyzer._generate_fallback_summary(deck_data)
                category = self.gemini_analyzer._classify_fallback(deck_data)
                insight = self.gemini_analyzer._generate_fallback_insight(scores.get('composite_score', 0), category)
                competitive_analysis = "Analysis not available"
            
            processing_time = time.time() - start_time
            
            # Combine all results
            result = {
                'deck_name': deck_name,
                'word_count': deck_data.get('word_count', 0),
                'extraction_method': deck_data.get('extraction_method', 'unknown'),
                'category': category,
                'summary': summary,
                'investability_insight': insight,
                'competitive_analysis': competitive_analysis,
                'processing_time': processing_time,
                'timestamp': datetime.now(),
                **scores
            }
            
            print(f"   ✅ Completed in {processing_time:.1f}s (Score: {scores['composite_score']:.2f}/10)")
            return result
            
        except Exception as e:
            return {
                'error': str(e), 
                'deck_name': deck_name,
                'processing_time': time.time() - start_time
            }
    
    def process_pitch_decks(self, pdf_directory: str, diagnose_first: bool = True) -> pd.DataFrame:
        """Process all pitch decks in directory"""
        
        # Optional diagnosis
        if diagnose_first:
            can_process = self.diagnose_pdf_directory(pdf_directory)
            if not can_process:
                return pd.DataFrame()
            
            proceed = input("\n🤔 Continue with processing? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Processing cancelled.")
                return pd.DataFrame()
        
        # Get PDF files
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            return pd.DataFrame()
        
        print(f"\n🚀 Starting batch processing of {len(pdf_files)} files...")
        print("=" * 70)
        
        all_results = []
        successful_count = 0
        total_start_time = time.time()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end=" ")
            
            result = self.process_single_pdf(pdf_path)
            
            if 'error' not in result:
                all_results.append(result)
                successful_count += 1
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*70}")
        print(f"🎉 BATCH PROCESSING COMPLETE")
        print(f"✅ Successfully processed: {successful_count}/{len(pdf_files)} files")
        print(f"⏱️ Total time: {total_time:.1f}s (avg: {total_time/len(pdf_files):.1f}s per file)")
        print(f"{'='*70}")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    def generate_comprehensive_analysis(self, results_df: pd.DataFrame, output_dir: str = None):
        """Generate comprehensive analysis and reports"""
        
        if results_df.empty:
            print("❌ No results to analyze")
            return None
        
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 GENERATING COMPREHENSIVE ANALYSIS...")
        print("=" * 60)
        
        # Create summary table
        summary_table = self.dashboard.create_scoring_summary_table(results_df)
        
        # Display key insights
        self._display_key_insights(results_df, summary_table)
        
        # Save CSV results
        csv_path = os.path.join(output_dir, 'pitch_deck_analysis.csv')
        summary_table.to_csv(csv_path, index=False)
        print(f"💾 Detailed results saved: {csv_path}")
        
        # Generate visualizations
        print(f"\n📈 Creating visualizations...")
        
        try:
            # Individual chart files
            charts = {
                'radar_chart.html': self.dashboard.create_radar_chart(results_df),
                'ranking_chart.html': self.dashboard.create_ranking_chart(results_df),
                'score_distributions.html': self.dashboard.create_score_distribution(results_df),
                'performance_matrix.html': self.dashboard.create_performance_matrix(results_df)
            }
            
            # Category analysis if applicable
            if 'category' in results_df.columns and results_df['category'].nunique() > 1:
                charts['category_analysis.html'] = self.dashboard.create_category_analysis(results_df)
            
            # Save individual charts
            for filename, chart in charts.items():
                if chart:
                    chart_path = os.path.join(output_dir, filename)
                    chart.write_html(chart_path)
                    print(f"   ✅ {filename}")
            
            # Generate comprehensive report
            report_path = self.dashboard.save_comprehensive_report(results_df, output_dir)
            if report_path:
                print(f"\n🌐 Interactive dashboard: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Some visualizations failed: {e}")
        
        print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
        return summary_table
    
    def _display_key_insights(self, results_df: pd.DataFrame, summary_table: pd.DataFrame):
        """Display key insights from the analysis"""
        
        n_decks = len(results_df)
        avg_score = results_df['composite_score'].mean()
        
        print(f"\n🏆 TOP 3 PERFORMERS:")
        top_3 = summary_table.head(3)
        for _, deck in top_3.iterrows():
            print(f"   {deck['rank']}. {deck['deck_name']} - {deck['composite_score']}/10")
            if 'investability_insight' in deck:
                print(f"      💡 {deck['investability_insight']}")
        
        if n_decks > 3:
            print(f"\n⚠️ NEEDS IMPROVEMENT:")
            bottom_3 = summary_table.tail(min(3, n_decks-3))
            for _, deck in bottom_3.iterrows():
                print(f"   {deck['rank']}. {deck['deck_name']} - {deck['composite_score']}/10")
        
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"   Average Score: {avg_score:.2f}/10")
        print(f"   Standard Deviation: {results_df['composite_score'].std():.2f}")
        print(f"   Median Score: {results_df['composite_score'].median():.2f}/10")
        print(f"   Score Range: {results_df['composite_score'].min():.2f} - {results_df['composite_score'].max():.2f}")
        
        # Dimension averages
        dimensions = ['problem_clarity', 'market_potential', 'traction_strength', 
                     'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
        
        print(f"\n📊 DIMENSION AVERAGES:")
        for dim in dimensions:
            if dim in results_df.columns:
                avg_score = results_df[dim].mean()
                print(f"   {dim.replace('_', ' ').title()}: {avg_score:.2f}/10")
        
        # Category insights
        if 'category' in results_df.columns:
            print(f"\n🏷️ CATEGORY DISTRIBUTION:")
            category_counts = results_df['category'].value_counts()
            for category, count in category_counts.head(5).items():
                percentage = (count / n_decks) * 100
                print(f"   {category}: {count} decks ({percentage:.1f}%)")

# Real-time monitoring classes
class PDFHandler(FileSystemEventHandler):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.processed_files = set()
    
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return
        
        # Wait for file to be fully written
        time.sleep(3)
        self.process_new_pdf(event.src_path)
    
    def process_new_pdf(self, pdf_path):
        if pdf_path in self.processed_files:
            return
        
        self.processed_files.add(pdf_path)
        
        print(f"\n🆕 NEW PDF DETECTED: {os.path.basename(pdf_path)}")
        print("=" * 50)
        
        result = self.evaluator.process_single_pdf(pdf_path)
        
        if 'error' not in result:
            # Display results
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   Score: {result['composite_score']:.2f}/10")
            print(f"   Category: {result['category']}")
            print(f"   Processing Time: {result['processing_time']:.1f}s")
            print(f"   💡 {result['investability_insight']}")
            
            # Save individual result
            self._save_individual_result(result)
        else:
            print(f"❌ Processing failed: {result['error']}")
    
    def _save_individual_result(self, result):
        """Save individual result to CSV"""
        os.makedirs("real_time_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}_{result['deck_name']}.csv"
        filepath = os.path.join("real_time_results", filename)
        
        pd.DataFrame([result]).to_csv(filepath, index=False)
        print(f"💾 Results saved: {filepath}")

class RealTimeMonitor:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.observer = Observer()
    
    def start_monitoring(self, watch_directory: str):
        """Start monitoring directory for new PDFs"""
        if not os.path.exists(watch_directory):
            os.makedirs(watch_directory)
            print(f"📁 Created watch directory: {watch_directory}")
        
        print(f"\n👁️ REAL-TIME MONITORING ACTIVE")
        print(f"📂 Watching: {watch_directory}")
        print(f"📋 Drop PDF files here for automatic analysis")
        print(f"⏹️ Press Ctrl+C to stop")
        print("=" * 60)
        
        handler = PDFHandler(self.evaluator)
        self.observer.schedule(handler, watch_directory, recursive=False)
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time monitor...")
            self.observer.stop()
        
        self.observer.join()
        print("✅ Real-time monitoring stopped")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Startup Pitch Deck Evaluator")
    parser.add_argument('--api-key', help='Gemini API key')
    parser.add_argument('--no-diagnose', action='store_true', help='Skip PDF diagnosis')
    parser.add_argument('--batch', help='Directory for batch processing')
    parser.add_argument('--monitor', help='Directory for real-time monitoring')
    
    args = parser.parse_args()
    
    print("🚀 STARTUP PITCH DECK EVALUATOR")
    print("Advanced NLP & AI Analysis System")
    print("=" * 50)
    
    # Get API key
    api_key = args.api_key or input("Enter your Gemini API key (or set GEMINI_API_KEY env var): ").strip()
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ Gemini API key is required!")
            print("   Get one at: https://makersuite.google.com/app/apikey")
            return 1
    
    # Initialize system
    try:
        evaluator = StartupPitchEvaluator(api_key=api_key)
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return 1
    
    # Handle batch processing argument
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ Directory not found: {args.batch}")
            return 1
        
        results_df = evaluator.process_pitch_decks(args.batch, diagnose_first=not args.no_diagnose)
        
        if not results_df.empty:
            evaluator.generate_comprehensive_analysis(results_df)
            print(f"\n🎉 Batch processing complete! {len(results_df)} decks analyzed")
        return 0
    
    # Handle real-time monitoring argument
    if args.monitor:
        monitor = RealTimeMonitor(evaluator)
        monitor.start_monitoring(args.monitor)
        return 0
    
    # Interactive mode
    while True:
        print(f"\n📋 SELECT MODE:")
        print("1. 📁 Batch process existing PDFs")
        print("2. 👁️ Real-time PDF monitoring") 
        print("3. 🔍 Diagnose PDF directory only")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            pdf_dir = input("Enter directory containing PDFs: ").strip()
            if not pdf_dir or not os.path.exists(pdf_dir):
                print("❌ Invalid directory!")
                continue
            
            results_df = evaluator.process_pitch_decks(pdf_dir, diagnose_first=not args.no_diagnose)
            
            if not results_df.empty:
                evaluator.generate_comprehensive_analysis(results_df)
                print(f"\n🎉 Analysis complete! {len(results_df)} decks processed")
            else:
                print("❌ No decks were successfully processed")
        
        elif choice == "2":
            watch_dir = input("Enter directory to monitor (default: ./pdf_watch): ").strip()
            if not watch_dir:
                watch_dir = "./pdf_watch"
            
            monitor = RealTimeMonitor(evaluator)
            monitor.start_monitoring(watch_dir)
        
        elif choice == "3":
            pdf_dir = input("Enter directory to diagnose: ").strip()
            if pdf_dir and os.path.exists(pdf_dir):
                evaluator.diagnose_pdf_directory(pdf_dir)
            else:
                print("❌ Invalid directory!")
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    sys.exit(main())
