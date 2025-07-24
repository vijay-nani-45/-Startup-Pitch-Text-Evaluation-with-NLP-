# real_time_processor.py - Fixed version without circular import
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from datetime import datetime
from evaluator_core import StartupPitchEvaluator  # Import from core module

class PDFHandler(FileSystemEventHandler):
    def __init__(self, evaluator, output_callback=None):
        self.evaluator = evaluator
        self.output_callback = output_callback
        self.processed_files = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.lower().endswith('.pdf'):
            # Wait a moment for file to be fully written
            time.sleep(2)
            self.process_new_pdf(event.src_path)
    
    def process_new_pdf(self, pdf_path):
        if pdf_path in self.processed_files:
            return
        
        self.processed_files.add(pdf_path)
        print(f"\n🆕 New PDF detected: {os.path.basename(pdf_path)}")
        
        try:
            # Process the single file
            result = self.evaluator.process_single_pdf(pdf_path)
            
            if 'error' not in result:
                print(f"✅ Analysis complete!")
                print(f"📊 Composite Score: {result['composite_score']:.2f}/10")
                print(f"🏷️ Category: {result['category']}")
                print(f"💡 Insight: {result['investability_insight']}")
                
                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_{timestamp}_{os.path.basename(pdf_path).replace('.pdf', '')}.csv"
                result_path = os.path.join("real_time_results", filename)
                
                os.makedirs("real_time_results", exist_ok=True)
                pd.DataFrame([result]).to_csv(result_path, index=False)
                print(f"💾 Results saved: {result_path}")
                
                if self.output_callback:
                    self.output_callback(result)
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")

class RealTimePDFAnalyzer:
    def __init__(self, api_key: str):
        self.evaluator = StartupPitchEvaluator(api_key=api_key)
        self.observer = Observer()
        
    def start_monitoring(self, watch_directory: str):
        """Start monitoring directory for new PDFs"""
        if not os.path.exists(watch_directory):
            os.makedirs(watch_directory)
            print(f"📁 Created watch directory: {watch_directory}")
        
        print(f"👁️ Starting real-time monitoring of: {watch_directory}")
        print("📋 Drop PDF files into this directory for automatic analysis")
        print("⏹️ Press Ctrl+C to stop monitoring\n")
        
        handler = PDFHandler(self.evaluator)
        self.observer.schedule(handler, watch_directory, recursive=False)
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time analyzer...")
            self.observer.stop()
        
        self.observer.join()
