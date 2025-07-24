# # streamlit_app.py - Fixed version with proper type handling
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import os
# import tempfile
# import time
# from datetime import datetime
# import sys
# import json

# # Import our enhanced modules
# from config import Config
# from pdf_extractor import EnhancedPDFExtractor
# from nlp_scorer import EnhancedNLPScorer
# from gemini_analyzer import GeminiPitchAnalyzer
# from dashboard import EnhancedDashboard

# # Page configuration
# st.set_page_config(
#     page_title="🚀 Pitch Deck Evaluator",
#     page_icon="🚀",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         text-align: center;
#     }
#     .score-good { color: #28a745; font-weight: bold; }
#     .score-medium { color: #ffc107; font-weight: bold; }
#     .score-poor { color: #dc3545; font-weight: bold; }
#     .stTab [data-baseweb="tab-list"] {
#         gap: 2px;
#     }
#     .processing-status {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 5px;
#         border-left: 4px solid #007bff;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'evaluator' not in st.session_state:
#     st.session_state.evaluator = None
# if 'results_history' not in st.session_state:
#     st.session_state.results_history = []
# if 'processing' not in st.session_state:
#     st.session_state.processing = False

# def safe_float_conversion(value, default=0.0):
#     """Safely convert any value to float"""
#     try:
#         if value is None:
#             return default
#         if isinstance(value, (int, float)):
#             return float(value)
#         if isinstance(value, str):
#             # Remove any non-numeric characters except decimal point and minus
#             import re
#             cleaned = re.sub(r'[^\d\.-]', '', str(value))
#             if cleaned:
#                 return float(cleaned)
#             return default
#         return float(value)
#     except (ValueError, TypeError):
#         return default

# def safe_int_conversion(value, default=0):
#     """Safely convert any value to int"""
#     try:
#         return int(safe_float_conversion(value, default))
#     except (ValueError, TypeError):
#         return default

# def normalize_result_data(result):
#     """Normalize all numeric fields in result data"""
#     if not isinstance(result, dict):
#         return result
    
#     # Score fields that should be numeric
#     score_fields = [
#         'composite_score', 'problem_clarity', 'market_potential', 'traction_strength',
#         'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
#     ]
    
#     # Convert score fields to float
#     for field in score_fields:
#         if field in result:
#             result[field] = safe_float_conversion(result[field])
    
#     # Convert word_count to int
#     if 'word_count' in result:
#         result['word_count'] = safe_int_conversion(result['word_count'])
    
#     # Ensure string fields are strings
#     string_fields = ['deck_name', 'category', 'summary', 'investability_insight', 
#                     'competitive_analysis', 'extraction_method']
    
#     for field in string_fields:
#         if field in result:
#             result[field] = str(result[field]) if result[field] is not None else ""
    
#     return result

# class StreamlitPitchEvaluator:
#     def __init__(self, api_key: str):
#         """Initialize the Streamlit evaluator"""
#         self.config = Config()
        
#         try:
#             self.extractor = EnhancedPDFExtractor()
#             self.scorer = EnhancedNLPScorer()
#             self.gemini_analyzer = GeminiPitchAnalyzer(api_key=api_key)
#             self.dashboard = EnhancedDashboard()
            
#         except Exception as e:
#             st.error(f"Failed to initialize system: {e}")
#             st.stop()
    
#     def process_uploaded_file(self, uploaded_file) -> dict:
#         """Process a single uploaded PDF file"""
        
#         # Create temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name
        
#         try:
#             # Extract deck data
#             deck_name = uploaded_file.name.replace('.pdf', '')
#             deck_data = self.extractor.extract_deck_data(tmp_file_path, deck_name)
            
#             if 'error' in deck_data:
#                 return {'error': deck_data['error'], 'deck_name': deck_name}
            
#             # Score the deck
#             scores = self.scorer.score_deck(deck_data)
            
#             # Ensure scores are numeric
#             scores = {k: safe_float_conversion(v) for k, v in scores.items()}
            
#             # Gemini analysis with error handling
#             try:
#                 summary = self.gemini_analyzer.generate_deck_summary(deck_data)
#                 category = self.gemini_analyzer.classify_startup_category(deck_data)
#                 insight = self.gemini_analyzer.generate_investability_insight(scores, deck_data)
#                 competitive_analysis = self.gemini_analyzer.analyze_competitive_landscape(deck_data)
#             except Exception as e:
#                 st.warning(f"AI analysis partially failed: {e}")
#                 summary = self.gemini_analyzer._generate_fallback_summary(deck_data)
#                 category = self.gemini_analyzer._classify_fallback(deck_data)
#                 insight = self.gemini_analyzer._generate_fallback_insight(scores.get('composite_score', 0), category)
#                 competitive_analysis = "Competitive analysis not available"
            
#             # Combine results
#             result = {
#                 'deck_name': str(deck_name),
#                 'word_count': safe_int_conversion(deck_data.get('word_count', 0)),
#                 'extraction_method': str(deck_data.get('extraction_method', 'unknown')),
#                 'category': str(category),
#                 'summary': str(summary),
#                 'investability_insight': str(insight),
#                 'competitive_analysis': str(competitive_analysis),
#                 'timestamp': datetime.now(),
#                 **scores,
#                 **{k: str(v) for k, v in deck_data.items() if k not in ['word_count', 'extraction_method']}
#             }
            
#             # Normalize the result data
#             result = normalize_result_data(result)
            
#             return result
            
#         except Exception as e:
#             return {'error': str(e), 'deck_name': uploaded_file.name}
        
#         finally:
#             # Clean up temporary file
#             if os.path.exists(tmp_file_path):
#                 os.unlink(tmp_file_path)

# def get_score_color_class(score):
#     """Get CSS class based on score - with safe type conversion"""
#     try:
#         score_float = safe_float_conversion(score)
#         if score_float >= 7:
#             return "score-good"
#         elif score_float >= 5:
#             return "score-medium"
#         else:
#             return "score-poor"
#     except:
#         return "score-poor"

# def display_score_card(title, score, max_score=10):
#     """Display a score card with color coding - with safe type conversion"""
#     try:
#         score_float = safe_float_conversion(score)
#         max_score_float = safe_float_conversion(max_score, 10)
#         color_class = get_score_color_class(score_float)
        
#         st.markdown(f"""
#         <div class="metric-card">
#             <h4>{title}</h4>
#             <h2 class="{color_class}">{score_float:.2f}/{max_score_float}</h2>
#         </div>
#         """, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error displaying score card: {e}")

# def main():
#     # Header
#     st.markdown("""
#     <div class="main-header">
#         <h1>🚀 Startup Pitch Deck Evaluator</h1>
#         <p>Advanced AI-Powered Analysis with Gemini Integration</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar configuration
#     st.sidebar.header("🔧 Configuration")
    
#     # API key input
#     api_key = st.sidebar.text_input(
#         "Gemini API Key", 
#         type="password",
#         help="Get your API key from https://makersuite.google.com/app/apikey"
#     )
    
#     if not api_key:
#         api_key = os.getenv('GEMINI_API_KEY')
    
#     if not api_key:
#         st.sidebar.error("Please enter your Gemini API key")
#         st.info("🔑 **Get Started:**\n1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Create a new API key\n3. Enter it in the sidebar")
#         return
    
#     # Initialize evaluator
#     if st.session_state.evaluator is None:
#         with st.spinner("Initializing AI analysis system..."):
#             try:
#                 st.session_state.evaluator = StreamlitPitchEvaluator(api_key)
#                 st.sidebar.success("✅ System initialized!")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Initialization failed: {e}")
#                 return
    
#     evaluator = st.session_state.evaluator
    
#     # Sidebar options
#     st.sidebar.header("📊 Analysis Options")
    
#     # Analysis mode
#     analysis_mode = st.sidebar.selectbox(
#         "Select Analysis Mode",
#         ["Single PDF Analysis", "Batch Analysis", "Results History"]
#     )
    
#     # Main content based on mode
#     if analysis_mode == "Single PDF Analysis":
#         single_pdf_analysis(evaluator)
#     elif analysis_mode == "Batch Analysis":
#         batch_analysis(evaluator)
#     elif analysis_mode == "Results History":
#         results_history()

# def single_pdf_analysis(evaluator):
#     """Single PDF analysis interface"""
    
#     st.header("📄 Single PDF Analysis")
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload a pitch deck PDF",
#         type="pdf",
#         help="Upload a PDF file of your startup pitch deck for comprehensive analysis"
#     )
    
#     if uploaded_file is not None:
#         # Display file info
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("File Name", uploaded_file.name)
#         with col2:
#             st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
#         with col3:
#             if st.button("🔍 Analyze Pitch Deck", type="primary"):
#                 st.session_state.processing = True
        
#         # Process file if button clicked
#         if st.session_state.processing:
#             analyze_single_pdf(evaluator, uploaded_file)

# def analyze_single_pdf(evaluator, uploaded_file):
#     """Analyze a single PDF and display results"""
    
#     # Processing status
#     status_placeholder = st.empty()
#     progress_bar = st.progress(0)
    
#     try:
#         # Step 1: PDF Extraction
#         status_placeholder.markdown("""
#         <div class="processing-status">
#             🔄 <strong>Step 1/4:</strong> Extracting text from PDF...
#         </div>
#         """, unsafe_allow_html=True)
#         progress_bar.progress(25)
#         time.sleep(0.5)
        
#         # Step 2: NLP Analysis
#         status_placeholder.markdown("""
#         <div class="processing-status">
#             🧠 <strong>Step 2/4:</strong> Analyzing content with NLP...
#         </div>
#         """, unsafe_allow_html=True)
#         progress_bar.progress(50)
#         time.sleep(0.5)
        
#         # Step 3: AI Analysis
#         status_placeholder.markdown("""
#         <div class="processing-status">
#             🤖 <strong>Step 3/4:</strong> Generating AI insights with Gemini...
#         </div>
#         """, unsafe_allow_html=True)
#         progress_bar.progress(75)
        
#         # Process the file
#         result = evaluator.process_uploaded_file(uploaded_file)
        
#         # Step 4: Complete
#         status_placeholder.markdown("""
#         <div class="processing-status">
#             ✅ <strong>Step 4/4:</strong> Analysis complete!
#         </div>
#         """, unsafe_allow_html=True)
#         progress_bar.progress(100)
#         time.sleep(0.5)
        
#         # Clear processing status
#         status_placeholder.empty()
#         progress_bar.empty()
        
#         if 'error' in result:
#             st.error(f"❌ Analysis failed: {result['error']}")
#         else:
#             # Normalize result data before adding to history
#             result = normalize_result_data(result)
            
#             # Add to history
#             st.session_state.results_history.append(result)
            
#             # Display results
#             display_analysis_results(result)
            
#     except Exception as e:
#         st.error(f"❌ Unexpected error: {e}")
#         st.error("Please check your input file and try again.")
    
#     finally:
#         st.session_state.processing = False

# def display_analysis_results(result):
#     """Display comprehensive analysis results"""
    
#     st.success("🎉 Analysis Complete!")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     # Overview metrics
#     st.subheader("📊 Overview")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         display_score_card("Composite Score", result.get('composite_score', 0))
#     with col2:
#         st.metric("Category", result.get('category', 'Unknown'))
#     with col3:
#         st.metric("Word Count", f"{safe_int_conversion(result.get('word_count', 0)):,}")
#     with col4:
#         st.metric("Extraction Method", result.get('extraction_method', 'Unknown'))
    
#     # Detailed analysis tabs
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📈 Scores", "💡 AI Insights", "📋 Content Analysis", 
#         "📊 Visualizations", "📥 Export"
#     ])
    
#     with tab1:
#         display_detailed_scores(result)
    
#     with tab2:
#         display_ai_insights(result)
    
#     with tab3:
#         display_content_analysis(result)
    
#     with tab4:
#         display_visualizations(result)
    
#     with tab5:
#         display_export_options(result)

# def display_detailed_scores(result):
#     """Display detailed scoring breakdown"""
    
#     st.subheader("🎯 Detailed Scoring Breakdown")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     # Score dimensions
#     dimensions = [
#         ('problem_clarity', 'Problem Clarity', 'How well-defined is the problem?'),
#         ('market_potential', 'Market Potential', 'Size and growth of target market'),
#         ('traction_strength', 'Traction Strength', 'Evidence of customer validation'),
#         ('team_experience', 'Team Experience', 'Background and expertise of team'),
#         ('business_model', 'Business Model', 'Revenue model clarity'),
#         ('vision_moat', 'Vision & Moat', 'Competitive advantages'),
#         ('overall_confidence', 'Overall Confidence', 'Professional presentation')
#     ]
    
#     # Display scores in grid
#     col1, col2 = st.columns(2)
    
#     for i, (key, title, description) in enumerate(dimensions):
#         score = safe_float_conversion(result.get(key, 0))
        
#         with col1 if i % 2 == 0 else col2:
#             st.markdown(f"**{title}**")
#             st.markdown(f"*{description}*")
            
#             # Progress bar with color
#             if score >= 7:
#                 color = "#28a745"
#             elif score >= 5:
#                 color = "#ffc107"
#             else:
#                 color = "#dc3545"
                
#             st.markdown(f"""
#             <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
#                 <div style="background-color: {color}; width: {min(score*10, 100)}%; height: 20px; 
#                      border-radius: 10px; display: flex; align-items: center; 
#                      justify-content: center; color: white; font-weight: bold;">
#                     {score:.1f}/10
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#             st.markdown("---")
    
#     # Radar chart
#     st.subheader("🎯 Performance Radar")
    
#     try:
#         # Create radar chart data
#         categories = [dim[1] for dim in dimensions]
#         scores = [safe_float_conversion(result.get(dim[0], 0)) for dim in dimensions]
        
#         fig = go.Figure()
        
#         fig.add_trace(go.Scatterpolar(
#             r=scores,
#             theta=categories,
#             fill='toself',
#             name=result.get('deck_name', 'Pitch Deck'),
#             line=dict(color='#667eea', width=3),
#             fillcolor='rgba(102, 126, 234, 0.25)'
#         ))
        
#         fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[0, 10]
#                 )),
#             showlegend=False,
#             height=500
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#     except Exception as e:
#         st.error(f"Error creating radar chart: {e}")

# def display_ai_insights(result):
#     """Display AI-generated insights"""
    
#     st.subheader("🤖 AI-Generated Insights")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     # Investability insight
#     st.markdown("### 💰 Investment Readiness")
#     st.info(result.get('investability_insight', 'Insight not available'))
    
#     # Summary
#     st.markdown("### 📋 Executive Summary")
#     summary = result.get('summary', 'Summary not available')
#     summary_lines = summary.split('\n')
#     for line in summary_lines:
#         if line.strip():
#             st.markdown(line)
    
#     # Competitive analysis
#     comp_analysis = result.get('competitive_analysis', '')
#     if comp_analysis and comp_analysis != "Analysis not available":
#         st.markdown("### 🏆 Competitive Landscape")
#         st.markdown(comp_analysis)
    
#     # Recommendations based on weak areas
#     st.markdown("### 🎯 Improvement Recommendations")
    
#     # Find weak areas
#     dimensions = ['problem_clarity', 'market_potential', 'traction_strength', 
#                  'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
    
#     weak_areas = []
#     strong_areas = []
    
#     for dim in dimensions:
#         score = safe_float_conversion(result.get(dim, 0))
#         if score < 5:
#             weak_areas.append(dim.replace('_', ' ').title())
#         elif score >= 8:
#             strong_areas.append(dim.replace('_', ' ').title())
    
#     if strong_areas:
#         st.success(f"**Strengths:** {', '.join(strong_areas)}")
    
#     if weak_areas:
#         st.warning(f"**Areas for Improvement:** {', '.join(weak_areas)}")
        
#         # Specific recommendations
#         recommendations = {
#             'Problem Clarity': "Clearly articulate the specific problem with quantified impact",
#             'Market Potential': "Include market size data (TAM/SAM/SOM) and growth projections",
#             'Traction Strength': "Add specific metrics, growth rates, and customer testimonials",
#             'Team Experience': "Highlight relevant experience, previous successes, and key skills",
#             'Business Model': "Clarify revenue streams, pricing strategy, and unit economics",
#             'Vision Moat': "Define competitive advantages and defensible market position",
#             'Overall Confidence': "Improve presentation quality and professional tone"
#         }
        
#         for area in weak_areas:
#             if area in recommendations:
#                 st.markdown(f"- **{area}:** {recommendations[area]}")
#     else:
#         st.success("🎉 Strong performance across all dimensions!")

# def display_content_analysis(result):
#     """Display content analysis by sections"""
    
#     st.subheader("📖 Content Analysis by Section")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     sections = [
#         ('problem', 'Problem Statement', '❗'),
#         ('solution', 'Solution Description', '💡'),
#         ('market', 'Market Analysis', '📈'),
#         ('traction', 'Traction & Metrics', '🚀'),
#         ('team', 'Team Information', '👥'),
#         ('business_model', 'Business Model', '💼'),
#         ('ask', 'Funding Ask', '💰')
#     ]
    
#     for section_key, section_title, icon in sections:
#         content = result.get(section_key, '')
        
#         if content and len(str(content).strip()) > 20:
#             with st.expander(f"{icon} {section_title}", expanded=False):
#                 content_str = str(content)
#                 display_content = content_str[:500] + "..." if len(content_str) > 500 else content_str
#                 st.write(display_content)
                
#                 # Word count for section
#                 word_count = len(content_str.split())
#                 st.caption(f"Word count: {word_count}")
#         else:
#             st.warning(f"{icon} {section_title}: No significant content detected")

# def display_visualizations(result):
#     """Display additional visualizations"""
    
#     st.subheader("📊 Additional Visualizations")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     try:
#         # Score comparison with ideal startup
#         st.markdown("### 🎯 Score vs Ideal Startup")
        
#         dimensions = ['Problem Clarity', 'Market Potential', 'Traction Strength', 
#                      'Team Experience', 'Business Model', 'Vision & Moat', 'Overall Confidence']
        
#         actual_scores = [
#             safe_float_conversion(result.get('problem_clarity', 0)),
#             safe_float_conversion(result.get('market_potential', 0)),
#             safe_float_conversion(result.get('traction_strength', 0)),
#             safe_float_conversion(result.get('team_experience', 0)),
#             safe_float_conversion(result.get('business_model', 0)),
#             safe_float_conversion(result.get('vision_moat', 0)),
#             safe_float_conversion(result.get('overall_confidence', 0))
#         ]
        
#         ideal_scores = [8.5, 8.0, 7.5, 8.0, 7.5, 6.5, 8.0]  # Benchmarks
        
#         fig = go.Figure()
        
#         fig.add_trace(go.Bar(
#             name='Your Pitch',
#             x=dimensions,
#             y=actual_scores,
#             marker_color='#667eea'
#         ))
        
#         fig.add_trace(go.Bar(
#             name='Investment-Ready Benchmark',
#             x=dimensions,
#             y=ideal_scores,
#             marker_color='#28a745',
#             opacity=0.7
#         ))
        
#         fig.update_layout(
#             title="Performance vs Investment-Ready Benchmark",
#             yaxis_title="Score (0-10)",
#             barmode='group',
#             height=400
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#     except Exception as e:
#         st.error(f"Error creating visualizations: {e}")
    
#     # Historical performance if available
#     if len(st.session_state.results_history) > 1:
#         st.markdown("### 📊 Historical Performance")
        
#         try:
#             # Create DataFrame from history with proper type conversion
#             history_data = []
#             for i, hist_result in enumerate(st.session_state.results_history):
#                 hist_result = normalize_result_data(hist_result)
#                 history_data.append({
#                     'analysis_num': i + 1,
#                     'composite_score': safe_float_conversion(hist_result.get('composite_score', 0)),
#                     'deck_name': hist_result.get('deck_name', f'Deck {i+1}')
#                 })
            
#             history_df = pd.DataFrame(history_data)
            
#             if len(history_df) >= 2:
#                 fig = px.line(
#                     history_df, 
#                     x='analysis_num', 
#                     y='composite_score',
#                     title='Score Progression Over Time',
#                     markers=True,
#                     hover_data=['deck_name']
#                 )
#                 fig.update_xaxis(title="Analysis #")
#                 fig.update_yaxis(title="Composite Score", range=[0, 10])
                
#                 st.plotly_chart(fig, use_container_width=True)
                
#         except Exception as e:
#             st.error(f"Error creating historical chart: {e}")

# def display_export_options(result):
#     """Display export and download options"""
    
#     st.subheader("📥 Export Options")
    
#     # Normalize result data
#     result = normalize_result_data(result)
    
#     # Create export data with safe conversions
#     export_data = {
#         'Metric': ['Composite Score', 'Problem Clarity', 'Market Potential', 
#                   'Traction Strength', 'Team Experience', 'Business Model', 
#                   'Vision & Moat', 'Overall Confidence'],
#         'Score': [
#             safe_float_conversion(result.get('composite_score', 0)),
#             safe_float_conversion(result.get('problem_clarity', 0)),
#             safe_float_conversion(result.get('market_potential', 0)),
#             safe_float_conversion(result.get('traction_strength', 0)),
#             safe_float_conversion(result.get('team_experience', 0)),
#             safe_float_conversion(result.get('business_model', 0)),
#             safe_float_conversion(result.get('vision_moat', 0)),
#             safe_float_conversion(result.get('overall_confidence', 0))
#         ]
#     }
    
#     export_df = pd.DataFrame(export_data)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # CSV download
#         csv = export_df.to_csv(index=False)
#         st.download_button(
#             label="📊 Download Scores (CSV)",
#             data=csv,
#             file_name=f"{result.get('deck_name', 'pitch_deck')}_analysis.csv",
#             mime="text/csv"
#         )
    
#     with col2:
#         # JSON download
#         # Create a serializable version of the result
#         json_result = {}
#         for key, value in result.items():
#             if key == 'timestamp':
#                 json_result[key] = str(value)
#             else:
#                 json_result[key] = value
        
#         json_data = json.dumps(json_result, indent=2, default=str)
#         st.download_button(
#             label="📋 Download Full Report (JSON)",
#             data=json_data,
#             file_name=f"{result.get('deck_name', 'pitch_deck')}_full_report.json",
#             mime="application/json"
#         )
    
#     # Display summary table
#     st.markdown("### 📋 Summary Table")
#     st.dataframe(export_df, use_container_width=True)
    
#     # Analysis metadata
#     st.markdown("### ℹ️ Analysis Metadata")
    
#     metadata_col1, metadata_col2 = st.columns(2)
    
#     with metadata_col1:
#         st.info(f"""
#         **Analysis Details:**
#         - **File:** {result.get('deck_name', 'Unknown')}
#         - **Category:** {result.get('category', 'Unknown')}
#         - **Word Count:** {safe_int_conversion(result.get('word_count', 0)):,}
#         - **Extraction Method:** {result.get('extraction_method', 'Unknown')}
#         """)
    
#     with metadata_col2:
#         st.info(f"""
#         **Scoring Weights:**
#         - Problem Clarity: 15%
#         - Market Potential: 20%
#         - Traction Strength: 25%
#         - Team Experience: 15%
#         - Business Model: 15%
#         - Vision & Moat: 5%
#         - Overall Confidence: 5%
#         """)

# def batch_analysis(evaluator):
#     """Batch analysis interface"""
    
#     st.header("📁 Batch Analysis")
#     st.info("Upload multiple PDF files for comparative analysis")
    
#     # Multiple file uploader
#     uploaded_files = st.file_uploader(
#         "Upload multiple pitch deck PDFs",
#         type="pdf",
#         accept_multiple_files=True,
#         help="Upload multiple PDF files to compare pitch decks"
#     )
    
#     if uploaded_files:
#         st.write(f"📄 Selected {len(uploaded_files)} files:")
#         for file in uploaded_files:
#             st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
#         if st.button("🔍 Analyze All Pitch Decks", type="primary"):
#             analyze_batch_pdfs(evaluator, uploaded_files)

# def analyze_batch_pdfs(evaluator, uploaded_files):
#     """Analyze multiple PDFs in batch"""
    
#     results = []
#     progress_placeholder = st.empty()
#     progress_bar = st.progress(0)
    
#     for i, uploaded_file in enumerate(uploaded_files):
#         progress_placeholder.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
#         progress_bar.progress((i + 1) / len(uploaded_files))
        
#         result = evaluator.process_uploaded_file(uploaded_file)
        
#         if 'error' not in result:
#             # Normalize result data
#             result = normalize_result_data(result)
#             results.append(result)
#             st.session_state.results_history.append(result)
#         else:
#             st.error(f"Failed to process {uploaded_file.name}: {result['error']}")
    
#     progress_placeholder.empty()
#     progress_bar.empty()
    
#     if results:
#         display_batch_results(results)

# def display_batch_results(results):
#     """Display batch analysis results"""
    
#     st.success(f"🎉 Successfully analyzed {len(results)} pitch decks!")
    
#     # Normalize all results
#     results = [normalize_result_data(result) for result in results]
    
#     # Create DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Ensure numeric columns are properly typed
#     numeric_columns = ['composite_score', 'problem_clarity', 'market_potential', 'traction_strength',
#                       'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
    
#     for col in numeric_columns:
#         if col in results_df.columns:
#             results_df[col] = results_df[col].apply(safe_float_conversion)
    
#     # Summary statistics
#     st.subheader("📊 Batch Analysis Summary")
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Decks", len(results))
#     with col2:
#         avg_score = results_df['composite_score'].mean() if 'composite_score' in results_df.columns else 0
#         st.metric("Average Score", f"{avg_score:.2f}")
#     with col3:
#         max_score = results_df['composite_score'].max() if 'composite_score' in results_df.columns else 0
#         st.metric("Best Score", f"{max_score:.2f}")
#     with col4:
#         if 'composite_score' in results_df.columns and len(results_df) > 0:
#             best_idx = results_df['composite_score'].idxmax()
#             best_deck = results_df.loc[best_idx, 'deck_name'] if 'deck_name' in results_df.columns else "Unknown"
#             st.metric("Top Performer", best_deck)
    
#     # Rankings table
#     st.subheader("🏆 Rankings")
    
#     try:
#         # Prepare ranking table
#         display_columns = ['deck_name', 'composite_score']
#         if 'category' in results_df.columns:
#             display_columns.append('category')
#         if 'investability_insight' in results_df.columns:
#             display_columns.append('investability_insight')
        
#         # Filter to existing columns
#         existing_columns = [col for col in display_columns if col in results_df.columns]
        
#         ranking_df = results_df[existing_columns].copy()
#         ranking_df = ranking_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
#         ranking_df.index += 1
#         ranking_df.index.name = 'Rank'
        
#         st.dataframe(ranking_df, use_container_width=True)
        
#         # Comparative visualizations
#         st.subheader("📈 Comparative Analysis")
        
#         # Score comparison chart
#         if len(ranking_df) > 0:
#             fig = px.bar(
#                 ranking_df.head(10),  # Top 10
#                 x='composite_score',
#                 y='deck_name',
#                 orientation='h',
#                 title='Top 10 Pitch Decks by Score',
#                 color='composite_score',
#                 color_continuous_scale='Viridis'
#             )
#             fig.update_layout(height=500)
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Category distribution
#         if 'category' in results_df.columns:
#             category_counts = results_df['category'].value_counts()
            
#             if len(category_counts) > 0:
#                 fig = px.pie(
#                     values=category_counts.values,
#                     names=category_counts.index,
#                     title='Distribution by Category'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Export batch results
#         st.subheader("📥 Export Batch Results")
        
#         csv = ranking_df.to_csv()
#         st.download_button(
#             label="📊 Download Batch Results (CSV)",
#             data=csv,
#             file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
        
#     except Exception as e:
#         st.error(f"Error displaying batch results: {e}")

# def results_history():
#     """Display results history"""
    
#     st.header("📊 Analysis History")
    
#     if not st.session_state.results_history:
#         st.info("No analysis history available. Analyze some pitch decks first!")
#         return
    
#     try:
#         # Normalize all history data
#         normalized_history = [normalize_result_data(result) for result in st.session_state.results_history]
        
#         # Display history
#         history_df = pd.DataFrame(normalized_history)
        
#         # Ensure numeric columns are properly typed
#         if 'composite_score' in history_df.columns:
#             history_df['composite_score'] = history_df['composite_score'].apply(safe_float_conversion)
        
#         st.subheader(f"📋 History ({len(history_df)} analyses)")
        
#         # Summary metrics
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Analyses", len(history_df))
#         with col2:
#             if 'composite_score' in history_df.columns:
#                 avg_score = history_df['composite_score'].mean()
#                 st.metric("Average Score", f"{avg_score:.2f}")
#         with col3:
#             if 'composite_score' in history_df.columns:
#                 max_score = history_df['composite_score'].max()
#                 st.metric("Best Score", f"{max_score:.2f}")
        
#         # History table
#         display_columns = ['deck_name', 'composite_score']
#         if 'category' in history_df.columns:
#             display_columns.append('category')
#         if 'timestamp' in history_df.columns:
#             display_columns.append('timestamp')
        
#         existing_columns = [col for col in display_columns if col in history_df.columns]
#         display_df = history_df[existing_columns].copy()
        
#         if 'timestamp' in display_df.columns:
#             display_df = display_df.sort_values('timestamp', ascending=False)
        
#         st.dataframe(display_df, use_container_width=True)
        
#         # Trend analysis
#         if len(history_df) > 1 and 'composite_score' in history_df.columns:
#             st.subheader("📈 Score Trends")
            
#             trend_data = history_df.reset_index()
#             trend_data['Analysis Number'] = trend_data.index + 1
            
#             fig = px.line(
#                 trend_data,
#                 x='Analysis Number',
#                 y='composite_score',
#                 title='Score Progression Over Time',
#                 markers=True,
#                 hover_data=['deck_name'] if 'deck_name' in trend_data.columns else None
#             )
#             fig.update_yaxis(title="Composite Score", range=[0, 10])
            
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Clear history button
#         if st.button("🗑️ Clear History", type="secondary"):
#             st.session_state.results_history = []
#             st.success("History cleared!")
#             st.rerun()
            
#     except Exception as e:
#         st.error(f"Error displaying history: {e}")

# if __name__ == "__main__":
#     main()


# streamlit_app.py - Complete Advanced Startup Pitch Deck Evaluator
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import time
from datetime import datetime
import json
import numpy as np

# Import our enhanced modules
from config import Config
from pdf_extractor import EnhancedPDFExtractor
from nlp_scorer import EnhancedNLPScorer
from gemini_analyzer import GeminiPitchAnalyzer
from dashboard import EnhancedDashboard
from advanced_analytics import (
    PitchSuccessPredictor, 
    AdvancedTextAnalyzer, 
    IndustryBenchmarkAnalyzer,
    AIInsightsGenerator
)

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Pitch Deck Evaluator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .score-good { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    .advanced-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .processing-status {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
    .insight-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'advanced_analytics' not in st.session_state:
    st.session_state.advanced_analytics = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False

def safe_float_conversion(value, default=0.0):
    """Safely convert any value to float"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            import re
            cleaned = re.sub(r'[^\d\.-]', '', str(value))
            if cleaned:
                return float(cleaned)
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    """Safely convert any value to int"""
    try:
        return int(safe_float_conversion(value, default))
    except (ValueError, TypeError):
        return default

def normalize_result_data(result):
    """Normalize all numeric fields in result data"""
    if not isinstance(result, dict):
        return result
    
    # Score fields that should be numeric
    score_fields = [
        'composite_score', 'problem_clarity', 'market_potential', 'traction_strength',
        'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
    ]
    
    # Convert score fields to float
    for field in score_fields:
        if field in result:
            result[field] = safe_float_conversion(result[field])
    
    # Convert word_count to int
    if 'word_count' in result:
        result['word_count'] = safe_int_conversion(result['word_count'])
    
    # Ensure string fields are strings
    string_fields = ['deck_name', 'category', 'summary', 'investability_insight', 
                    'competitive_analysis', 'extraction_method']
    
    for field in string_fields:
        if field in result:
            result[field] = str(result[field]) if result[field] is not None else ""
    
    return result

class AdvancedStreamlitEvaluator:
    def __init__(self, api_key: str):
        """Initialize the advanced Streamlit evaluator"""
        self.config = Config()
        
        try:
            self.extractor = EnhancedPDFExtractor()
            self.scorer = EnhancedNLPScorer()
            self.gemini_analyzer = GeminiPitchAnalyzer(api_key=api_key)
            self.dashboard = EnhancedDashboard()
            
            # Initialize advanced analytics
            self.success_predictor = PitchSuccessPredictor()
            self.text_analyzer = AdvancedTextAnalyzer()
            self.benchmark_analyzer = IndustryBenchmarkAnalyzer()
            self.insights_generator = AIInsightsGenerator(self.gemini_analyzer)
            
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            st.stop()
    
    def process_uploaded_file(self, uploaded_file) -> dict:
        """Process a single uploaded PDF file with advanced analytics"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract deck data
            deck_name = uploaded_file.name.replace('.pdf', '')
            deck_data = self.extractor.extract_deck_data(tmp_file_path, deck_name)
            
            if 'error' in deck_data:
                return {'error': deck_data['error'], 'deck_name': deck_name}
            
            # Score the deck
            scores = self.scorer.score_deck(deck_data)
            scores = {k: safe_float_conversion(v) for k, v in scores.items()}
            
            # Gemini analysis with error handling
            try:
                summary = self.gemini_analyzer.generate_deck_summary(deck_data)
                category = self.gemini_analyzer.classify_startup_category(deck_data)
                insight = self.gemini_analyzer.generate_investability_insight(scores, deck_data)
                competitive_analysis = self.gemini_analyzer.analyze_competitive_landscape(deck_data)
            except Exception as e:
                st.warning(f"AI analysis partially failed: {e}")
                summary = self.gemini_analyzer._generate_fallback_summary(deck_data)
                category = self.gemini_analyzer._classify_fallback(deck_data)
                insight = self.gemini_analyzer._generate_fallback_insight(scores.get('composite_score', 0), category)
                competitive_analysis = "Competitive analysis not available"
            
            # Advanced text analysis
            language_analysis = self.text_analyzer.analyze_pitch_language(deck_data)
            
            # Combine results
            result = {
                'deck_name': str(deck_name),
                'word_count': safe_int_conversion(deck_data.get('word_count', 0)),
                'extraction_method': str(deck_data.get('extraction_method', 'unknown')),
                'category': str(category),
                'summary': str(summary),
                'investability_insight': str(insight),
                'competitive_analysis': str(competitive_analysis),
                'timestamp': datetime.now(),
                **scores,
                **language_analysis,
                **{k: str(v) for k, v in deck_data.items() if k not in ['word_count', 'extraction_method']}
            }
            
            # Normalize the result data
            result = normalize_result_data(result)
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'deck_name': uploaded_file.name}
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

def display_score_card(title, score, max_score=10):
    """Display a score card with color coding"""
    try:
        score_float = safe_float_conversion(score)
        max_score_float = safe_float_conversion(max_score, 10)
        
        if score_float >= 7:
            color_class = "score-good"
        elif score_float >= 5:
            color_class = "score-medium"
        else:
            color_class = "score-poor"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2 class="{color_class}">{score_float:.2f}/{max_score_float}</h2>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying score card: {e}")

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Startup Pitch Deck Evaluator</h1>
        <p>Complete AI-Powered Analysis with Machine Learning & Advanced Visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API key input
    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key")
        st.info("🔑 **Get Started:**\n1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Create a new API key\n3. Enter it in the sidebar")
        return
    
    # Initialize evaluator
    if st.session_state.evaluator is None:
        with st.spinner("Initializing advanced AI analysis system..."):
            try:
                st.session_state.evaluator = AdvancedStreamlitEvaluator(api_key)
                st.session_state.advanced_analytics = True
                st.sidebar.success("✅ Advanced system initialized!")
            except Exception as e:
                st.sidebar.error(f"❌ Initialization failed: {e}")
                return
    
    evaluator = st.session_state.evaluator
    
    # Sidebar options
    st.sidebar.header("📊 Analysis Options")
    
    # Analysis mode
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Single PDF Analysis", "Batch Analysis", "Advanced Analytics", "Results History"]
    )
    
    # Main content based on mode
    if analysis_mode == "Single PDF Analysis":
        single_pdf_analysis(evaluator)
    elif analysis_mode == "Batch Analysis":
        batch_analysis(evaluator)
    elif analysis_mode == "Advanced Analytics":
        advanced_analytics_dashboard(evaluator)
    elif analysis_mode == "Results History":
        results_history()

def single_pdf_analysis(evaluator):
    """Single PDF analysis interface with advanced features"""
    
    st.header("📄 Advanced Single PDF Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a pitch deck PDF",
        type="pdf",
        help="Upload a PDF file of your startup pitch deck for comprehensive AI analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            if st.button("🔍 Analyze Pitch Deck", type="primary"):
                st.session_state.processing = True
        
        # Process file if button clicked
        if st.session_state.processing:
            analyze_single_pdf_advanced(evaluator, uploaded_file)

def analyze_single_pdf_advanced(evaluator, uploaded_file):
    """Analyze a single PDF with advanced analytics"""
    
    # Processing status
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Step 1: PDF Extraction
        status_placeholder.markdown("""
        <div class="processing-status">
            🔄 <strong>Step 1/5:</strong> Extracting text from PDF...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Step 2: NLP Analysis
        status_placeholder.markdown("""
        <div class="processing-status">
            🧠 <strong>Step 2/5:</strong> Analyzing content with advanced NLP...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3: AI Analysis
        status_placeholder.markdown("""
        <div class="processing-status">
            🤖 <strong>Step 3/5:</strong> Generating AI insights with Gemini...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(60)
        time.sleep(0.5)
        
        # Step 4: Advanced Analytics
        status_placeholder.markdown("""
        <div class="processing-status">
            🔬 <strong>Step 4/5:</strong> Running advanced analytics and ML models...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(80)
        
        # Process the file
        result = evaluator.process_uploaded_file(uploaded_file)
        
        # Step 5: Complete
        status_placeholder.markdown("""
        <div class="processing-status">
            ✅ <strong>Step 5/5:</strong> Advanced analysis complete!
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear processing status
        status_placeholder.empty()
        progress_bar.empty()
        
        if 'error' in result:
            st.error(f"❌ Analysis failed: {result['error']}")
        else:
            # Normalize result data before adding to history
            result = normalize_result_data(result)
            
            # Add to history
            st.session_state.results_history.append(result)
            
            # Display advanced results
            display_advanced_analysis_results(result, evaluator)
            
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.error("Please check your input file and try again.")
    
    finally:
        st.session_state.processing = False

def display_advanced_analysis_results(result, evaluator):
    """Display comprehensive advanced analysis results"""
    
    st.success("🎉 Advanced Analysis Complete!")
    
    # Normalize result data
    result = normalize_result_data(result)
    
    # Overview metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        display_score_card("Composite Score", result.get('composite_score', 0))
    with col2:
        st.metric("Category", result.get('category', 'Unknown'))
    with col3:
        st.metric("Word Count", f"{safe_int_conversion(result.get('word_count', 0)):,}")
    with col4:
        st.metric("Tech Sophistication", f"{result.get('tech_sophistication', 0):.1f}")
    with col5:
        st.metric("Confidence Score", f"{result.get('confidence_score', 0):.1f}")
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Scores", "💡 AI Insights", "🔬 Advanced NLP", "📊 Visualizations", 
        "🎯 ML Predictions", "📋 Content Analysis", "📥 Export"
    ])
    
    with tab1:
        display_detailed_scores(result)
    
    with tab2:
        display_ai_insights(result)
    
    with tab3:
        display_advanced_nlp_analysis(result)
    
    with tab4:
        display_single_deck_visualizations(result)
    
    with tab5:
        display_ml_predictions(result, evaluator)
    
    with tab6:
        display_content_analysis(result)
    
    with tab7:
        display_export_options(result)

def display_detailed_scores(result):
    """Display detailed scoring breakdown"""
    
    st.subheader("🎯 Detailed Scoring Breakdown")
    
    # Normalize result data
    result = normalize_result_data(result)
    
    # Score dimensions
    dimensions = [
        ('problem_clarity', 'Problem Clarity', 'How well-defined is the problem?'),
        ('market_potential', 'Market Potential', 'Size and growth of target market'),
        ('traction_strength', 'Traction Strength', 'Evidence of customer validation'),
        ('team_experience', 'Team Experience', 'Background and expertise of team'),
        ('business_model', 'Business Model', 'Revenue model clarity'),
        ('vision_moat', 'Vision & Moat', 'Competitive advantages'),
        ('overall_confidence', 'Overall Confidence', 'Professional presentation')
    ]
    
    # Display scores in grid
    col1, col2 = st.columns(2)
    
    for i, (key, title, description) in enumerate(dimensions):
        score = safe_float_conversion(result.get(key, 0))
        
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"**{title}**")
            st.markdown(f"*{description}*")
            
            # Progress bar with color
            if score >= 7:
                color = "#28a745"
            elif score >= 5:
                color = "#ffc107"
            else:
                color = "#dc3545"
                
            st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                <div style="background-color: {color}; width: {min(score*10, 100)}%; height: 20px; 
                     border-radius: 10px; display: flex; align-items: center; 
                     justify-content: center; color: white; font-weight: bold;">
                    {score:.1f}/10
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
    
    # Radar chart
    st.subheader("🎯 Performance Radar")
    
    try:
        # Create radar chart data
        categories = [dim[1] for dim in dimensions]
        scores = [safe_float_conversion(result.get(dim[0], 0)) for dim in dimensions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=result.get('deck_name', 'Pitch Deck'),
            line=dict(color='#667eea', width=3),
            fillcolor='rgba(102, 126, 234, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")

def display_ai_insights(result):
    """Display AI-generated insights"""
    
    st.subheader("🤖 AI-Generated Insights")
    
    # Normalize result data
    result = normalize_result_data(result)
    
    # Investability insight
    st.markdown("### 💰 Investment Readiness")
    st.info(result.get('investability_insight', 'Insight not available'))
    
    # Summary
    st.markdown("### 📋 Executive Summary")
    summary = result.get('summary', 'Summary not available')
    summary_lines = summary.split('\n')
    for line in summary_lines:
        if line.strip():
            st.markdown(line)
    
    # Competitive analysis
    comp_analysis = result.get('competitive_analysis', '')
    if comp_analysis and comp_analysis != "Analysis not available":
        st.markdown("### 🏆 Competitive Landscape")
        st.markdown(comp_analysis)
    
    # Recommendations based on weak areas
    st.markdown("### 🎯 Improvement Recommendations")
    
    # Find weak areas
    dimensions = ['problem_clarity', 'market_potential', 'traction_strength', 
                 'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
    
    weak_areas = []
    strong_areas = []
    
    for dim in dimensions:
        score = safe_float_conversion(result.get(dim, 0))
        if score < 5:
            weak_areas.append(dim.replace('_', ' ').title())
        elif score >= 8:
            strong_areas.append(dim.replace('_', ' ').title())
    
    if strong_areas:
        st.success(f"**Strengths:** {', '.join(strong_areas)}")
    
    if weak_areas:
        st.warning(f"**Areas for Improvement:** {', '.join(weak_areas)}")
        
        # Specific recommendations
        recommendations = {
            'Problem Clarity': "Clearly articulate the specific problem with quantified impact",
            'Market Potential': "Include market size data (TAM/SAM/SOM) and growth projections",
            'Traction Strength': "Add specific metrics, growth rates, and customer testimonials",
            'Team Experience': "Highlight relevant experience, previous successes, and key skills",
            'Business Model': "Clarify revenue streams, pricing strategy, and unit economics",
            'Vision Moat': "Define competitive advantages and defensible market position",
            'Overall Confidence': "Improve presentation quality and professional tone"
        }
        
        for area in weak_areas:
            if area in recommendations:
                st.markdown(f"- **{area}:** {recommendations[area]}")
    else:
        st.success("🎉 Strong performance across all dimensions!")

def display_advanced_nlp_analysis(result):
    """Display advanced NLP analysis results"""
    
    st.subheader("🔬 Advanced Natural Language Processing Analysis")
    
    # Language sophistication metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Vocabulary Diversity", 
            f"{result.get('vocabulary_diversity', 0):.3f}",
            help="Ratio of unique words to total words - higher indicates more varied vocabulary"
        )
    
    with col2:
        st.metric(
            "Average Sentence Length", 
            f"{result.get('avg_sentence_length', 0):.1f} words",
            help="Average number of words per sentence"
        )
    
    with col3:
        st.metric(
            "Readability Score", 
            f"{result.get('readability_score', 0):.1f}/10",
            help="Readability score - higher means more accessible"
        )
    
    # Sentiment analysis by sections
    st.subheader("📊 Section-wise Sentiment Analysis")
    
    sections = ['problem', 'solution', 'market', 'traction', 'team']
    sentiment_data = []
    
    for section in sections:
        sentiment_key = f'{section}_sentiment'
        if sentiment_key in result:
            sentiment_data.append({
                'Section': section.title(),
                'Sentiment': result[sentiment_key],
                'Sentiment_Label': 'Positive' if result[sentiment_key] > 0.1 else 'Negative' if result[sentiment_key] < -0.1 else 'Neutral'
            })
    
    if sentiment_data:
        sentiment_df = pd.DataFrame(sentiment_data)
        
        fig = px.bar(
            sentiment_df,
            x='Section',
            y='Sentiment',
            color='Sentiment_Label',
            title='Sentiment Analysis by Pitch Section',
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Language sophistication breakdown
    st.subheader("📝 Language Analysis Breakdown")
    
    analysis_metrics = {
        'Confidence Indicators': result.get('confidence_score', 0),
        'Technical Sophistication': result.get('tech_sophistication', 0),
        'Vocabulary Diversity': result.get('vocabulary_diversity', 0) * 100,
        'Readability Score': result.get('readability_score', 0)
    }
    
    fig = go.Figure(go.Bar(
        x=list(analysis_metrics.keys()),
        y=list(analysis_metrics.values()),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ))
    
    fig.update_layout(
        title='Language Sophistication Metrics',
        yaxis_title='Score / Frequency',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_single_deck_visualizations(result):
    """Display advanced visualizations for single deck"""
    
    st.subheader("📊 Advanced Visualizations")
    
    # Gauge chart for composite score
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = result.get('composite_score', 0),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Composite Score"},
        delta = {'reference': 7, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 4], 'color': "lightgray"},
                {'range': [4, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Radar chart for this specific deck
    dimensions = [
        'problem_clarity', 'market_potential', 'traction_strength',
        'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
    ]
    
    values = [result.get(dim, 0) for dim in dimensions]
    labels = [dim.replace('_', ' ').title() for dim in dimensions]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=result.get('deck_name', 'Pitch Deck'),
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.25)'
    ))
    
    # Add benchmark comparison
    benchmark_values = [7.0, 7.5, 6.8, 7.2, 7.1, 6.5, 7.3]  # Industry averages
    fig_radar.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=labels,
        fill='toself',
        name='Industry Average',
        line=dict(color='red', width=2, dash='dash'),
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Performance vs Industry Benchmark",
        height=600
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def display_ml_predictions(result, evaluator):
    """Display machine learning predictions"""
    
    st.subheader("🎯 Machine Learning Predictions")
    
    # Create a mini dataset for prediction
    temp_df = pd.DataFrame([result])
    
    try:
        # Get success probability
        probabilities = evaluator.success_predictor.predict_success_probability(temp_df)
        success_prob = probabilities[0] if len(probabilities) > 0 else 0.5
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            # Success probability gauge
            fig_prob = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = success_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Investment Success Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_prob.update_layout(height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Risk assessment
            risk_factors = {
                'Market Risk': 10 - result.get('market_potential', 0),
                'Execution Risk': 10 - result.get('traction_strength', 0),
                'Team Risk': 10 - result.get('team_experience', 0),
                'Business Model Risk': 10 - result.get('business_model', 0)
            }
            
            fig_risk = px.bar(
                x=list(risk_factors.keys()),
                y=list(risk_factors.values()),
                title='Risk Assessment',
                color=list(risk_factors.values()),
                color_continuous_scale='Reds'
            )
            fig_risk.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Feature importance if model is trained
        if hasattr(evaluator.success_predictor, 'feature_importance'):
            st.subheader("📊 Key Success Factors")
            importance_fig = evaluator.success_predictor.get_feature_importance_chart()
            st.plotly_chart(importance_fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"ML predictions not available: {e}")
        st.info("💡 ML predictions will be available after analyzing multiple pitch decks")

def display_content_analysis(result):
    """Display content analysis by sections"""
    
    st.subheader("📖 Content Analysis by Section")
    
    # Normalize result data
    result = normalize_result_data(result)
    
    sections = [
        ('problem', 'Problem Statement', '❗'),
        ('solution', 'Solution Description', '💡'),
        ('market', 'Market Analysis', '📈'),
        ('traction', 'Traction & Metrics', '🚀'),
        ('team', 'Team Information', '👥'),
        ('business_model', 'Business Model', '💼'),
        ('ask', 'Funding Ask', '💰')
    ]
    
    for section_key, section_title, icon in sections:
        content = result.get(section_key, '')
        
        if content and len(str(content).strip()) > 20:
            with st.expander(f"{icon} {section_title}", expanded=False):
                content_str = str(content)
                display_content = content_str[:500] + "..." if len(content_str) > 500 else content_str
                st.write(display_content)
                
                # Word count for section
                word_count = len(content_str.split())
                st.caption(f"Word count: {word_count}")
        else:
            st.warning(f"{icon} {section_title}: No significant content detected")

def display_export_options(result):
    """Display export and download options"""
    
    st.subheader("📥 Export Options")
    
    # Normalize result data
    result = normalize_result_data(result)
    
    # Create export data with safe conversions
    export_data = {
        'Metric': ['Composite Score', 'Problem Clarity', 'Market Potential', 
                  'Traction Strength', 'Team Experience', 'Business Model', 
                  'Vision & Moat', 'Overall Confidence'],
        'Score': [
            safe_float_conversion(result.get('composite_score', 0)),
            safe_float_conversion(result.get('problem_clarity', 0)),
            safe_float_conversion(result.get('market_potential', 0)),
            safe_float_conversion(result.get('traction_strength', 0)),
            safe_float_conversion(result.get('team_experience', 0)),
            safe_float_conversion(result.get('business_model', 0)),
            safe_float_conversion(result.get('vision_moat', 0)),
            safe_float_conversion(result.get('overall_confidence', 0))
        ]
    }
    
    export_df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Scores (CSV)",
            data=csv,
            file_name=f"{result.get('deck_name', 'pitch_deck')}_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON download
        json_result = {}
        for key, value in result.items():
            if key == 'timestamp':
                json_result[key] = str(value)
            else:
                json_result[key] = value
        
        json_data = json.dumps(json_result, indent=2, default=str)
        st.download_button(
            label="📋 Download Full Report (JSON)",
            data=json_data,
            file_name=f"{result.get('deck_name', 'pitch_deck')}_full_report.json",
            mime="application/json"
        )
    
    # Display summary table
    st.markdown("### 📋 Summary Table")
    st.dataframe(export_df, use_container_width=True)
    
    # Analysis metadata
    st.markdown("### ℹ️ Analysis Metadata")
    
    metadata_col1, metadata_col2 = st.columns(2)
    
    with metadata_col1:
        st.info(f"""
        **Analysis Details:**
        - **File:** {result.get('deck_name', 'Unknown')}
        - **Category:** {result.get('category', 'Unknown')}
        - **Word Count:** {safe_int_conversion(result.get('word_count', 0)):,}
        - **Extraction Method:** {result.get('extraction_method', 'Unknown')}
        """)
    
    with metadata_col2:
        st.info(f"""
        **Scoring Weights:**
        - Problem Clarity: 15%
        - Market Potential: 20%
        - Traction Strength: 25%
        - Team Experience: 15%
        - Business Model: 15%
        - Vision & Moat: 5%
        - Overall Confidence: 5%
        """)

def batch_analysis(evaluator):
    """Batch analysis interface"""
    
    st.header("📁 Advanced Batch Analysis")
    st.info("Upload multiple PDF files for comprehensive comparative analysis with ML insights")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload multiple pitch deck PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF files to compare pitch decks and generate portfolio insights"
    )
    
    if uploaded_files:
        st.write(f"📄 Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🔍 Analyze All Pitch Decks", type="primary"):
            analyze_batch_pdfs_advanced(evaluator, uploaded_files)

def analyze_batch_pdfs_advanced(evaluator, uploaded_files):
    """Analyze multiple PDFs in batch with advanced analytics"""
    
    results = []
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_placeholder.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        result = evaluator.process_uploaded_file(uploaded_file)
        
        if 'error' not in result:
            result = normalize_result_data(result)
            results.append(result)
            st.session_state.results_history.append(result)
        else:
            st.error(f"Failed to process {uploaded_file.name}: {result['error']}")
    
    progress_placeholder.empty()
    progress_bar.empty()
    
    if results:
        display_batch_results_advanced(results, evaluator)

def display_batch_results_advanced(results, evaluator):
    """Display advanced batch analysis results"""
    
    st.success(f"🎉 Successfully analyzed {len(results)} pitch decks!")
    
    # Normalize all results
    results = [normalize_result_data(result) for result in results]
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['composite_score', 'problem_clarity', 'market_potential', 'traction_strength',
                      'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
    
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(safe_float_conversion)
    
    # Generate advanced insights
    portfolio_insights = evaluator.insights_generator.generate_portfolio_insights(results_df)
    
    # Display executive summary
    st.markdown("""
    <div class="advanced-section">
        <h2>🎯 Portfolio Intelligence Summary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Decks", len(results))
    with col2:
        avg_score = results_df['composite_score'].mean()
        st.metric("Portfolio Avg", f"{avg_score:.2f}")
    with col3:
        investment_ready = len(results_df[results_df['composite_score'] >= 7])
        st.metric("Investment Ready", f"{investment_ready}")
    with col4:
        categories = results_df['category'].nunique() if 'category' in results_df.columns else 0
        st.metric("Categories", categories)
    with col5:
        max_score = results_df['composite_score'].max()
        st.metric("Top Score", f"{max_score:.2f}")
    
    # Advanced analytics tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Rankings", "🔮 ML Insights", "📊 Advanced Viz", "🎯 Benchmarks", "💡 AI Recommendations", "📥 Export"
    ])
    
    with tab1:
        st.subheader("🏆 Comprehensive Rankings")
        ranking_table = evaluator.dashboard.create_scoring_summary_table(results_df)
        st.dataframe(ranking_table, use_container_width=True)
        
        # Advanced ranking chart
        ranking_fig = evaluator.dashboard.create_ranking_chart(results_df)
        st.plotly_chart(ranking_fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Machine Learning Insights")
        
        # Train ML model on batch data
        try:
            ml_results = evaluator.success_predictor.train_model(results_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{ml_results['test_score']:.3f}")
            with col2:
                st.metric("Training Score", f"{ml_results['train_score']:.3f}")
            
            # Feature importance
            st.subheader("📊 Key Success Factors")
            if 'feature_importance' in ml_results:
                importance_df = ml_results['feature_importance'].head(10)
                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Most Important Features for Investment Success'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Success predictions
            st.subheader("🎯 Investment Success Predictions")
            probabilities = evaluator.success_predictor.predict_success_probability(results_df)
            results_df['success_probability'] = probabilities
            
            pred_df = results_df[['deck_name', 'composite_score', 'success_probability', 'category']].copy()
            pred_df = pred_df.sort_values('success_probability', ascending=False)
            pred_df['success_probability'] = (pred_df['success_probability'] * 100).round(1)
            
            st.dataframe(pred_df, use_container_width=True)
            
        except Exception as e:
            st.warning(f"ML analysis not available: {e}")
    
    with tab3:
        st.subheader("📊 Advanced Visualizations")
        
        # 3D scatter plot
        scatter_3d_fig = evaluator.dashboard.create_3d_performance_scatter(results_df)
        st.plotly_chart(scatter_3d_fig, use_container_width=True)
        
        # Parallel coordinates
        parallel_fig = evaluator.dashboard.create_parallel_coordinates(results_df)
        st.plotly_chart(parallel_fig, use_container_width=True)
        
        # Competitive landscape
        competitive_fig = evaluator.dashboard.create_competitive_landscape_map(results_df)
        st.plotly_chart(competitive_fig, use_container_width=True)
        
        # Risk assessment heatmap
        risk_fig = evaluator.dashboard.create_risk_assessment_heatmap(results_df)
        st.plotly_chart(risk_fig, use_container_width=True)
    
    with tab4:
        st.subheader("🎯 Industry Benchmark Analysis")
        
        benchmark_comparison = evaluator.benchmark_analyzer.create_benchmark_comparison(results_df)
        
        if not benchmark_comparison.empty:
            # Benchmark metrics
            above_benchmark = len(benchmark_comparison[benchmark_comparison['composite_vs_benchmark'] > 0])
            avg_vs_benchmark = benchmark_comparison['composite_vs_benchmark'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Above Benchmark", f"{above_benchmark}/{len(benchmark_comparison)}")
            with col2:
                st.metric("Avg vs Industry", f"{avg_vs_benchmark:+.2f}")
            with col3:
                top_performer = benchmark_comparison.loc[benchmark_comparison['composite_vs_benchmark'].idxmax()]
                st.metric("Top vs Benchmark", f"+{top_performer['composite_vs_benchmark']:.2f}")
            
            # Benchmark visualization
            benchmark_fig = evaluator.benchmark_analyzer.create_benchmark_visualization(benchmark_comparison)
            st.plotly_chart(benchmark_fig, use_container_width=True)
            
            # Benchmark table
            st.dataframe(benchmark_comparison, use_container_width=True)
        else:
            st.info("No benchmark data available for current categories")
    
    with tab5:
        st.subheader("💡 AI-Powered Investment Recommendations")
        
        # Investment recommendations
        recommendations = portfolio_insights['investment_recommendations']
        
        for rec in recommendations:
            if rec['type'] == 'STRONG_BUY':
                st.success(f"🎯 **{rec['deck']}** ({rec['category']}) - {rec['reasoning']} | Score: {rec['score']:.2f}/10")
            elif rec['type'] == 'CONSIDER':
                st.info(f"🤔 **{rec['deck']}** ({rec['category']}) - {rec['reasoning']} | Score: {rec['score']:.2f}/10")
            elif rec['type'] == 'WATCH':
                st.warning(f"👀 **{rec['deck']}** ({rec['category']}) - {rec['reasoning']} | Score: {rec['score']:.2f}/10")
        
        # Risk assessment
        st.subheader("⚠️ Portfolio Risk Assessment")
        risk_assessment = portfolio_insights['risk_assessment']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_color = 'red' if risk_assessment.get('concentration_risk') == 'High' else 'orange' if risk_assessment.get('concentration_risk') == 'Medium' else 'green'
            st.markdown(f"**Concentration Risk:** :{risk_color}[{risk_assessment.get('concentration_risk', 'Unknown')}]")
        with col2:
            risk_color = 'red' if risk_assessment.get('quality_risk') == 'High' else 'orange' if risk_assessment.get('quality_risk') == 'Medium' else 'green'
            st.markdown(f"**Quality Risk:** :{risk_color}[{risk_assessment.get('quality_risk', 'Unknown')}]")
        with col3:
            risk_color = 'red' if risk_assessment.get('execution_risk') == 'High' else 'orange' if risk_assessment.get('execution_risk') == 'Medium' else 'green'
            st.markdown(f"**Execution Risk:** :{risk_color}[{risk_assessment.get('execution_risk', 'Unknown')}]")
        
        # Due diligence flags
        due_diligence_flags = portfolio_insights['due_diligence_flags']
        if due_diligence_flags:
            st.subheader("🚨 Due Diligence Flags")
            for flag in due_diligence_flags:
                with st.expander(f"⚠️ {flag['deck_name']} (Score: {flag['composite_score']:.2f})"):
                    for flag_item in flag['flags']:
                        st.warning(flag_item)
    
    with tab6:
        st.subheader("📥 Portfolio Export Options")
        
        # Export comprehensive results
        csv_data = ranking_table.to_csv(index=False)
        st.download_button(
            label="📊 Download Portfolio Analysis (CSV)",
            data=csv_data,
            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Export with ML predictions if available
        if 'success_probability' in results_df.columns:
            ml_export = results_df[['deck_name', 'composite_score', 'success_probability', 'category']].copy()
            ml_csv = ml_export.to_csv(index=False)
            st.download_button(
                label="🔮 Download ML Predictions (CSV)",
                data=ml_csv,
                file_name=f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def advanced_analytics_dashboard(evaluator):
    """Advanced analytics dashboard for portfolio analysis"""
    
    st.header("🔬 Advanced Analytics Dashboard")
    
    if not st.session_state.results_history:
        st.info("No analysis history available. Please analyze some pitch decks first!")
        return
    
    # Create DataFrame from history
    history_df = pd.DataFrame(st.session_state.results_history)
    
    # Normalize data
    for col in ['composite_score', 'problem_clarity', 'market_potential', 'traction_strength',
               'team_experience', 'business_model', 'vision_moat', 'overall_confidence']:
        if col in history_df.columns:
            history_df[col] = history_df[col].apply(safe_float_conversion)
    
    st.markdown("""
    <div class="advanced-section">
        <h2>🧠 Advanced Portfolio Intelligence</h2>
        <p>Deep insights powered by machine learning and AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyzed", len(history_df))
    with col2:
        st.metric("Portfolio Average", f"{history_df['composite_score'].mean():.2f}")
    with col3:
        st.metric("Investment Ready", len(history_df[history_df['composite_score'] >= 7]))
    with col4:
        st.metric("Categories", history_df['category'].nunique() if 'category' in history_df.columns else 0)
    
    # Advanced analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 3D Analytics", "🎯 ML Clustering", "📈 Advanced Charts", "🔮 Predictive Models"
    ])
    
    with tab1:
        st.subheader("🌐 3D Performance Analysis")
        scatter_3d = evaluator.dashboard.create_3d_performance_scatter(history_df)
        st.plotly_chart(scatter_3d, use_container_width=True)
        
        st.subheader("🌞 Hierarchical Analysis")
        sunburst = evaluator.dashboard.create_sunburst_analysis(history_df)
        st.plotly_chart(sunburst, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Performance Clustering")
        clustering = evaluator.dashboard.create_performance_clusters(history_df)
        st.plotly_chart(clustering, use_container_width=True)
        
        st.subheader("📊 Parallel Coordinates")
        parallel = evaluator.dashboard.create_parallel_coordinates(history_df)
        st.plotly_chart(parallel, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Competitive Landscape")
        competitive = evaluator.dashboard.create_competitive_landscape_map(history_df)
        st.plotly_chart(competitive, use_container_width=True)
        
        st.subheader("⚠️ Risk Assessment Matrix")
        risk_matrix = evaluator.dashboard.create_risk_assessment_heatmap(history_df)
        st.plotly_chart(risk_matrix, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Machine Learning Predictions")
        
        try:
            # Train ML model
            ml_results = evaluator.success_predictor.train_model(history_df)
            
            # Show model performance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{ml_results['test_score']:.3f}")
            with col2:
                st.metric("Feature Count", len(ml_results['feature_importance']))
            
            # Feature importance
            importance_fig = px.bar(
                ml_results['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Success Factors'
            )
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Predictions
            probabilities = evaluator.success_predictor.predict_success_probability(history_df)
            history_df['success_probability'] = probabilities
            
            # Success probability distribution
            fig_dist = px.histogram(
                history_df,
                x='success_probability',
                title='Investment Success Probability Distribution',
                nbins=20
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Predictive modeling not available: {e}")

def results_history():
    """Display results history with advanced analytics"""
    
    st.header("📊 Analysis History & Portfolio Overview")
    
    if not st.session_state.results_history:
        st.info("No analysis history available. Analyze some pitch decks first!")
        return
    
    # Normalize all history data
    normalized_history = [normalize_result_data(result) for result in st.session_state.results_history]
    
    # Display history
    history_df = pd.DataFrame(normalized_history)
    
    # Ensure numeric columns are properly typed
    if 'composite_score' in history_df.columns:
        history_df['composite_score'] = history_df['composite_score'].apply(safe_float_conversion)
    
    st.subheader(f"📋 Complete Analysis History ({len(history_df)} analyses)")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(history_df))
    with col2:
        if 'composite_score' in history_df.columns:
            avg_score = history_df['composite_score'].mean()
            st.metric("Average Score", f"{avg_score:.2f}")
    with col3:
        if 'composite_score' in history_df.columns:
            max_score = history_df['composite_score'].max()
            st.metric("Best Score", f"{max_score:.2f}")
    with col4:
        investment_ready = len(history_df[history_df['composite_score'] >= 7]) if 'composite_score' in history_df.columns else 0
        st.metric("Investment Ready", investment_ready)
    
    # History table with filters
    st.subheader("🔍 Detailed History")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if 'category' in history_df.columns:
            categories = ['All'] + list(history_df['category'].unique())
            selected_category = st.selectbox("Filter by Category", categories)
        else:
            selected_category = 'All'
    
    with col2:
        min_score = st.slider("Minimum Score", 0.0, 10.0, 0.0, 0.1)
    
    # Apply filters
    filtered_df = history_df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if 'composite_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['composite_score'] >= min_score]
    
    # Display filtered results
    display_columns = ['deck_name', 'composite_score']
    if 'category' in filtered_df.columns:
        display_columns.append('category')
    if 'timestamp' in filtered_df.columns:
        display_columns.append('timestamp')
    
    existing_columns = [col for col in display_columns if col in filtered_df.columns]
    display_df = filtered_df[existing_columns].copy()
    
    if 'timestamp' in display_df.columns:
        display_df = display_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Advanced history analytics
    if len(history_df) > 1:
        st.subheader("📈 Historical Trends")
        
        # Score progression
        history_df_sorted = history_df.reset_index()
        history_df_sorted['Analysis_Number'] = history_df_sorted.index + 1
        
        if 'composite_score' in history_df_sorted.columns:
            fig_trend = px.line(
                history_df_sorted,
                x='Analysis_Number',
                y='composite_score',
                title='Score Progression Over Time',
                markers=True,
                hover_data=['deck_name'] if 'deck_name' in history_df_sorted.columns else None
            )
            fig_trend.update_yaxis(title="Composite Score", range=[0, 10])
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Category performance over time
        if 'category' in history_df.columns and len(history_df['category'].unique()) > 1:
            cat_performance = history_df.groupby('category')['composite_score'].agg(['mean', 'count']).reset_index()
            
            fig_cat = px.scatter(
                cat_performance,
                x='mean',
                y='count',
                text='category',
                title='Category Performance vs Volume',
                labels={'mean': 'Average Score', 'count': 'Number of Pitches'}
            )
            fig_cat.update_traces(textposition="top center")
            st.plotly_chart(fig_cat, use_container_width=True)
    
    # Clear history option
    if st.button("🗑️ Clear All History", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            st.session_state.results_history = []
            st.session_state.confirm_clear = False
            st.success("History cleared!")
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm clearing all history")

if __name__ == "__main__":
    main()
