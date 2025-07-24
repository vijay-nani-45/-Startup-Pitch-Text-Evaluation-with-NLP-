# dashboard.py - Enhanced with advanced visualizations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
from wordcloud import WordCloud
from config import Config
import os
from datetime import datetime
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class EnhancedDashboard:
    def __init__(self):
        self.config = Config()
        self.colors = px.colors.qualitative.Set3
        plt.style.use('default')
        
    def create_scoring_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive scoring summary table"""
        if results_df.empty:
            return pd.DataFrame()
        
        display_df = results_df.copy()
        
        # Score columns to round
        score_columns = [
            'problem_clarity', 'market_potential', 'traction_strength', 
            'team_experience', 'business_model', 'vision_moat', 
            'overall_confidence', 'composite_score'
        ]
        
        # Round scores and handle missing columns
        for col in score_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        # Add ranking
        display_df['rank'] = display_df['composite_score'].rank(ascending=False, method='min').astype(int)
        
        # Reorder columns for better presentation
        column_order = ['rank', 'deck_name', 'composite_score']
        
        # Add score columns that exist
        for col in score_columns:
            if col != 'composite_score' and col in display_df.columns:
                column_order.append(col)
        
        # Add other important columns
        for col in ['category', 'investability_insight', 'word_count', 'extraction_method']:
            if col in display_df.columns:
                column_order.append(col)
        
        # Filter to existing columns
        existing_columns = [col for col in column_order if col in display_df.columns]
        
        return display_df[existing_columns].sort_values('composite_score', ascending=False)
    
    def create_3d_performance_scatter(self, results_df: pd.DataFrame):
        """3D scatter plot showing market potential vs traction vs team experience"""
        
        if results_df.empty or len(results_df) < 2:
            return go.Figure()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=results_df['market_potential'],
            y=results_df['traction_strength'], 
            z=results_df['team_experience'],
            mode='markers+text',
            marker=dict(
                size=results_df['composite_score'] * 2,
                color=results_df['composite_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Composite Score"),
                opacity=0.8
            ),
            text=results_df['deck_name'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Market Potential: %{x:.1f}<br>' +
                         'Traction: %{y:.1f}<br>' +
                         'Team Experience: %{z:.1f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title="3D Performance Analysis: Market vs Traction vs Team",
            scene=dict(
                xaxis_title='Market Potential',
                yaxis_title='Traction Strength',
                zaxis_title='Team Experience',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        return fig
    
    def create_parallel_coordinates(self, results_df: pd.DataFrame):
        """Parallel coordinates for multi-dimensional comparison"""
        
        if results_df.empty:
            return go.Figure()
        
        dimensions = [
            'problem_clarity', 'market_potential', 'traction_strength',
            'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
        ]
        
        # Filter to existing dimensions
        existing_dimensions = [dim for dim in dimensions if dim in results_df.columns]
        
        if not existing_dimensions:
            return go.Figure()
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=results_df['composite_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Composite Score")
                ),
                dimensions=list([
                    dict(
                        range=[0, 10],
                        label=dim.replace('_', ' ').title(),
                        values=results_df[dim]
                    ) for dim in existing_dimensions
                ])
            )
        )
        
        fig.update_layout(
            title="Multi-Dimensional Performance Comparison",
            height=500
        )
        
        return fig
    
    def create_competitive_landscape_map(self, results_df: pd.DataFrame):
        """Create competitive positioning map"""
        
        if results_df.empty or len(results_df) < 2:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create quadrants
        fig.add_shape(
            type="line", x0=5, y0=0, x1=5, y1=10,
            line=dict(color="gray", width=1, dash="dash")
        )
        fig.add_shape(
            type="line", x0=0, y0=5, x1=10, y1=5,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Add quadrant labels
        annotations = [
            dict(x=2.5, y=7.5, text="High Market<br>Low Execution", showarrow=False, font=dict(color="red")),
            dict(x=7.5, y=7.5, text="High Market<br>High Execution", showarrow=False, font=dict(color="green")),
            dict(x=2.5, y=2.5, text="Low Market<br>Low Execution", showarrow=False, font=dict(color="red")),
            dict(x=7.5, y=2.5, text="Low Market<br>High Execution", showarrow=False, font=dict(color="orange"))
        ]
        
        # Plot decks by category
        if 'category' in results_df.columns:
            categories = results_df['category'].unique()
        else:
            categories = ['All Decks']
            results_df['category'] = 'All Decks'
        
        for category in categories:
            category_data = results_df[results_df['category'] == category]
            
            fig.add_trace(go.Scatter(
                x=category_data['market_potential'],
                y=category_data['traction_strength'],
                mode='markers+text',
                name=category,
                text=category_data['deck_name'],
                textposition="top center",
                marker=dict(
                    size=category_data['composite_score'] * 3,
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title="Competitive Landscape Positioning",
            xaxis_title="Market Potential",
            yaxis_title="Execution Strength (Traction)",
            annotations=annotations,
            height=600,
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10])
        )
        
        return fig
    
    def create_risk_assessment_heatmap(self, results_df: pd.DataFrame):
        """Risk assessment based on multiple dimensions"""
        
        if results_df.empty:
            return go.Figure()
        
        # Calculate risk scores (inverse of performance scores)
        risk_factors = {
            'Market Risk': 10 - results_df['market_potential'],
            'Execution Risk': 10 - results_df['traction_strength'],  
            'Team Risk': 10 - results_df['team_experience'],
            'Business Model Risk': 10 - results_df['business_model'],
            'Competition Risk': 10 - results_df['vision_moat']
        }
        
        risk_df = pd.DataFrame(risk_factors, index=results_df['deck_name'])
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_df.values,
            x=risk_df.columns,
            y=risk_df.index,
            colorscale='RdYlGn_r',  # Reverse colors (red = high risk)
            text=risk_df.round(1).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Risk Level")
        ))
        
        fig.update_layout(
            title="Investment Risk Assessment Matrix",
            height=max(400, len(results_df) * 30),
            xaxis_title="Risk Factors",
            yaxis_title="Pitch Decks"
        )
        
        return fig
    
    def create_performance_clusters(self, results_df: pd.DataFrame):
        """Create clustering visualization based on performance metrics"""
        
        if results_df.empty or len(results_df) < 3:
            return go.Figure()
        
        # Prepare data for clustering
        feature_columns = ['problem_clarity', 'market_potential', 'traction_strength', 
                          'team_experience', 'business_model', 'vision_moat', 'overall_confidence']
        
        existing_features = [col for col in feature_columns if col in results_df.columns]
        
        if len(existing_features) < 3:
            return go.Figure()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(results_df[existing_features])
        
        # Perform clustering
        n_clusters = min(4, len(results_df))  # Max 4 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Reduce dimensionality for visualization using t-SNE
        if len(results_df) > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(results_df)-1))
            coords_2d = tsne.fit_transform(features_scaled)
        else:
            coords_2d = features_scaled[:, :2]  # Use first two features
        
        # Create scatter plot
        fig = go.Figure()
        
        cluster_names = ['High Performers', 'Medium Performers', 'Developing', 'Needs Focus']
        colors = ['green', 'blue', 'orange', 'red']
        
        for i in range(n_clusters):
            cluster_data = results_df[clusters == i]
            cluster_coords = coords_2d[clusters == i]
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers+text',
                name=cluster_names[i] if i < len(cluster_names) else f'Cluster {i+1}',
                text=cluster_data['deck_name'],
                textposition="top center",
                marker=dict(
                    size=cluster_data['composite_score'] * 2,
                    color=colors[i] if i < len(colors) else 'gray',
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title="Performance-Based Clustering Analysis",
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            height=600
        )
        
        return fig
    
    def create_animated_performance_timeline(self, historical_data: pd.DataFrame):
        """Animated timeline showing pitch evolution over time"""
        
        if historical_data.empty or 'analysis_date' not in historical_data.columns:
            return go.Figure()
        
        fig = px.scatter(
            historical_data,
            x='market_potential',
            y='traction_strength',
            animation_frame='analysis_date',
            animation_group='deck_name',
            size='composite_score',
            color='category' if 'category' in historical_data.columns else None,
            hover_name='deck_name',
            size_max=30,
            range_x=[0, 10],
            range_y=[0, 10],
            title="Pitch Deck Performance Evolution Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Market Potential Score",
            yaxis_title="Traction Strength Score",
            height=600
        )
        
        return fig
    
    def create_sunburst_analysis(self, results_df: pd.DataFrame):
        """Hierarchical breakdown by category and performance tiers"""
        
        if results_df.empty or 'category' not in results_df.columns:
            return go.Figure()
        
        # Create performance tiers
        results_df = results_df.copy()
        results_df['performance_tier'] = pd.cut(
            results_df['composite_score'], 
            bins=[0, 4, 7, 10], 
            labels=['Needs Work', 'Promising', 'Investment Ready']
        )
        
        # Create hierarchical data
        sunburst_data = []
        
        # Add root
        sunburst_data.extend([
            dict(ids="Total", labels="Portfolio", parents="", values=len(results_df))
        ])
        
        # Add categories
        for category in results_df['category'].unique():
            cat_count = len(results_df[results_df['category'] == category])
            sunburst_data.append(
                dict(ids=category, labels=category, parents="Total", values=cat_count)
            )
        
        # Add tiers within categories
        for category in results_df['category'].unique():
            cat_data = results_df[results_df['category'] == category]
            for tier in cat_data['performance_tier'].unique():
                if pd.notna(tier):
                    tier_count = len(cat_data[cat_data['performance_tier'] == tier])
                    tier_id = f"{category}-{tier}"
                    sunburst_data.append(
                        dict(ids=tier_id, labels=str(tier), parents=category, values=tier_count)
                    )
        
        fig = go.Figure(go.Sunburst(
            ids=[item['ids'] for item in sunburst_data],
            labels=[item['labels'] for item in sunburst_data],
            parents=[item['parents'] for item in sunburst_data],
            values=[item['values'] for item in sunburst_data],
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Hierarchical Performance Analysis",
            height=600
        )
        
        return fig

    # Keep all existing methods from the original dashboard.py
    def create_radar_chart(self, results_df: pd.DataFrame, top_n: int = 5):
        """Create radar chart for top N decks"""
        if results_df.empty:
            return go.Figure()
        
        dimensions = [
            'problem_clarity', 'market_potential', 'traction_strength', 
            'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
        ]
        
        # Filter to existing dimensions
        existing_dimensions = [dim for dim in dimensions if dim in results_df.columns]
        
        if not existing_dimensions:
            return go.Figure()
        
        # Get top N decks
        top_decks = results_df.nlargest(min(top_n, len(results_df)), 'composite_score')
        
        fig = go.Figure()
        
        for i, (_, deck) in enumerate(top_decks.iterrows()):
            values = [deck.get(dim, 0) for dim in existing_dimensions]
            values.append(values[0])  # Close the radar chart
            
            # Truncate long names
            deck_name = deck['deck_name']
            if len(deck_name) > 25:
                deck_name = deck_name[:22] + "..."
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=existing_dimensions + [existing_dimensions[0]],
                fill='toself',
                name=deck_name,
                opacity=0.7,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )
            ),
            showlegend=True,
            title=f"Top {len(top_decks)} Pitch Decks - Multi-Dimensional Analysis",
            height=600,
            font=dict(size=12)
        )
        
        return fig

    def create_ranking_chart(self, results_df: pd.DataFrame):
        """Create horizontal bar chart showing deck rankings"""
        if results_df.empty:
            return go.Figure()
        
        # Sort by composite score and limit to top 15 for readability
        sorted_df = results_df.sort_values('composite_score', ascending=True).tail(15)
        
        # Truncate long names
        deck_names = [name[:30] + "..." if len(name) > 30 else name for name in sorted_df['deck_name']]
        
        fig = go.Figure(go.Bar(
            x=sorted_df['composite_score'],
            y=deck_names,
            orientation='h',
            text=[f"{score:.1f}" for score in sorted_df['composite_score']],
            textposition='inside',
            marker=dict(
                color=sorted_df['composite_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            )
        ))
        
        fig.update_layout(
            title="Pitch Deck Rankings by Composite Score",
            xaxis_title="Composite Score (0-10)",
            yaxis_title="Pitch Decks",
            height=max(400, len(sorted_df) * 35),
            showlegend=False,
            margin=dict(l=200)
        )
        
        return fig

    def create_score_distribution(self, results_df: pd.DataFrame):
        """Create distribution plots for scoring dimensions"""
        if results_df.empty:
            return go.Figure()
        
        dimensions = [
            'problem_clarity', 'market_potential', 'traction_strength', 
            'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
        ]
        
        # Filter to existing dimensions
        existing_dimensions = [dim for dim in dimensions if dim in results_df.columns]
        
        if not existing_dimensions:
            return go.Figure()
        
        # Calculate subplot layout
        n_dims = len(existing_dimensions) + 1  # +1 for composite score
        cols = 3
        rows = (n_dims + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=existing_dimensions + ['Composite Score'],
            specs=[[{"type": "histogram"}] * cols for _ in range(rows)]
        )
        
        # Add histograms for each dimension
        for i, dim in enumerate(existing_dimensions):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=results_df[dim], 
                    name=dim.replace('_', ' ').title(),
                    nbinsx=min(10, len(results_df)),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add composite score
        if 'composite_score' in results_df.columns:
            comp_row = len(existing_dimensions) // cols + 1
            comp_col = len(existing_dimensions) % cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=results_df['composite_score'], 
                    name='Composite Score',
                    nbinsx=min(10, len(results_df)),
                    showlegend=False
                ),
                row=comp_row, col=comp_col
            )
        
        fig.update_layout(
            height=200 * rows, 
            title_text="Score Distributions Across All Dimensions",
            showlegend=False
        )
        
        return fig

    def create_correlation_heatmap(self, results_df: pd.DataFrame):
        """Create correlation heatmap between scoring dimensions"""
        dimensions = [
            'problem_clarity', 'market_potential', 'traction_strength', 
            'team_experience', 'business_model', 'vision_moat', 
            'overall_confidence', 'composite_score'
        ]
        
        existing_dimensions = [dim for dim in dimensions if dim in results_df.columns]
        
        if len(existing_dimensions) < 2:
            return go.Figure()
        
        correlation_matrix = results_df[existing_dimensions].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title="Correlation Matrix - Scoring Dimensions",
            height=600,
            width=800
        )
        
        return fig

    def create_category_analysis(self, results_df: pd.DataFrame):
        """Create analysis by startup categories"""
        if results_df.empty or 'category' not in results_df.columns:
            return None
        
        # Group by category and calculate stats
        category_stats = results_df.groupby('category').agg({
            'composite_score': ['mean', 'median', 'count', 'std'],
            'problem_clarity': 'mean',
            'market_potential': 'mean',
            'traction_strength': 'mean'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = [
            'Avg_Score', 'Median_Score', 'Count', 'Std_Score',
            'Avg_Problem', 'Avg_Market', 'Avg_Traction'
        ]
        
        # Reset index to make category a column
        category_stats = category_stats.reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Average Scores by Category', 'Category Distribution'],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=category_stats['category'], 
                y=category_stats['Avg_Score'],
                name='Average Score',
                text=[f"{score:.1f}" for score in category_stats['Avg_Score']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=category_stats['category'], 
                values=category_stats['Count'],
                name='Distribution'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=False, title_text="Category Analysis")
        return fig

    def create_performance_matrix(self, results_df: pd.DataFrame):
        """Create a performance matrix showing strengths and weaknesses"""
        if results_df.empty:
            return go.Figure()
        
        dimensions = [
            'problem_clarity', 'market_potential', 'traction_strength',
            'team_experience', 'business_model', 'vision_moat', 'overall_confidence'
        ]
        
        existing_dimensions = [dim for dim in dimensions if dim in results_df.columns]
        
        if not existing_dimensions or len(results_df) < 2:
            return go.Figure()
        
        # Create matrix data
        matrix_data = []
        for _, deck in results_df.iterrows():
            deck_scores = [deck.get(dim, 0) for dim in existing_dimensions]
            matrix_data.append(deck_scores)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=[dim.replace('_', ' ').title() for dim in existing_dimensions],
            y=[name[:20] + "..." if len(name) > 20 else name for name in results_df['deck_name']],
            colorscale='RdYlGn',
            zmid=5,
            text=[[f"{val:.1f}" for val in row] for row in matrix_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Performance Matrix - All Decks vs All Dimensions",
            height=max(400, len(results_df) * 30),
            width=max(800, len(existing_dimensions) * 100)
        )
        
        return fig

    def generate_executive_summary(self, results_df: pd.DataFrame) -> str:
        """Generate executive summary text"""
        if results_df.empty:
            return "No data available for analysis."
        
        n_decks = len(results_df)
        avg_score = results_df['composite_score'].mean()
        top_performer = results_df.loc[results_df['composite_score'].idxmax()]
        bottom_performer = results_df.loc[results_df['composite_score'].idxmin()]
        
        # Category analysis
        category_info = ""
        if 'category' in results_df.columns:
            top_category = results_df['category'].mode().iloc[0] if not results_df['category'].mode().empty else "Unknown"
            category_info = f"Most common category: {top_category}. "
        
        # Performance insights
        high_performers = len(results_df[results_df['composite_score'] >= 7])
        low_performers = len(results_df[results_df['composite_score'] < 5])
        
        summary = f"""
        📊 EXECUTIVE SUMMARY
        
        Analysis of {n_decks} pitch decks completed with an average composite score of {avg_score:.2f}/10.
        
        🏆 TOP PERFORMER: {top_performer['deck_name']} ({top_performer['composite_score']:.2f}/10)
        ⚠️ NEEDS IMPROVEMENT: {bottom_performer['deck_name']} ({bottom_performer['composite_score']:.2f}/10)
        
        📈 PERFORMANCE DISTRIBUTION:
        • High performers (7+ score): {high_performers} decks ({high_performers/n_decks*100:.1f}%)
        • Needs improvement (<5 score): {low_performers} decks ({low_performers/n_decks*100:.1f}%)
        
        {category_info}
        
        💡 KEY INSIGHTS:
        • Average problem clarity: {results_df.get('problem_clarity', [0]).mean():.1f}/10
        • Average market potential: {results_df.get('market_potential', [0]).mean():.1f}/10
        • Average traction strength: {results_df.get('traction_strength', [0]).mean():.1f}/10
        """
        
        return summary

    def save_comprehensive_report(self, results_df: pd.DataFrame, output_dir: str = None):
        """Generate and save comprehensive HTML report with advanced visualizations"""
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all visualizations
        try:
            radar_fig = self.create_radar_chart(results_df)
            ranking_fig = self.create_ranking_chart(results_df)
            distribution_fig = self.create_score_distribution(results_df)
            category_fig = self.create_category_analysis(results_df)
            matrix_fig = self.create_performance_matrix(results_df)
            
            # Advanced visualizations
            scatter_3d_fig = self.create_3d_performance_scatter(results_df)
            parallel_fig = self.create_parallel_coordinates(results_df)
            competitive_fig = self.create_competitive_landscape_map(results_df)
            risk_fig = self.create_risk_assessment_heatmap(results_df)
            clustering_fig = self.create_performance_clusters(results_df)
            sunburst_fig = self.create_sunburst_analysis(results_df)
            
            # Generate executive summary
            exec_summary = self.generate_executive_summary(results_df)
            
            # Create HTML report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Advanced Startup Pitch Deck Analysis Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        margin: 40px; 
                        background-color: #f8f9fa;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }}
                    .summary {{ 
                        background-color: white; 
                        padding: 25px; 
                        border-radius: 8px; 
                        margin: 20px 0; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .chart-container {{ 
                        background-color: white;
                        margin: 30px 0; 
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .advanced-section {{
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 30px 0;
                    }}
                    h1, h2 {{ color: #2c3e50; }}
                    .timestamp {{ color: #6c757d; font-size: 0.9em; }}
                    pre {{ 
                        background-color: #f1f3f4; 
                        padding: 15px; 
                        border-radius: 5px; 
                        white-space: pre-wrap;
                        font-family: 'Courier New', monospace;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 Advanced Startup Pitch Deck Analysis Report</h1>
                    <p class="timestamp">Generated on: {timestamp}</p>
                    <p>Complete AI-Powered Analysis with Advanced Visualizations</p>
                </div>
                
                <div class="summary">
                    <pre>{exec_summary}</pre>
                </div>
                
                <div class="chart-container">
                    <h2>📊 Deck Rankings</h2>
                    <div id="ranking-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>🎯 Multi-Dimensional Performance</h2>
                    <div id="radar-chart"></div>
                </div>
                
                <div class="advanced-section">
                    <h2>🔬 Advanced Analytics</h2>
                </div>
                
                <div class="chart-container">
                    <h2>🌐 3D Performance Analysis</h2>
                    <div id="scatter-3d-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>📈 Parallel Coordinates</h2>
                    <div id="parallel-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>🗺️ Competitive Landscape</h2>
                    <div id="competitive-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>⚠️ Risk Assessment Matrix</h2>
                    <div id="risk-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>🎯 Performance Clusters</h2>
                    <div id="clustering-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>🌞 Hierarchical Analysis</h2>
                    <div id="sunburst-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>📈 Performance Matrix</h2>
                    <div id="matrix-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>📊 Score Distributions</h2>
                    <div id="distribution-chart"></div>
                </div>
            """
            
            if category_fig:
                html_content += """
                <div class="chart-container">
                    <h2>🏷️ Category Analysis</h2>
                    <div id="category-chart"></div>
                </div>
                """
            
            html_content += """
                <script>
            """
            
            # Add all chart data
            html_content += f"Plotly.newPlot('ranking-chart', {ranking_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('radar-chart', {radar_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('scatter-3d-chart', {scatter_3d_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('parallel-chart', {parallel_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('competitive-chart', {competitive_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('risk-chart', {risk_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('clustering-chart', {clustering_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('sunburst-chart', {sunburst_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('matrix-chart', {matrix_fig.to_json()});\n"
            html_content += f"Plotly.newPlot('distribution-chart', {distribution_fig.to_json()});\n"
            
            if category_fig:
                html_content += f"Plotly.newPlot('category-chart', {category_fig.to_json()});\n"
            
            html_content += """
                </script>
            </body>
            </html>
            """
            
            # Save report
            report_path = os.path.join(output_dir, 'advanced_comprehensive_report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📋 Advanced comprehensive report saved: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"⚠️ Error generating comprehensive report: {e}")
            return None
