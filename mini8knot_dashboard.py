# Data manipulation and analysis
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical analysis
from scipy.stats import normaltest, shapiro
from scipy.stats import zscore
from scipy.signal import savgol_filter

# Time series
from datetime import datetime, timedelta
import time

# Streamlit
import streamlit as st

# System and utilities
import warnings
import io
import sys
from typing import Dict, List, Tuple, Optional
import logging

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_kde(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Kernel Density Estimation."""
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    return x_range, kde(x_range)

def calculate_process_capability(df: pd.DataFrame) -> float:
    """Calculate process capability index (Cpk)."""
    try:
        # Assuming 'value' column and specification limits
        values = df['value'].dropna()
        mean = values.mean()
        std = values.std()
        usl = mean + 3*std  # Example spec limits
        lsl = mean - 3*std
        
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        
        return cpk
    except Exception as e:
        logger.error(f"Error calculating Cpk: {str(e)}")
        return 0.0

def add_dashboard_tab():
    """Main function to create the dashboard tab."""
    try:
        st.header("Analytics Dashboard")
        
        # Create sample data for demonstration
        def create_sample_data():
            """Create sample data for the dashboard"""
            np.random.seed(42)  # For reproducibility
            date_rng = pd.date_range(start='2024-01-01', end='2024-02-01', freq='H')
            size = len(date_rng)
            
            df = pd.DataFrame({
                'timestamp': date_rng,
                'value': np.random.normal(100, 10, size),
                'commits': np.random.poisson(5, size),
                'contributors': np.random.poisson(3, size),
                'issues': np.random.poisson(8, size),
                'pull_requests': np.random.poisson(4, size),
                'response_time': np.random.exponential(2, size),
                'category': np.random.choice(['A', 'B', 'C', 'D'], size),
                'status': np.random.choice(['active', 'inactive'], size, p=[0.8, 0.2])
            })
            
            # Add some trends and patterns
            df['value'] += np.linspace(0, 20, size)  # Adding upward trend
            df['activity_score'] = df['commits'] + df['pull_requests'] + df['issues']
            
            return df

        # Create or get the data
        if 'dashboard_df' not in st.session_state:
            st.session_state.dashboard_df = create_sample_data()
        
        df = st.session_state.dashboard_df

        # Add data source disclaimer
        st.info("""
        🔍 **Data Source Information**
        
        This dashboard currently shows conceptual data for demonstration purposes. In a production environment, 
        it would be connected to Augur's database to show real open source project metrics and analytics.
        """)
        # Create sections using columns
        metrics_col, trends_col = st.columns([1, 2])
        
        with metrics_col:
            st.subheader("Key Performance Metrics")
            
            # Add metric cards with explanations
            st.metric(
                label="Total Samples",
                value=f"{len(df):,}",
                delta="Active Samples"
            )
            with st.expander("ℹ️ About this metric"):
                st.markdown("""
                **Total Samples** represents the number of data points collected from the project.
                In Augur, this would show:
                - Total number of commits
                - Number of active contributors
                - Repository activity metrics
                """)
            
            # Process capability metrics
            cpk_value = calculate_process_capability(df)
            st.metric(
                label="Process Capability (Cpk)",
                value=f"{cpk_value:.2f}",
                delta=f"{'Above' if cpk_value > 1.33 else 'Below'} Target",
                delta_color="normal" if cpk_value > 1.33 else "inverse"
            )
            with st.expander("ℹ️ About Process Capability"):
                st.markdown("""
                **Process Capability (Cpk)** measures how well the process meets specifications:
                - Cpk > 1.33: Process is capable
                - Cpk < 1.33: Process needs improvement
                
                In Augur context, this would measure:
                - Code quality metrics
                - Review process efficiency
                - Release cycle stability
                """)

        with trends_col:
            st.subheader("Trend Analysis")
            fig = plot_trend_prediction(df)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📊 Understanding Trend Analysis"):
                st.markdown("""
                **Trend Analysis** shows the pattern of activity over time:
                - Blue line: Historical data
                - Red dashed line: Predicted trend
                
                In Augur, this would visualize:
                - Commit frequency
                - Issue resolution rates
                - Community growth patterns
                - Development velocity
                """)

        # Statistical Insights section
        st.subheader("Statistical Insights")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.markdown("#### Distribution Analysis")
            fig = plot_distribution_analysis(df)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📈 About Distribution Analysis"):
                st.markdown("""
                **Distribution Analysis** shows the spread of values:
                - Histogram: Frequency of values
                - Red line: Kernel Density Estimation
                
                In Augur context, this would show:
                - Code contribution patterns
                - Issue response times
                - Pull request sizes
                - Review durations
                """)
            
        with stats_col2:
            st.markdown("#### Feature Correlations")
            fig = plot_correlation_matrix(df)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("🔗 Understanding Correlations"):
                st.markdown("""
                **Correlation Matrix** shows relationships between metrics:
                - Red: Positive correlation
                - Blue: Negative correlation
                - Color intensity: Correlation strength
                
                In Augur, this would reveal:
                - Relationships between different metrics
                - Impact of community size on productivity
                - Dependencies between different activities
                """)
            
        with stats_col3:
            st.markdown("#### System Load")
            fig = plot_system_load()
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("🖥️ About System Load"):
                st.markdown("""
                **System Load** monitors performance metrics:
                - Green line: Current load
                - Red dashed line: Critical threshold
                
                In production, this would track:
                - Database query performance
                - API response times
                - Resource utilization
                - System health metrics
                """)

        # Database Performance section
        st.subheader("Database Performance")
        db_col1, db_col2 = st.columns(2)
        
        with db_col1:
            st.markdown("#### Query Performance")
            fig = plot_query_performance()
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("⚡ About Query Performance"):
                st.markdown("""
                **Query Performance** compares different database approaches:
                - Main DB: Direct database queries
                - Replica DB: Read-replica queries
                - Optimized: Query with indexing and caching
                
                This demonstrates:
                - Performance optimization strategies
                - Query response times
                - System scalability
                """)
            
        with db_col2:
            st.markdown("#### Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Response Time", "45ms", "-23%")
            with col2:
                st.metric("Query Success Rate", "99.9%", "0.5%")
            with st.expander("📊 About Performance Metrics"):
                st.markdown("""
                **Performance Metrics** track system health:
                - Response Time: Average query duration
                - Success Rate: Percentage of successful queries
                
                In Augur, this would monitor:
                - API performance
                - Data collection efficiency
                - System reliability
                """)

        # Add footer with data refresh information
        st.markdown("---")
        st.caption("""
        📝 **Note**: This is a demonstration dashboard using simulated data. 
        In production, it would be connected to Augur's database and show real-time metrics 
        from open source projects. Last updated: {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        st.error(f"An error occurred while creating the dashboard: {str(e)}")
        st.error("Please check the logs for details.")

def plot_distribution_analysis(df: pd.DataFrame) -> go.Figure:
    """Create distribution analysis plot."""
    try:
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['value'],
            name='Distribution',
            nbinsx=30,
            showlegend=False
        ))
        
        # Add kernel density estimation
        kde_x, kde_y = calculate_kde(df['value'].values)
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Value Distribution Analysis",
            xaxis_title="Value",
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in distribution analysis: {str(e)}")
        return go.Figure()

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Create correlation matrix plot."""
    try:
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in correlation matrix: {str(e)}")
        return go.Figure()
    
# Additional helper functions

def plot_trend_prediction(df: pd.DataFrame) -> go.Figure:
    """Create trend prediction plot using exponential smoothing."""
    try:
        # Prepare data
        ts_data = df['value'].copy()
        
        # Fit exponential smoothing model
        model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        ).fit()
        
        # Make prediction
        forecast = model.forecast(30)
        
        # Create plot
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ts_data,
            mode='lines',
            name='Actual Data',
            line=dict(color='blue')
        ))
        
        # Add prediction
        fig.add_trace(go.Scatter(
            x=pd.RangeIndex(start=len(ts_data), stop=len(ts_data)+30),
            y=forecast,
            mode='lines',
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Trend Prediction",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in trend prediction: {str(e)}")
        return go.Figure()

def plot_query_performance() -> go.Figure:
    """Create query performance comparison plot."""
    try:
        # Example data
        methods = ['Main DB', 'Replica DB', 'Optimized']
        times = [100, 45, 20]  # Example times in milliseconds
        
        fig = go.Figure(data=[
            go.Bar(
                x=methods,
                y=times,
                text=times,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Query Performance Comparison",
            xaxis_title="Method",
            yaxis_title="Response Time (ms)",
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in query performance plot: {str(e)}")
        return go.Figure()

def plot_system_load() -> go.Figure:
    """Create system load analysis plot."""
    try:
        # Example data
        time_points = pd.date_range(start='2024-01-01', periods=24, freq='H')
        load_values = np.random.normal(60, 15, 24)  # Example load percentages
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=load_values,
            mode='lines+markers',
            name='System Load',
            line=dict(color='green', width=2)
        ))
        
        # Add threshold line
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Load Threshold"
        )
        
        fig.update_layout(
            title="System Load Over Time",
            xaxis_title="Time",
            yaxis_title="System Load (%)",
            showlegend=True
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in system load plot: {str(e)}")
        return go.Figure()

def calculate_advanced_metrics(df: pd.DataFrame) -> Dict:
    """Calculate advanced statistical metrics."""
    try:
        metrics = {
            'skewness': stats.skew(df['value']),
            'kurtosis': stats.kurtosis(df['value']),
            'normality_test': shapiro(df['value']),
            'trend_coefficient': np.polyfit(np.arange(len(df)), df['value'], 1)[0]
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {str(e)}")
        return {}

