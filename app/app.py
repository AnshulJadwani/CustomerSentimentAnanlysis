"""
Customer Sentiment Analysis - Streamlit Web Application
This app allows users to predict sentiment of product reviews in real-time.
"""

import streamlit as st
import joblib
import json
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing utilities
from utils.preprocessing import clean_text, get_sentiment_label

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .sentiment-positive {
        color: #28a745;
        font-size: 24px;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and vectorizer."""
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        # Load best model
        best_model = joblib.load(models_dir / 'best_model.pkl')
        
        # Load vectorizer
        vectorizer = joblib.load(models_dir / 'tfidf_vectorizer.pkl')
        
        # Load metadata
        with open(models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Try to load all models
        all_models = {}
        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl',
            'SVM': 'svm_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                all_models[model_name] = joblib.load(model_path)
        
        return best_model, vectorizer, metadata, all_models
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run the Jupyter notebook first to train and save the models.")
        return None, None, None, None


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text."""
    # Clean text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return None, None, None
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    sentiment_label = get_sentiment_label(prediction)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vectorized)[0]
    
    return prediction, sentiment_label, probabilities


def get_sentiment_color(sentiment):
    """Get color for sentiment."""
    colors = {
        'Positive': '#28a745',
        'Neutral': '#ffc107',
        'Negative': '#dc3545'
    }
    return colors.get(sentiment, '#6c757d')


def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment."""
    emojis = {
        'Positive': '😊',
        'Neutral': '😐',
        'Negative': '😞'
    }
    return emojis.get(sentiment, '🤔')


def main():
    """Main application function."""
    
    # Load models
    best_model, vectorizer, metadata, all_models = load_models()
    
    if best_model is None:
        st.stop()
    
    # Header
    st.title("🎯 Customer Sentiment Analysis")
    st.markdown("### Analyze sentiment of product reviews in real-time")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        if metadata:
            st.info(f"**Best Model:** {metadata.get('best_model', 'N/A')}")
            st.metric("Training Samples", metadata.get('training_samples', 'N/A'))
            st.metric("Test Samples", metadata.get('test_samples', 'N/A'))
            st.metric("Features", metadata.get('feature_count', 'N/A'))
        
        st.markdown("---")
        
        # Model selection
        st.header("🤖 Model Selection")
        available_models = list(all_models.keys()) if all_models else [metadata.get('best_model', 'Best Model')]
        
        selected_model_name = st.selectbox(
            "Choose a model:",
            available_models,
            index=0
        )
        
        # Use selected model
        selected_model = all_models.get(selected_model_name, best_model) if all_models else best_model
        
        st.markdown("---")
        
        # Model performance
        if metadata and 'model_performance' in metadata:
            st.header("📈 Model Performance")
            perf_df = pd.DataFrame(metadata['model_performance'])
            
            for _, row in perf_df.iterrows():
                if row['Model'] == selected_model_name:
                    st.metric("Accuracy", f"{row['Accuracy']:.2%}")
                    st.metric("Precision", f"{row['Precision']:.2%}")
                    st.metric("Recall", f"{row['Recall']:.2%}")
                    st.metric("F1-Score", f"{row['F1-Score']:.2%}")
                    break
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        This application uses Machine Learning to analyze 
        the sentiment of customer product reviews.
        
        **Sentiment Classes:**
        - 😊 Positive
        - 😐 Neutral
        - 😞 Negative
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Review Text")
        
        # Text input
        review_text = st.text_area(
            "Type or paste a product review:",
            height=150,
            placeholder="Example: This phone is amazing! Great camera quality and long battery life. Highly recommended!",
            help="Enter a product review to analyze its sentiment"
        )
        
        # Predict button
        predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        # Sample reviews
        st.markdown("---")
        st.subheader("💡 Try Sample Reviews")
        
        sample_reviews = {
            "Positive": "Absolutely love this product! Excellent quality and fast delivery. Best purchase ever!",
            "Negative": "Terrible experience. Product arrived damaged and customer service was unhelpful. Very disappointed.",
            "Neutral": "It's okay. Does what it's supposed to do. Nothing special but gets the job done."
        }
        
        col_sample1, col_sample2, col_sample3 = st.columns(3)
        
        with col_sample1:
            if st.button("😊 Positive Sample", use_container_width=True):
                review_text = sample_reviews["Positive"]
                st.rerun()
        
        with col_sample2:
            if st.button("😐 Neutral Sample", use_container_width=True):
                review_text = sample_reviews["Neutral"]
                st.rerun()
        
        with col_sample3:
            if st.button("😞 Negative Sample", use_container_width=True):
                review_text = sample_reviews["Negative"]
                st.rerun()
    
    with col2:
        st.header("🎯 Results")
        
        result_placeholder = st.empty()
        
        # Display instructions if no prediction yet
        if not predict_button and not review_text:
            with result_placeholder.container():
                st.info("👈 Enter a review and click 'Analyze Sentiment' to see results")
    
    # Prediction
    if predict_button and review_text:
        if len(review_text.strip()) < 10:
            st.warning("⚠️ Please enter a longer review (at least 10 characters)")
        else:
            with st.spinner("Analyzing sentiment..."):
                prediction, sentiment_label, probabilities = predict_sentiment(
                    review_text, selected_model, vectorizer
                )
                
                if sentiment_label is None:
                    st.error("❌ Could not analyze the review. Please try again with different text.")
                else:
                    # Display results
                    with col2:
                        with result_placeholder.container():
                            # Sentiment result
                            emoji = get_sentiment_emoji(sentiment_label)
                            color = get_sentiment_color(sentiment_label)
                            
                            st.markdown(
                                f"<div style='text-align: center; padding: 2rem; background-color: {color}20; border-radius: 10px; border: 2px solid {color};'>"
                                f"<h1 style='color: {color};'>{emoji}</h1>"
                                f"<h2 style='color: {color};'>{sentiment_label}</h2>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Confidence scores
                            if probabilities is not None:
                                st.subheader("📊 Confidence Scores")
                                
                                labels = ['Negative', 'Neutral', 'Positive']
                                colors_list = ['#dc3545', '#ffc107', '#28a745']
                                
                                # Create bar chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=labels,
                                        y=probabilities * 100,
                                        marker_color=colors_list,
                                        text=[f"{p:.1f}%" for p in probabilities * 100],
                                        textposition='auto',
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Probability Distribution",
                                    xaxis_title="Sentiment",
                                    yaxis_title="Confidence (%)",
                                    yaxis_range=[0, 100],
                                    showlegend=False,
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display confidence metrics
                                for i, (label, prob) in enumerate(zip(labels, probabilities)):
                                    st.metric(
                                        label=f"{label} Confidence",
                                        value=f"{prob*100:.2f}%"
                                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Built with ❤️ using Streamlit | Customer Sentiment Analysis Project</p>
            <p>Model trained on e-commerce product reviews</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
