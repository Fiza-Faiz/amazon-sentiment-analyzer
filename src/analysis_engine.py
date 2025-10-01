"""
Analysis Engine for Amazon Sentiment Analysis
Combines LSTM predictions with NLTK-based sentiment analysis and provides business insights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import re
from collections import Counter

# NLTK imports
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    from nltk.probability import FreqDist
    
    # Download required NLTK data
    def download_nltk_data():
        """Download required NLTK datasets."""
        datasets = ['vader_lexicon', 'punkt', 'stopwords', 'averaged_perceptron_tagger']
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                try:
                    nltk.download(dataset, quiet=True)
                except Exception as e:
                    print(f"Warning: Could not download {dataset}: {e}")
    
    # Attempt to download NLTK data
    download_nltk_data()
    
except ImportError:
    print("NLTK not installed. Please install: pip install nltk")
    raise

# Local imports
from data_processor import ReviewDataProcessor
from model_trainer import LSTMSentimentTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisEngine:
    """
    Comprehensive sentiment analysis engine combining LSTM and NLTK approaches.
    """
    
    def __init__(self, lstm_model_path: Optional[str] = None, 
                 tokenizer_path: Optional[str] = None):
        """
        Initialize the analysis engine.
        
        Args:
            lstm_model_path: Path to trained LSTM model
            tokenizer_path: Path to saved tokenizer
        """
        self.lstm_trainer = None
        self.data_processor = ReviewDataProcessor()
        self.sia = SentimentIntensityAnalyzer()  # VADER sentiment analyzer
        self.stemmer = PorterStemmer()
        
        # Load models if paths provided
        if lstm_model_path and tokenizer_path:
            self.load_models(lstm_model_path, tokenizer_path)
        
        # Load English stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("English stopwords not found. Using empty set.")
            self.stop_words = set()
    
    def load_models(self, lstm_model_path: str, tokenizer_path: str):
        """
        Load trained LSTM model and tokenizer.
        
        Args:
            lstm_model_path: Path to LSTM model
            tokenizer_path: Path to tokenizer
        """
        try:
            # Load tokenizer first to get vocab size
            self.data_processor.load_tokenizer(tokenizer_path)
            vocab_size = len(self.data_processor.word_index)
            
            # Initialize and load LSTM model
            self.lstm_trainer = LSTMSentimentTrainer(
                vocab_size=vocab_size,
                max_sequence_length=self.data_processor.max_sequence_length
            )
            self.lstm_trainer.load_model(lstm_model_path)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def analyze_text_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using NLTK's VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        scores = self.sia.polarity_scores(text)
        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu']
        }
    
    def analyze_text_with_lstm(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using trained LSTM model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with LSTM predictions
        """
        if self.lstm_trainer is None or self.data_processor.word_index is None:
            logger.warning("LSTM model not loaded. Returning default values.")
            return {
                'lstm_probability': 0.5,
                'lstm_prediction': 0,
                'lstm_confidence': 0.0
            }
        
        try:
            # Preprocess text
            X = self.data_processor.preprocess_single_text(text)
            
            # Get prediction
            probability = self.lstm_trainer.predict(X)[0]
            prediction = int(probability >= 0.5)
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            
            return {
                'lstm_probability': float(probability),
                'lstm_prediction': prediction,
                'lstm_confidence': float(confidence)
            }
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return {
                'lstm_probability': 0.5,
                'lstm_prediction': 0,
                'lstm_confidence': 0.0
            }
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various text features using NLTK.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text features
        """
        # Clean text
        clean_text = self.data_processor.clean_text(text)
        
        # Basic features
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Tokenize and get word frequencies
        try:
            tokens = word_tokenize(clean_text.lower())
            filtered_tokens = [word for word in tokens if word not in self.stop_words and word.isalpha()]
            
            if filtered_tokens:
                freq_dist = FreqDist(filtered_tokens)
                features.update({
                    'unique_words': len(set(filtered_tokens)),
                    'most_common_words': freq_dist.most_common(5),
                    'vocabulary_richness': len(set(filtered_tokens)) / len(filtered_tokens)
                })
            else:
                features.update({
                    'unique_words': 0,
                    'most_common_words': [],
                    'vocabulary_richness': 0
                })
        except Exception as e:
            logger.warning(f"Error in text feature extraction: {e}")
            features.update({
                'unique_words': 0,
                'most_common_words': [],
                'vocabulary_richness': 0
            })
        
        return features
    
    def comprehensive_analysis(self, text: str, actual_rating: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis combining LSTM and NLTK.
        
        Args:
            text: Text to analyze
            actual_rating: Actual star rating (if available)
            
        Returns:
            Complete analysis results
        """
        # Get VADER sentiment
        vader_results = self.analyze_text_with_vader(text)
        
        # Get LSTM sentiment
        lstm_results = self.analyze_text_with_lstm(text)
        
        # Extract text features
        text_features = self.extract_text_features(text)
        
        # Combine results
        analysis = {
            'text': text,
            'actual_rating': actual_rating,
            **vader_results,
            **lstm_results,
            **text_features
        }
        
        # Add interpretations
        analysis['vader_sentiment'] = self._interpret_vader_score(vader_results['vader_compound'])
        analysis['lstm_sentiment'] = 'positive' if lstm_results['lstm_prediction'] == 1 else 'negative'
        
        # Check for sentiment conflicts
        vader_binary = 1 if vader_results['vader_compound'] > 0 else 0
        lstm_binary = lstm_results['lstm_prediction']
        analysis['sentiment_agreement'] = vader_binary == lstm_binary
        
        # Rating vs sentiment analysis
        if actual_rating is not None:
            analysis['rating_lstm_conflict'] = self._check_rating_sentiment_conflict(
                actual_rating, lstm_results['lstm_prediction']
            )
            analysis['rating_vader_conflict'] = self._check_rating_sentiment_conflict(
                actual_rating, vader_binary
            )
        
        return analysis
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze entire dataset and generate insights.
        
        Args:
            df: DataFrame with reviews
            
        Returns:
            Dataset-wide analysis results
        """
        logger.info("Analyzing dataset...")
        
        results = []
        for _, row in df.iterrows():
            try:
                analysis = self.comprehensive_analysis(
                    row['review_text'], 
                    row['rating']
                )
                analysis['review_id'] = row['review_id']
                analysis['product_id'] = row['product_id']
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Error analyzing review {row.get('review_id', 'unknown')}: {e}")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Generate insights
        insights = self._generate_insights(results_df)
        
        return {
            'individual_results': results,
            'results_dataframe': results_df,
            'insights': insights
        }
    
    def find_misclassified_reviews(self, results_df: pd.DataFrame, 
                                 top_n: int = 10) -> Dict[str, List[Dict]]:
        """
        Find reviews where LSTM sentiment conflicts with star rating.
        
        Args:
            results_df: DataFrame with analysis results
            top_n: Number of top examples to return
            
        Returns:
            Dictionary with misclassified examples
        """
        # High-rated reviews with negative LSTM sentiment
        high_rated_negative = results_df[
            (results_df['actual_rating'] >= 4) & 
            (results_df['lstm_prediction'] == 0)
        ].nlargest(top_n, 'lstm_confidence')
        
        # Low-rated reviews with positive LSTM sentiment
        low_rated_positive = results_df[
            (results_df['actual_rating'] <= 2) & 
            (results_df['lstm_prediction'] == 1)
        ].nlargest(top_n, 'lstm_confidence')
        
        return {
            'high_rated_negative_sentiment': high_rated_negative.to_dict('records'),
            'low_rated_positive_sentiment': low_rated_positive.to_dict('records')
        }
    
    def extract_frequent_words_in_misclassified(self, results_df: pd.DataFrame,
                                              top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract most frequent words in misclassified reviews.
        
        Args:
            results_df: DataFrame with analysis results
            top_n: Number of top words to return
            
        Returns:
            Dictionary with frequent words in different categories
        """
        # Get misclassified reviews
        high_rated_negative = results_df[
            (results_df['actual_rating'] >= 4) & 
            (results_df['lstm_prediction'] == 0)
        ]
        
        low_rated_positive = results_df[
            (results_df['actual_rating'] <= 2) & 
            (results_df['lstm_prediction'] == 1)
        ]
        
        def extract_words(texts):
            all_words = []
            for text in texts:
                clean_text = self.data_processor.clean_text(str(text))
                tokens = word_tokenize(clean_text.lower())
                filtered_tokens = [
                    word for word in tokens 
                    if word not in self.stop_words and word.isalpha() and len(word) > 2
                ]
                all_words.extend(filtered_tokens)
            return Counter(all_words).most_common(top_n)
        
        return {
            'high_rated_negative_words': extract_words(high_rated_negative['text'].tolist()),
            'low_rated_positive_words': extract_words(low_rated_positive['text'].tolist())
        }
    
    def _interpret_vader_score(self, compound_score: float) -> str:
        """Interpret VADER compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _check_rating_sentiment_conflict(self, rating: int, sentiment_prediction: int) -> bool:
        """Check if rating conflicts with sentiment prediction."""
        if rating <= 2:  # Negative rating
            return sentiment_prediction == 1  # But positive sentiment
        elif rating >= 4:  # Positive rating
            return sentiment_prediction == 0  # But negative sentiment
        else:  # Neutral rating
            return False  # No conflict for neutral ratings
    
    def _generate_insights(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate business insights from analysis results."""
        total_reviews = len(results_df)
        
        if total_reviews == 0:
            return {}
        
        # Basic sentiment distribution
        lstm_positive = (results_df['lstm_prediction'] == 1).sum()
        vader_positive = (results_df['vader_compound'] > 0).sum()
        
        # Rating vs sentiment correlation
        rating_lstm_conflicts = results_df['rating_lstm_conflict'].sum() if 'rating_lstm_conflict' in results_df.columns else 0
        rating_vader_conflicts = results_df['rating_vader_conflict'].sum() if 'rating_vader_conflict' in results_df.columns else 0
        
        # Sentiment agreement between methods
        sentiment_agreement = results_df['sentiment_agreement'].sum()
        
        # Average confidence scores
        avg_lstm_confidence = results_df['lstm_confidence'].mean()
        avg_vader_compound = results_df['vader_compound'].mean()
        
        insights = {
            'total_reviews_analyzed': total_reviews,
            'lstm_positive_percentage': (lstm_positive / total_reviews) * 100,
            'vader_positive_percentage': (vader_positive / total_reviews) * 100,
            'sentiment_agreement_percentage': (sentiment_agreement / total_reviews) * 100,
            'rating_lstm_conflict_percentage': (rating_lstm_conflicts / total_reviews) * 100,
            'rating_vader_conflict_percentage': (rating_vader_conflicts / total_reviews) * 100,
            'avg_lstm_confidence': avg_lstm_confidence,
            'avg_vader_score': avg_vader_compound,
            'lstm_vs_vader_correlation': results_df['lstm_probability'].corr(results_df['vader_compound'])
        }
        
        return insights

if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Engine Example")
    
    # Initialize engine
    engine = SentimentAnalysisEngine()
    
    # Example texts
    positive_text = "This product is absolutely amazing! Great quality and fast shipping."
    negative_text = "Terrible product. Broke immediately and customer service was awful."
    neutral_text = "It's okay. Does what it's supposed to do but nothing special."
    
    # Analyze texts
    for text, label in [(positive_text, "Positive"), (negative_text, "Negative"), (neutral_text, "Neutral")]:
        print(f"\n{label} Example:")
        print(f"Text: {text}")
        
        # VADER analysis
        vader_results = engine.analyze_text_with_vader(text)
        print(f"VADER Score: {vader_results['vader_compound']:.3f}")
        
        # Text features
        features = engine.extract_text_features(text)
        print(f"Word Count: {features['word_count']}")
        print(f"Vocabulary Richness: {features['vocabulary_richness']:.3f}")
    
    print("\nNote: To use LSTM analysis, load trained models with load_models() method.")