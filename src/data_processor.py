"""
Data Processing Module for Amazon Sentiment Analysis
Handles loading, cleaning, and vectorizing review data for LSTM input.
"""

import pandas as pd
import numpy as np
import re
import pickle
from typing import Tuple, List, Optional
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataProcessor:
    """
    Processes Amazon review data for sentiment analysis.
    """
    
    def __init__(self, max_vocab_size: int = 10000, max_sequence_length: int = 200):
        """
        Initialize the data processor.
        
        Args:
            max_vocab_size: Maximum vocabulary size for tokenization
            max_sequence_length: Maximum sequence length for padding
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.word_index = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load review data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing review data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} reviews")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_sentiment_labels(self, ratings: pd.Series) -> np.ndarray:
        """
        Convert star ratings to binary sentiment labels.
        
        Args:
            ratings: Series of star ratings (1-5)
            
        Returns:
            Binary sentiment labels (0=negative, 1=positive)
        """
        # Convert ratings to binary sentiment
        # 1-2 stars = negative (0), 4-5 stars = positive (1)
        # We'll exclude 3-star reviews for clearer binary classification
        sentiment_labels = []
        
        for rating in ratings:
            if rating <= 2:
                sentiment_labels.append(0)  # Negative
            elif rating >= 4:
                sentiment_labels.append(1)  # Positive
            else:
                sentiment_labels.append(-1)  # Neutral (to be filtered out)
        
        return np.array(sentiment_labels)
    
    def build_vocabulary(self, texts: List[str]) -> dict:
        """
        Build vocabulary from text data.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary mapping words to indices
        """
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.max_vocab_size-2]  # Reserve space for PAD and UNK
        
        # Create word to index mapping
        word_to_index = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(top_words):
            word_to_index[word] = i + 2
        
        self.word_index = word_to_index
        logger.info(f"Built vocabulary with {len(word_to_index)} words")
        
        return word_to_index
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sequences (lists of integers)
        """
        if self.word_index is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        sequences = []
        for text in texts:
            words = text.split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index['<UNK>'])  # Unknown word
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pad sequences to uniform length.
        
        Args:
            sequences: List of sequences to pad
            
        Returns:
            Padded sequences as numpy array
        """
        padded = np.zeros((len(sequences), self.max_sequence_length))
        
        for i, seq in enumerate(sequences):
            if len(seq) > self.max_sequence_length:
                # Truncate if too long
                padded[i] = seq[:self.max_sequence_length]
            else:
                # Pad if too short
                padded[i, :len(seq)] = seq
        
        return padded
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by cleaning, tokenizing, and splitting.
        
        Args:
            df: DataFrame containing review data
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Clean review text
        logger.info("Cleaning text data...")
        df['cleaned_text'] = df['review_text'].apply(self.clean_text)
        
        # Create sentiment labels
        df['sentiment'] = self.create_sentiment_labels(df['rating'])
        
        # Filter out neutral reviews (rating = 3)
        df_filtered = df[df['sentiment'] != -1].copy()
        logger.info(f"Filtered dataset: {len(df_filtered)} reviews (removed neutral ratings)")
        
        # Get texts and labels
        texts = df_filtered['cleaned_text'].tolist()
        labels = df_filtered['sentiment'].values
        
        # Build vocabulary
        self.build_vocabulary(texts)
        
        # Convert texts to sequences
        sequences = self.texts_to_sequences(texts)
        
        # Pad sequences
        X = self.pad_sequences(sequences)
        y = labels.astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Positive samples: {np.sum(y == 1)}")
        logger.info(f"Negative samples: {np.sum(y == 0)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_tokenizer(self, file_path: str):
        """
        Save the tokenizer (word index) to a pickle file.
        
        Args:
            file_path: Path to save the tokenizer
        """
        if self.word_index is None:
            raise ValueError("No tokenizer to save. Build vocabulary first.")
        
        tokenizer_data = {
            'word_index': self.word_index,
            'max_vocab_size': self.max_vocab_size,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        logger.info(f"Tokenizer saved to {file_path}")
    
    def load_tokenizer(self, file_path: str):
        """
        Load the tokenizer from a pickle file.
        
        Args:
            file_path: Path to load the tokenizer from
        """
        with open(file_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.word_index = tokenizer_data['word_index']
        self.max_vocab_size = tokenizer_data['max_vocab_size']
        self.max_sequence_length = tokenizer_data['max_sequence_length']
        
        logger.info(f"Tokenizer loaded from {file_path}")
    
    def preprocess_single_text(self, text: str) -> np.ndarray:
        """
        Preprocess a single text for prediction.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text as numpy array
        """
        if self.word_index is None:
            raise ValueError("Tokenizer not loaded. Load tokenizer first.")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Convert to sequence
        sequences = self.texts_to_sequences([cleaned_text])
        
        # Pad sequence
        X = self.pad_sequences(sequences)
        
        return X

def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: DataFrame containing review data
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_reviews': len(df),
        'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
        'verified_purchases': df['verified_purchase'].sum(),
        'avg_helpful_votes': df['helpful_votes'].mean(),
        'unique_products': df['product_id'].nunique(),
        'unique_users': df['user_id'].nunique(),
        'avg_review_length': df['review_text'].str.len().mean()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    processor = ReviewDataProcessor()
    
    # Load and process data
    df = processor.load_data('../data/raw_reviews.csv')
    
    # Get statistics
    stats = get_data_stats(df)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = processor.prepare_data(df)
    
    # Save tokenizer
    processor.save_tokenizer('../models/tokenizer.pkl')