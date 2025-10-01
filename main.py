#!/usr/bin/env python3
"""
Amazon Sentiment Analyzer - Main Application
Entry point for running sentiment analysis on Amazon product reviews.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from src.data_processor import ReviewDataProcessor, get_data_stats
from src.model_trainer import LSTMSentimentTrainer
from src.analysis_engine import SentimentAnalysisEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(data_file: str, model_save_path: str, tokenizer_save_path: str,
                epochs: int = 30, batch_size: int = 32):
    """
    Train a new LSTM sentiment model.
    
    Args:
        data_file: Path to the training data CSV
        model_save_path: Path to save the trained model
        tokenizer_save_path: Path to save the tokenizer
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    logger.info("Starting model training...")
    
    # Initialize data processor
    processor = ReviewDataProcessor()
    
    # Load and prepare data
    logger.info(f"Loading data from {data_file}")
    df = processor.load_data(data_file)
    
    # Show dataset statistics
    stats = get_data_stats(df)
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = processor.prepare_data(df)
    
    # Get vocabulary size
    vocab_size = len(processor.word_index)
    
    # Initialize and build model
    trainer = LSTMSentimentTrainer(
        vocab_size=vocab_size,
        embedding_dim=100,
        lstm_units=128,
        max_sequence_length=processor.max_sequence_length
    )
    
    model = trainer.build_model(
        dropout_rate=0.3,
        recurrent_dropout=0.3,
        l2_reg=0.01,
        bidirectional=True
    )
    
    # Train model
    logger.info("Training model...")
    history = trainer.train_model(
        X_train, y_train, X_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model and tokenizer
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)
    
    trainer.save_model(model_save_path)
    processor.save_tokenizer(tokenizer_save_path)
    
    logger.info(f"Model saved to {model_save_path}")
    logger.info(f"Tokenizer saved to {tokenizer_save_path}")
    
    return history

def analyze_reviews(data_file: str, model_path: str = None, tokenizer_path: str = None,
                   output_file: str = None):
    """
    Analyze sentiment of reviews in the dataset.
    
    Args:
        data_file: Path to the reviews CSV file
        model_path: Path to trained LSTM model (optional)
        tokenizer_path: Path to tokenizer (optional)
        output_file: Path to save analysis results (optional)
    """
    logger.info("Starting sentiment analysis...")
    
    # Initialize analysis engine
    if model_path and tokenizer_path:
        try:
            engine = SentimentAnalysisEngine(model_path, tokenizer_path)
            logger.info("Loaded LSTM model for analysis")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            logger.info("Proceeding with VADER-only analysis")
            engine = SentimentAnalysisEngine()
    else:
        logger.info("No LSTM model provided, using VADER-only analysis")
        engine = SentimentAnalysisEngine()
    
    # Load data
    processor = ReviewDataProcessor()
    df = processor.load_data(data_file)
    
    # Analyze dataset
    logger.info("Analyzing reviews...")
    results = engine.analyze_dataset(df)
    
    # Display insights
    insights = results['insights']
    logger.info("Analysis Complete! Key Insights:")
    logger.info(f"  Total reviews analyzed: {insights.get('total_reviews_analyzed', 0)}")
    logger.info(f"  LSTM positive sentiment: {insights.get('lstm_positive_percentage', 0):.1f}%")
    logger.info(f"  VADER positive sentiment: {insights.get('vader_positive_percentage', 0):.1f}%")
    logger.info(f"  Sentiment agreement: {insights.get('sentiment_agreement_percentage', 0):.1f}%")
    logger.info(f"  Rating-LSTM conflicts: {insights.get('rating_lstm_conflict_percentage', 0):.1f}%")
    logger.info(f"  LSTM-VADER correlation: {insights.get('lstm_vs_vader_correlation', 0):.3f}")
    
    # Find misclassified examples
    if 'lstm_prediction' in results['results_dataframe'].columns:
        misclassified = engine.find_misclassified_reviews(results['results_dataframe'], top_n=5)
        
        logger.info("\nTop Misclassified Examples:")
        
        # High-rated but negative sentiment
        high_neg = misclassified['high_rated_negative_sentiment']
        if high_neg:
            logger.info("High-rated reviews with negative predicted sentiment:")
            for i, review in enumerate(high_neg[:3], 1):
                logger.info(f"  {i}. Rating: {review['actual_rating']}, "
                          f"LSTM: {review['lstm_probability']:.3f}, "
                          f"Text: {review['text'][:100]}...")
        
        # Low-rated but positive sentiment
        low_pos = misclassified['low_rated_positive_sentiment']
        if low_pos:
            logger.info("Low-rated reviews with positive predicted sentiment:")
            for i, review in enumerate(low_pos[:3], 1):
                logger.info(f"  {i}. Rating: {review['actual_rating']}, "
                          f"LSTM: {review['lstm_probability']:.3f}, "
                          f"Text: {review['text'][:100]}...")
    
    # Save results if requested
    if output_file:
        logger.info(f"Saving results to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save insights as JSON
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_insights = {}
            for k, v in insights.items():
                if hasattr(v, 'item'):  # numpy scalar
                    serializable_insights[k] = v.item()
                else:
                    serializable_insights[k] = v
            
            json.dump(serializable_insights, f, indent=2)
        
        # Save detailed results
        results_csv_file = output_file.replace('.json', '_detailed.csv')
        results['results_dataframe'].to_csv(results_csv_file, index=False)
        logger.info(f"Detailed results saved to {results_csv_file}")
    
    return results

def analyze_single_text(text: str, model_path: str = None, tokenizer_path: str = None):
    """
    Analyze sentiment of a single text.
    
    Args:
        text: Text to analyze
        model_path: Path to trained LSTM model (optional)
        tokenizer_path: Path to tokenizer (optional)
    """
    logger.info("Analyzing single text...")
    
    # Initialize analysis engine
    if model_path and tokenizer_path:
        try:
            engine = SentimentAnalysisEngine(model_path, tokenizer_path)
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            engine = SentimentAnalysisEngine()
    else:
        engine = SentimentAnalysisEngine()
    
    # Analyze text
    results = engine.comprehensive_analysis(text)
    
    # Display results
    print(f"\nText: {text}")
    print(f"VADER Score: {results['vader_compound']:.3f} ({results['vader_sentiment']})")
    
    if 'lstm_probability' in results:
        print(f"LSTM Score: {results['lstm_probability']:.3f} ({results['lstm_sentiment']})")
        print(f"LSTM Confidence: {results['lstm_confidence']:.3f}")
    
    print(f"Word Count: {results['word_count']}")
    print(f"Vocabulary Richness: {results['vocabulary_richness']:.3f}")
    
    return results

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Amazon Product Reviews Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --data data/raw_reviews.csv --epochs 20
  
  # Analyze reviews with trained model
  python main.py analyze --data data/raw_reviews.csv --model models/lstm_sentiment_model.h5 --tokenizer models/tokenizer.pkl
  
  # Analyze reviews with VADER only
  python main.py analyze --data data/raw_reviews.csv
  
  # Analyze single text
  python main.py single --text "This product is amazing!"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new LSTM model')
    train_parser.add_argument('--data', required=True, help='Path to training data CSV')
    train_parser.add_argument('--model', default='models/lstm_sentiment_model.h5', help='Path to save model')
    train_parser.add_argument('--tokenizer', default='models/tokenizer.pkl', help='Path to save tokenizer')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze sentiment of reviews')
    analyze_parser.add_argument('--data', required=True, help='Path to reviews CSV')
    analyze_parser.add_argument('--model', help='Path to trained LSTM model')
    analyze_parser.add_argument('--tokenizer', help='Path to tokenizer')
    analyze_parser.add_argument('--output', help='Path to save analysis results')
    
    # Single text command
    single_parser = subparsers.add_parser('single', help='Analyze single text')
    single_parser.add_argument('--text', required=True, help='Text to analyze')
    single_parser.add_argument('--model', help='Path to trained LSTM model')
    single_parser.add_argument('--tokenizer', help='Path to tokenizer')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_model(
                args.data, args.model, args.tokenizer,
                args.epochs, args.batch_size
            )
        
        elif args.command == 'analyze':
            analyze_reviews(
                args.data, args.model, args.tokenizer, args.output
            )
        
        elif args.command == 'single':
            analyze_single_text(args.text, args.model, args.tokenizer)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()