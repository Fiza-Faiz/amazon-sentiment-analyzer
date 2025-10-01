"""
LSTM Model Trainer for Amazon Sentiment Analysis
Defines, trains, and saves LSTM models for sentiment classification.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import os

# TensorFlow and Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
except ImportError:
    print("TensorFlow not installed. Please install: pip install tensorflow")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMSentimentTrainer:
    """
    LSTM model trainer for sentiment analysis.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, lstm_units: int = 128,
                 max_sequence_length: int = 200):
        """
        Initialize the LSTM trainer.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units
            max_sequence_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_sequence_length = max_sequence_length
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def build_model(self, dropout_rate: float = 0.3, recurrent_dropout: float = 0.3,
                   l2_reg: float = 0.01, bidirectional: bool = True) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Args:
            dropout_rate: Dropout rate for regularization
            recurrent_dropout: Recurrent dropout rate for LSTM
            l2_reg: L2 regularization factor
            bidirectional: Whether to use bidirectional LSTM
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model...")
        
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                mask_zero=True,  # Handle padding
                embeddings_regularizer=l2(l2_reg),
                name='embedding'
            ),
            
            # Dropout for embedding
            Dropout(dropout_rate, name='embedding_dropout'),
            
            # LSTM layer(s)
            Bidirectional(
                LSTM(
                    self.lstm_units,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg),
                    return_sequences=False,  # Only return last output
                    name='lstm_layer'
                ),
                name='bidirectional_lstm'
            ) if bidirectional else LSTM(
                self.lstm_units,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg),
                return_sequences=False,
                name='lstm_layer'
            ),
            
            # Dense layer with dropout
            Dense(64, activation='relu', kernel_regularizer=l2(l2_reg), name='dense_1'),
            Dropout(dropout_rate, name='dense_dropout'),
            
            # Output layer
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        # Print model summary
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   epochs: int = 50, batch_size: int = 32,
                   validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of training data for validation
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            raise ValueError("Model not built")
        
        logger.info(f"Training model for {epochs} epochs...")
        
        # Prepare callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath='models/best_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1-Score: {test_f1:.4f}")
        
        # Store history and test results
        self.history = history
        
        # Add test results to history
        history_dict = history.history.copy()
        history_dict['test_loss'] = test_loss
        history_dict['test_accuracy'] = test_accuracy
        history_dict['test_precision'] = test_precision
        history_dict['test_recall'] = test_recall
        history_dict['test_f1'] = test_f1
        
        return history_dict
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            logger.error("No model to save. Train model first.")
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to load the model from
        """
        try:
            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            logger.error("No model loaded. Load or train model first.")
            raise ValueError("No model loaded")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_sentiment(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sentiment classes and probabilities.
        
        Args:
            X: Input data for prediction
            threshold: Classification threshold
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        probabilities = self.predict(X)
        predicted_classes = (probabilities >= threshold).astype(int)
        
        return predicted_classes, probabilities

def create_simple_lstm(vocab_size: int, embedding_dim: int = 50, 
                      max_sequence_length: int = 200) -> Sequential:
    """
    Create a simple LSTM model for quick testing.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        max_sequence_length: Maximum sequence length
        
    Returns:
        Simple LSTM model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive evaluation of the model.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    
    return metrics

if __name__ == "__main__":
    # Example usage
    print("LSTM Sentiment Trainer Example")
    
    # Model parameters
    vocab_size = 10000
    embedding_dim = 100
    lstm_units = 128
    max_sequence_length = 200
    
    # Create trainer
    trainer = LSTMSentimentTrainer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_sequence_length=max_sequence_length
    )
    
    # Build model
    model = trainer.build_model(
        dropout_rate=0.3,
        recurrent_dropout=0.3,
        l2_reg=0.01,
        bidirectional=True
    )
    
    print("Model built successfully!")
    print(f"Model has {model.count_params():,} trainable parameters")
    
    # Note: To train, you would need to load actual data:
    # from data_processor import ReviewDataProcessor
    # processor = ReviewDataProcessor()
    # df = processor.load_data('data/raw_reviews.csv')
    # X_train, X_test, y_train, y_test = processor.prepare_data(df)
    # history = trainer.train_model(X_train, y_train, X_test, y_test)
    # trainer.save_model('models/lstm_sentiment_model.h5')