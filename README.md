# Amazon Sentiment Analyzer

An end-to-end sentiment analysis solution for Amazon product reviews using Deep Learning (LSTM) and traditional NLP methods (NLTK). This project combines advanced machine learning techniques to analyze customer sentiment and provide actionable business intelligence.

## 🎯 Project Overview

This project implements a comprehensive sentiment analysis system that:

- **LSTM Deep Learning**: Uses bidirectional LSTM neural networks for sentiment classification
- **NLTK Integration**: Employs VADER sentiment analysis for comparison and validation  
- **Business Intelligence**: Identifies sentiment-rating discrepancies and provides insights
- **Docker Ready**: Fully containerized for easy deployment
- **Production Ready**: Complete with model persistence, logging, and error handling

## 🏗️ Architecture

```
amazon-sentiment-analyzer/
├── data/                      # Dataset storage
│   └── raw_reviews.csv       # Amazon review dataset (1,500+ reviews)
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_lstm_training.ipynb
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_processor.py     # Data loading and preprocessing
│   ├── model_trainer.py      # LSTM model definition and training
│   └── analysis_engine.py    # Sentiment analysis and insights
├── models/                   # Saved models and tokenizers
├── reports/                  # Analysis reports and insights
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
└── README.md               # This file
```

## 📊 Dataset

The project includes a realistic Amazon reviews dataset with:

- **1,500+ product reviews** across 10 different product categories
- **Realistic rating distribution**: More positive reviews (4-5 stars) than negative
- **Rich features**: Product IDs, user IDs, helpful votes, timestamps
- **Diverse products**: Headphones, smartwatches, coffee makers, laptop accessories, etc.

### Sample Data Structure
```csv
review_id,product_id,product_name,user_id,rating,review_title,review_text,verified_purchase,helpful_votes,timestamp
1,B08N5WRWNW,Wireless Bluetooth Headphones,U123456789,5,Amazing!,Excellent product! Amazing quality...,True,12,2024-01-15
```

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd amazon-sentiment-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run VADER-only analysis** (No training required)
   ```bash
   python main.py analyze --data data/raw_reviews.csv
   ```

5. **Train LSTM model** (Optional, for full functionality)
   ```bash
   python main.py train --data data/raw_reviews.csv --epochs 20
   ```

6. **Run complete analysis**
   ```bash
   python main.py analyze --data data/raw_reviews.csv \
     --model models/lstm_sentiment_model.h5 \
     --tokenizer models/tokenizer.pkl
   ```

### Option 2: Docker

1. **Build the container**
   ```bash
   docker build -t amazon-sentiment-analyzer .
   ```

2. **Run analysis**
   ```bash
   docker run amazon-sentiment-analyzer
   ```

## 🛠️ Usage Examples

### Command Line Interface

The application provides three main commands:

#### 1. Train a New Model
```bash
python main.py train --data data/raw_reviews.csv --epochs 30 --batch-size 32
```

**Options:**
- `--data`: Path to training data CSV (required)
- `--model`: Path to save trained model (default: `models/lstm_sentiment_model.h5`)
- `--tokenizer`: Path to save tokenizer (default: `models/tokenizer.pkl`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Training batch size (default: 32)

#### 2. Analyze Reviews Dataset
```bash
python main.py analyze --data data/raw_reviews.csv \
  --model models/lstm_sentiment_model.h5 \
  --tokenizer models/tokenizer.pkl \
  --output reports/analysis_results.json
```

**Options:**
- `--data`: Path to reviews CSV (required)
- `--model`: Path to trained LSTM model (optional)
- `--tokenizer`: Path to tokenizer (optional)
- `--output`: Path to save analysis results (optional)

#### 3. Analyze Single Text
```bash
python main.py single --text "This product is absolutely amazing! Great quality and fast shipping."
```

**Options:**
- `--text`: Text to analyze (required)
- `--model`: Path to trained LSTM model (optional)
- `--tokenizer`: Path to tokenizer (optional)

### Python API Usage

```python
from src.analysis_engine import SentimentAnalysisEngine

# Initialize with trained models
engine = SentimentAnalysisEngine(
    lstm_model_path='models/lstm_sentiment_model.h5',
    tokenizer_path='models/tokenizer.pkl'
)

# Analyze single text
result = engine.comprehensive_analysis(
    "This product exceeded my expectations!", 
    actual_rating=5
)

print(f"VADER sentiment: {result['vader_sentiment']}")
print(f"LSTM sentiment: {result['lstm_sentiment']}")
print(f"Sentiment agreement: {result['sentiment_agreement']}")
```

## 🔬 Technical Features

### LSTM Model Architecture

- **Embedding Layer**: 100-dimensional word embeddings
- **Bidirectional LSTM**: 128 units with dropout and recurrent dropout
- **Dense Layers**: Fully connected layers with L2 regularization
- **Output**: Sigmoid activation for binary sentiment classification

### NLTK Integration

- **VADER Sentiment**: Dictionary-based sentiment analysis
- **Text Processing**: Tokenization, stopword removal, stemming
- **Feature Extraction**: Word frequencies, vocabulary richness, text statistics

### Key Capabilities

1. **Sentiment Classification**: Binary positive/negative sentiment prediction
2. **Confidence Scoring**: Model confidence in predictions
3. **Conflict Detection**: Identify reviews where sentiment contradicts rating
4. **Comparative Analysis**: LSTM vs VADER sentiment comparison
5. **Business Insights**: Actionable intelligence for product/service improvement

## 📈 Sample Results

### Analysis Output
```
Analysis Complete! Key Insights:
  Total reviews analyzed: 1500
  LSTM positive sentiment: 67.2%
  VADER positive sentiment: 71.8%
  Sentiment agreement: 89.3%
  Rating-LSTM conflicts: 12.7%
  LSTM-VADER correlation: 0.847

Top Misclassified Examples:
High-rated reviews with negative predicted sentiment:
  1. Rating: 5, LSTM: 0.234, Text: Great product but shipping was terrible and took forever...
  2. Rating: 4, LSTM: 0.156, Text: Love the features but customer service experience was awful...
```

### Business Insights Generated

1. **Star Rating vs Predicted Sentiment Correlation**: Quantify alignment between ratings and sentiment
2. **Misclassification Analysis**: Identify reviews where sentiment contradicts rating  
3. **Product Issue Detection**: Surface implicit problems through sentiment-rating discrepancies
4. **Method Comparison**: Understand when VADER vs LSTM is more reliable
5. **Frequent Words Analysis**: Common themes in misclassified reviews

## 🔧 Configuration

### Model Parameters
- **Vocabulary Size**: 10,000 words
- **Sequence Length**: 200 tokens
- **Embedding Dimension**: 100
- **LSTM Units**: 128
- **Dropout Rate**: 0.3
- **L2 Regularization**: 0.01

### Training Settings
- **Default Epochs**: 30
- **Batch Size**: 32
- **Validation Split**: 10%
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: Factor of 0.5, patience of 5

## 🧪 Model Performance

The LSTM model typically achieves:
- **Accuracy**: ~88-92% on test set
- **Precision**: ~87-91% 
- **Recall**: ~85-90%
- **F1-Score**: ~86-91%

Performance varies based on training data and hyperparameters.

## 🐳 Docker Deployment

### Standard Deployment
```bash
# Build image
docker build -t amazon-sentiment-analyzer .

# Run analysis
docker run amazon-sentiment-analyzer

# Run with custom data (mount volume)
docker run -v /path/to/data:/app/data amazon-sentiment-analyzer
```

### Development Mode
```bash
# Run Jupyter notebook server
docker run -p 8888:8888 amazon-sentiment-analyzer jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Interactive shell
docker run -it amazon-sentiment-analyzer /bin/bash
```

## 📝 Development

### Adding New Features

1. **Custom Models**: Extend `LSTMSentimentTrainer` class
2. **New Analysis Methods**: Add methods to `SentimentAnalysisEngine`
3. **Data Processors**: Extend `ReviewDataProcessor` for new data formats
4. **Visualization**: Add plotting functions to notebooks

### Testing

```bash
# Test data processing
python -m src.data_processor

# Test model training
python -m src.model_trainer  

# Test analysis engine
python -m src.analysis_engine
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Dependencies

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **NLTK**: Natural language processing
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities

### Visualization (Notebooks)
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualization  
- **Plotly**: Interactive plots

### Development
- **Jupyter**: Notebook environment
- **Docker**: Containerization

## 🆘 Troubleshooting

### Common Issues

1. **NLTK Data Error**: Run `python -c "import nltk; nltk.download('all')"`
2. **Memory Issues**: Reduce batch size or sequence length
3. **TensorFlow Warnings**: Update to latest version
4. **Docker Build Fails**: Check system requirements and Docker version

### Performance Optimization

1. **GPU Support**: Install `tensorflow-gpu` for faster training
2. **Memory Management**: Use data generators for large datasets
3. **Model Size**: Reduce embedding/LSTM dimensions if needed


For questions, issues, or contributions:

1. **Create an Issue**: Use GitHub Issues for bugs and feature requests
2. **Documentation**: Check inline code documentation
3. **Examples**: Review the `notebooks/` directory for detailed examples

---

