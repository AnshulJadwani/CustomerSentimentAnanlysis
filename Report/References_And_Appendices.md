# References

## Research Papers and Academic Publications

1. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval*, 2(1-2), 1-135.

2. Liu, B. (2012). Sentiment analysis and opinion mining. *Synthesis Lectures on Human Language Technologies*, 5(1), 1-167.

3. Medhat, W., Hassan, A., & Korashy, H. (2014). Sentiment analysis algorithms and applications: A survey. *Ain Shams Engineering Journal*, 5(4), 1093-1113.

4. Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 8(4), e1253.

5. Tripathi, V., Joshi, A., & Agrawal, P. (2016). Analysis of different machine learning algorithms for sentiment analysis. *International Journal of Computer Applications*, 156(2), 1-6.

6. Sharma, A., & Dey, S. (2012). A comparative study of feature selection and machine learning techniques for sentiment analysis. *Proceedings of the 2012 ACM Research in Applied Computation Symposium*, 1-7.

7. Wang, G., Sun, J., Ma, J., Xu, K., & Gu, J. (2014). Sentiment classification: The contribution of ensemble learning. *Decision Support Systems*, 57, 77-93.

8. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*, 142-150.

9. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171-4186.

10. Agarwal, B., Mittal, N., Bansal, P., & Garg, S. (2015). Sentiment analysis using common-sense and context information. *Computational Intelligence and Neuroscience*, 2015.

11. Taboada, M., Brooke, J., Tofiloski, M., Voll, K., & Stede, M. (2011). Lexicon-based methods for sentiment analysis. *Computational Linguistics*, 37(2), 267-307.

12. Pak, A., & Paroubek, P. (2010). Twitter as a corpus for sentiment analysis and opinion mining. *Proceedings of the Seventh International Conference on Language Resources and Evaluation*, 1320-1326.

13. Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon. *Computational Intelligence*, 29(3), 436-465.

14. Kiritchenko, S., Zhu, X., & Mohammad, S. M. (2014). Sentiment analysis of short informal texts. *Journal of Artificial Intelligence Research*, 50, 723-762.

15. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect based sentiment analysis. *Proceedings of the 8th International Workshop on Semantic Evaluation*, 27-35.

## Books and Textbooks

16. Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed.). Pearson.

17. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

18. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

19. Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.

20. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

## Online Resources and Documentation

21. Scikit-learn Documentation. (2025). Machine learning in Python. Retrieved from https://scikit-learn.org/

22. NLTK Documentation. (2025). Natural Language Toolkit. Retrieved from https://www.nltk.org/

23. Streamlit Documentation. (2025). The fastest way to build data apps. Retrieved from https://streamlit.io/

24. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

25. Python Software Foundation. (2025). Python Language Reference, version 3.8. Retrieved from https://www.python.org/

---

**Note:** This reference list includes foundational works in sentiment analysis, natural language processing, and machine learning. The specific research papers referenced in your context (the 4 papers mentioned in your image) should be added here with complete citation details including authors, year, title, journal/conference, volume, and page numbers.

---

# Appendix A: Code Samples

## A.1 Text Preprocessing Function

```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Preprocess review text for sentiment analysis.
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)
```

## A.2 Model Training Script

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
    """
    Train sentiment analysis model.
    
    Args:
        X: Review texts
        y: Sentiment labels
        
    Returns:
        model: Trained SVM model
        vectorizer: Fitted TF-IDF vectorizer
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer
```

## A.3 Prediction Function

```python
def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Review text
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (sentiment_label, confidence_scores)
    """
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform to TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Map to sentiment labels
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_map[prediction]
    
    confidence = {
        'Positive': probabilities[2],
        'Neutral': probabilities[1],
        'Negative': probabilities[0]
    }
    
    return sentiment, confidence
```

---

# Appendix B: User Manual

## B.1 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection for downloading packages

### Step 1: Clone or Download the Project
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## B.2 Running the Web Application

### Start the Application
```bash
streamlit run app/app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Enter Review Text**: Type or paste a customer review in the text area
2. **Select Model**: Choose between Logistic Regression, Naive Bayes, or SVM
3. **Analyze Sentiment**: Click the "Analyze Sentiment" button
4. **View Results**: The predicted sentiment appears with color coding:
   - 🟢 Green = Positive
   - 🟡 Yellow = Neutral
   - 🔴 Red = Negative
5. **Check Confidence**: View probability scores for all sentiment classes

## B.3 Using the Command-Line Interface

### Train Models
```bash
python main.py --mode train --data data/clean_review.csv
```

### Predict Single Review
```bash
python main.py --mode predict --text "This phone is amazing!"
```

### Batch Prediction
```bash
python main.py --mode batch --input reviews.csv --output results.csv
```

## B.4 Troubleshooting

### Common Issues and Solutions

**Issue**: ModuleNotFoundError  
**Solution**: Ensure virtual environment is activated and all dependencies are installed

**Issue**: NLTK data not found  
**Solution**: Run NLTK download commands again

**Issue**: Low memory error  
**Solution**: Reduce max_features in TF-IDF vectorizer or process data in smaller batches

**Issue**: Model file not found  
**Solution**: Run training first to generate model files

---

# Appendix C: Glossary

**Accuracy**: The percentage of correct predictions out of total predictions made.

**Bigram**: A sequence of two adjacent words in a text, used to capture context.

**Classification**: The task of assigning predefined categories to text or data.

**Confusion Matrix**: A table showing correct and incorrect predictions for each class.

**F1-Score**: The harmonic mean of precision and recall, providing a balanced metric.

**Feature**: An individual measurable property used by machine learning models.

**Lemmatization**: Reducing words to their base or dictionary form (e.g., "running" → "run").

**Machine Learning**: A field of AI where computers learn patterns from data without explicit programming.

**Natural Language Processing (NLP)**: Technology that enables computers to understand human language.

**Naive Bayes**: A probabilistic classifier based on Bayes' theorem with independence assumptions.

**Precision**: The proportion of positive predictions that are actually correct.

**Recall**: The proportion of actual positives that are correctly identified.

**Sentiment Analysis**: The computational study of opinions, sentiments, and emotions in text.

**Stopwords**: Common words (like "the", "is", "and") that are filtered out during preprocessing.

**Support Vector Machine (SVM)**: A machine learning algorithm that finds optimal boundaries between classes.

**TF-IDF**: Term Frequency-Inverse Document Frequency, a numerical statistic reflecting word importance.

**Tokenization**: The process of splitting text into individual words or tokens.

**Training Data**: Labeled examples used to teach a machine learning model.

**Vectorization**: Converting text into numerical format that machines can process.

---

**End of Complete Report**
