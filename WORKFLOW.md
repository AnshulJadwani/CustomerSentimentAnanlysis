# 🔄 Project Workflow Diagram

## Complete Sentiment Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CUSTOMER SENTIMENT ANALYSIS                          │
│                         Complete Workflow                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA COLLECTION & LOADING                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📁 Input: clean_review.csv                                             │
│     ├── mobile_names    (Product information)                           │
│     ├── title           (Review title)                                  │
│     ├── body            (Review content)                                │
│     └── Sentiment       (Original labels)                               │
│                                                                          │
│  📊 Dataset: ~3,300+ reviews                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: DATA PREPROCESSING                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🧹 Text Cleaning (utils/preprocessing.py)                              │
│     ├── Lowercase conversion                                            │
│     ├── URL removal                                                     │
│     ├── HTML tag removal                                                │
│     ├── Emoji removal                                                   │
│     ├── Punctuation removal                                             │
│     ├── Number removal                                                  │
│     ├── Stopword removal (NLTK)                                         │
│     └── Lemmatization (WordNetLemmatizer)                               │
│                                                                          │
│  🔢 Label Encoding                                                      │
│     ├── Extremely Negative/Negative  → 0 (Negative)                    │
│     ├── Neutral                      → 1 (Neutral)                      │
│     └── Positive/Extremely Positive  → 2 (Positive)                    │
│                                                                          │
│  🗑️ Data Quality                                                        │
│     ├── Handle missing values                                           │
│     ├── Remove empty reviews                                            │
│     └── Combine title + body                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: EXPLORATORY DATA ANALYSIS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📊 Statistical Analysis                                                │
│     ├── Sentiment distribution                                          │
│     ├── Review length analysis                                          │
│     ├── Word count statistics                                           │
│     └── Class imbalance check                                           │
│                                                                          │
│  📈 Visualizations                                                      │
│     ├── Word clouds (per sentiment)                                     │
│     ├── Top frequent words                                              │
│     ├── Distribution plots                                              │
│     └── Box plots by sentiment                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: FEATURE EXTRACTION                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🔤 TF-IDF Vectorization                                                │
│     ├── Parameters:                                                     │
│     │   ├── max_features = 5000                                         │
│     │   ├── min_df = 2                                                  │
│     │   ├── max_df = 0.8                                                │
│     │   └── ngram_range = (1, 2)                                        │
│     │                                                                    │
│     └── Output: 5000-dimensional feature vectors                        │
│                                                                          │
│  ✂️ Train-Test Split                                                    │
│     ├── Training: 80% (stratified)                                      │
│     └── Testing: 20% (stratified)                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: MODEL TRAINING                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🤖 Model 1: Logistic Regression                                        │
│     ├── Algorithm: Linear classification                                │
│     ├── Training Time: Fast                                             │
│     ├── Expected Accuracy: 88-92%                                       │
│     └── Interpretable coefficients                                      │
│                                                                          │
│  🤖 Model 2: Naive Bayes                                                │
│     ├── Algorithm: Multinomial NB                                       │
│     ├── Training Time: Very Fast                                        │
│     ├── Expected Accuracy: 85-89%                                       │
│     └── Probabilistic predictions                                       │
│                                                                          │
│  🤖 Model 3: Support Vector Machine                                     │
│     ├── Algorithm: Linear SVM                                           │
│     ├── Training Time: Slow                                             │
│     ├── Expected Accuracy: 89-93%                                       │
│     └── Best performance                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: MODEL EVALUATION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📊 Performance Metrics                                                 │
│     ├── Accuracy Score                                                  │
│     ├── Precision (weighted)                                            │
│     ├── Recall (weighted)                                               │
│     └── F1-Score (weighted)                                             │
│                                                                          │
│  📈 Detailed Analysis                                                   │
│     ├── Classification reports                                          │
│     ├── Confusion matrices                                              │
│     ├── Per-class metrics                                               │
│     └── Feature importance                                              │
│                                                                          │
│  🏆 Model Selection                                                     │
│     └── Select best model based on accuracy                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 7: MODEL PERSISTENCE                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  💾 Save Artifacts (models/ directory)                                  │
│     ├── best_model.pkl                (Best performing model)           │
│     ├── logistic_regression_model.pkl (LR model)                        │
│     ├── naive_bayes_model.pkl         (NB model)                        │
│     ├── svm_model.pkl                 (SVM model)                       │
│     ├── tfidf_vectorizer.pkl          (Feature extractor)               │
│     └── model_metadata.json           (Performance data)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 8: DEPLOYMENT & USAGE                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🌐 Option 1: Streamlit Web App                                         │
│     ├── Launch: streamlit run app/app.py                                │
│     ├── Features:                                                       │
│     │   ├── Real-time predictions                                       │
│     │   ├── Confidence visualization                                    │
│     │   ├── Model comparison                                            │
│     │   └── Interactive interface                                       │
│     └── URL: http://localhost:8501                                      │
│                                                                          │
│  💻 Option 2: Command Line                                              │
│     ├── Training: python main.py train                                  │
│     └── Prediction: python main.py predict --text "..."                 │
│                                                                          │
│  📓 Option 3: Jupyter Notebook                                          │
│     ├── Interactive exploration                                         │
│     ├── Visualization generation                                        │
│     └── Educational purposes                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: SENTIMENT PREDICTION                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📝 Input: Customer Review Text                                         │
│                                                                          │
│  🔮 Output:                                                             │
│     ├── Sentiment: Positive/Neutral/Negative                            │
│     ├── Confidence: 0.00 - 1.00                                         │
│     └── Probabilities: [P(Neg), P(Neu), P(Pos)]                        │
│                                                                          │
│  💡 Business Value:                                                     │
│     ├── Instant customer feedback analysis                              │
│     ├── Product improvement insights                                    │
│     ├── Brand reputation monitoring                                     │
│     └── Automated decision support                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                        KEY COMPONENTS MATRIX
═══════════════════════════════════════════════════════════════════════════

Component          │ File                      │ Purpose
───────────────────┼───────────────────────────┼──────────────────────────
Data Storage       │ data/clean_review.csv     │ Raw review dataset
Preprocessing      │ utils/preprocessing.py    │ Text cleaning utilities
Analysis           │ notebooks/sentiment_*.ipynb│ Complete ML pipeline
Training CLI       │ main.py                   │ Command-line training
Web Interface      │ app/app.py                │ Streamlit application
Model Storage      │ models/*.pkl              │ Trained models
Documentation      │ README.md                 │ Project documentation
Quick Start        │ QUICKSTART.md             │ Getting started guide
Dependencies       │ requirements.txt          │ Python packages

═══════════════════════════════════════════════════════════════════════════
                         TECHNOLOGY STACK
═══════════════════════════════════════════════════════════════════════════

Layer              │ Technologies
───────────────────┼──────────────────────────────────────────────────────
Data Processing    │ pandas, numpy
NLP                │ nltk (tokenization, lemmatization, stopwords)
Feature Extraction │ scikit-learn (TfidfVectorizer)
Machine Learning   │ scikit-learn (LogisticRegression, MultinomialNB, SVC)
Visualization      │ matplotlib, seaborn, wordcloud, plotly
Web Framework      │ Streamlit
Model Persistence  │ joblib, pickle
Development        │ Jupyter, Python 3.8+

═══════════════════════════════════════════════════════════════════════════
