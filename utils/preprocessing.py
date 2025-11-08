"""
Text Preprocessing Utilities for Sentiment Analysis
This module contains functions for cleaning and preprocessing text data.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_html(text):
    """Remove HTML tags from text."""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)


def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)


def remove_extra_whitespace(text):
    """Remove extra whitespace from text."""
    return ' '.join(text.split())


def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower()


def remove_stopwords(text):
    """Remove stopwords from text."""
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def lemmatize_text(text):
    """Lemmatize text."""
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized)


def clean_text(text, remove_stopwords_flag=True, lemmatize=True):
    """
    Apply all cleaning operations to text.
    
    Parameters:
    -----------
    text : str
        Input text to clean
    remove_stopwords_flag : bool
        Whether to remove stopwords (default: True)
    lemmatize : bool
        Whether to apply lemmatization (default: True)
    
    Returns:
    --------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Apply cleaning steps in order
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    
    if remove_stopwords_flag:
        text = remove_stopwords(text)
    
    if lemmatize:
        text = lemmatize_text(text)
    
    return text


def convert_sentiment_to_numeric(sentiment):
    """
    Convert sentiment labels to numeric values.
    
    Parameters:
    -----------
    sentiment : str
        Sentiment label (Extremely Negative, Negative, Neutral, Positive, Extremely Positive)
    
    Returns:
    --------
    int
        Numeric sentiment value (0: Negative, 1: Neutral, 2: Positive)
    """
    sentiment = str(sentiment).lower()
    
    if 'negative' in sentiment:
        return 0  # Negative
    elif 'neutral' in sentiment:
        return 1  # Neutral
    elif 'positive' in sentiment:
        return 2  # Positive
    else:
        return 1  # Default to Neutral


def get_sentiment_label(numeric_sentiment):
    """
    Convert numeric sentiment to label.
    
    Parameters:
    -----------
    numeric_sentiment : int
        Numeric sentiment (0, 1, or 2)
    
    Returns:
    --------
    str
        Sentiment label
    """
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels.get(numeric_sentiment, 'Neutral')
