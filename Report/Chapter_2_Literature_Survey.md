# Chapter 2: Literature Survey

## 2.1 Introduction to Sentiment Analysis

Sentiment analysis is a field of study that deals with finding opinions in text. It started in the early 2000s when online reviews became common. Researchers wanted to know if a piece of text was positive or negative. Early work focused on movie reviews and product ratings. Over time, the field grew to cover social media, news, and customer feedback.

The main goal of sentiment analysis is to understand what people think about products, services, or topics. Companies use it to track customer satisfaction. Governments use it to monitor public opinion. Researchers use it to study social trends. The field combines computer science and linguistics. It uses techniques from machine learning and natural language processing.

Early methods relied on lists of positive and negative words. These lexicon-based approaches counted good and bad words in a text. If positive words were more common, the text was labeled positive. This method was simple but had problems. It missed sarcasm and context. It did not handle complex sentences well. Later, machine learning methods improved accuracy. They learned patterns from labeled examples instead of relying on fixed word lists.

Modern sentiment analysis uses deep learning and transformer models. These methods understand context better. They can handle long texts and complex language. However, simpler machine learning models still work well for many tasks. They are faster and easier to train. They need less data and computing power. This project uses traditional machine learning because it fits the task well.

## 2.2 Evolution of Text Classification Methods

Text classification started with rule-based systems. Experts wrote rules to assign labels to documents. For example, if a document contained certain keywords, it belonged to a specific category. These systems were accurate but took a lot of time to build. They did not adapt to new data easily. Any change required manual updates.

Machine learning changed this approach in the 1990s. Researchers began using algorithms that learned from examples. They collected labeled documents and trained models to find patterns. Naive Bayes was one of the first successful algorithms. It assumed that words appeared independently in documents. This assumption was not always true, but the method worked well in practice. Naive Bayes became popular for email spam filtering and document categorization.

Support Vector Machines appeared in the late 1990s. They found the best boundary between classes in high-dimensional space. SVMs worked well with text because text data has many features. Each word can be a feature. SVMs handled thousands of features without overfitting. They became a standard choice for text classification tasks.

Logistic Regression also gained popularity. It modeled the probability of each class. It was fast and easy to interpret. Researchers could see which words mattered most for each class. This helped them understand what the model learned. Logistic Regression worked especially well when combined with good features.

Feature engineering was crucial for these methods. Raw text could not be used directly. Researchers had to convert text into numbers. The bag-of-words model counted word frequencies. TF-IDF weighted words by their importance. N-grams captured word sequences. These techniques improved model performance significantly. This project uses TF-IDF with n-grams because it balances simplicity and effectiveness.

## 2.3 Natural Language Processing for Text Preprocessing

Text preprocessing is a critical step in any text analysis project. Raw text contains noise that confuses machine learning models. This noise includes punctuation, special characters, and inconsistent formatting. Preprocessing cleans the text and makes it ready for analysis.

The first step is usually converting text to lowercase. This ensures that "Good" and "good" are treated as the same word. Next, punctuation and special characters are removed. They rarely carry useful information for sentiment analysis. Numbers can be kept or removed depending on the task. In this project, numbers were removed because they did not help predict sentiment.

Stopwords are common words that appear frequently but carry little meaning. Examples include "the," "is," "and," and "in." Removing stopwords reduces the feature space. It helps models focus on meaningful words. Standard stopword lists exist for most languages. However, some tasks require custom lists. In sentiment analysis, words like "not" should be kept because they change meaning.

Tokenization splits text into words or tokens. This step seems simple but has challenges. Contractions like "don't" need to be split correctly. Special cases like URLs and email addresses need handling. Good tokenization improves model performance.

Stemming and lemmatization reduce words to their base form. Stemming cuts off word endings using simple rules. It is fast but can produce non-words. For example, "running" becomes "run," but "studies" might become "studi." Lemmatization uses a dictionary to find the proper base form. It is slower but more accurate. This project uses lemmatization because it produces better results.

After preprocessing, the text is much cleaner. It contains only meaningful words in a standard format. This makes it easier for machine learning models to find patterns. Good preprocessing can improve accuracy by several percentage points.

## 2.4 Machine Learning Approaches for Sentiment Analysis

Machine learning methods have become the standard for sentiment analysis. They learn patterns from labeled examples without needing hand-written rules. Three main types of algorithms are commonly used: probabilistic methods, linear classifiers, and support vector machines.

Naive Bayes is a probabilistic classifier. It calculates the probability of each sentiment class given the words in a review. It assumes that words are independent of each other. This assumption is not realistic, but the method still works well. Naive Bayes is very fast to train. It needs less data than other methods. It serves as a good baseline for comparison. Many researchers start with Naive Bayes before trying more complex models.

Logistic Regression is a linear classifier. It learns a weight for each feature. Features with positive weights support positive sentiment. Features with negative weights support negative sentiment. The model combines these weights to predict the class. Logistic Regression is simple and fast. It provides probabilities for predictions. These probabilities help users understand model confidence. The model also offers good interpretability. Users can examine weights to see which words matter most.

Support Vector Machines find the best decision boundary between classes. They maximize the margin between different sentiment groups. SVMs handle high-dimensional data very well. Text data often has thousands of features because each unique word becomes a feature. SVMs perform well even with this many features. They use kernel functions to handle non-linear patterns. The linear kernel works well for text classification. It is faster than other kernels and gives good results.

All three methods need feature extraction. TF-IDF is the most common approach. It converts text into numerical vectors. Each word gets a score based on its frequency and importance. Rare words that appear in few documents get higher scores. Common words that appear everywhere get lower scores. This helps models identify distinctive words for each sentiment class.

This project compares all three methods. Each has strengths and weaknesses. Naive Bayes is fastest but least accurate. Logistic Regression balances speed and accuracy. SVM gives the best accuracy but takes longer to train. The choice depends on the specific requirements of the task.

## 2.5 E-Commerce Review Analysis

E-commerce platforms generate massive amounts of review data. Customers write reviews after purchasing products. These reviews describe their experience with the product. They mention quality, features, delivery, and customer service. This information is valuable for businesses and future buyers.

Product reviews usually include a rating and text. The rating is a number from one to five stars. The text explains the rating. Sometimes the text and rating do not match. A customer might give four stars but write negative comments. This makes sentiment analysis challenging. Models must focus on the text content, not just the rating.

Reviews vary in length and style. Some are short phrases like "Great product" or "Waste of money." Others are long paragraphs describing detailed experiences. Some customers write formal text. Others use casual language with slang and abbreviations. Models must handle this variety.

Sentiment in reviews is not always clear. Some reviews are mixed. They praise certain features but criticize others. A customer might say "Good battery life but poor camera quality." This is neither fully positive nor fully negative. Three-class classification helps here. It adds a neutral category for mixed reviews.

Reviews also contain specific domain vocabulary. Phone reviews mention terms like "battery," "camera," "display," and "performance." These words appear frequently and carry strong sentiment signals. Models can learn that "excellent display" suggests positive sentiment while "poor battery" suggests negative sentiment.

Fake reviews are a growing problem. Some sellers post fake positive reviews to boost ratings. Competitors sometimes post fake negative reviews. Detecting fake reviews requires different techniques. This project assumes all reviews are genuine. It focuses on sentiment classification, not fraud detection.

Many studies have analyzed e-commerce reviews. Researchers found that reviews influence buying decisions. Positive reviews increase sales. Negative reviews decrease them. Companies that respond to reviews show better customer relations. Quick response to negative reviews can prevent customer loss.

This project uses mobile phone reviews. These reviews show clear sentiment patterns. Customers express strong opinions about phone features. The dataset includes positive, negative, and neutral reviews. This makes it suitable for training classification models.

## 2.6 Related Work and Research Gaps

Many researchers have worked on sentiment analysis of product reviews. Early studies used simple methods like counting positive and negative words. Later work applied machine learning algorithms. Recent studies use deep learning models. Each approach has contributed to the field.

Some researchers focused on feature selection. They identified the most important words for sentiment prediction. They found that adjectives carry strong sentiment signals. Words like "excellent," "terrible," "good," and "bad" are highly predictive. Combining single words with word pairs improved accuracy. This finding supports the use of n-grams in feature extraction.

Other studies compared different machine learning algorithms. Most found that SVM performs well on text data. Logistic Regression offers a good balance of speed and accuracy. Naive Bayes works as a fast baseline. Ensemble methods that combine multiple models sometimes improve results. However, they also increase complexity.

Deep learning approaches have gained attention recently. Recurrent neural networks and transformers can capture complex patterns. They understand context better than traditional methods. However, they need large amounts of training data. They also require significant computing power. For smaller datasets, traditional machine learning often works better.

Some research focused on specific domains. Studies analyzed hotel reviews, restaurant reviews, and movie reviews. Each domain has unique characteristics. Product reviews differ from service reviews. Technical products like phones have different vocabulary than food products. Domain-specific models often outperform general models.

Despite this extensive research, gaps remain. Most studies use English text only. Work on other languages is limited. Most datasets focus on single products or categories. Cross-category analysis is rare. Real-time sentiment analysis receives less attention. Most work assumes batch processing.

Another gap is the handling of short and informal text. Social media posts are shorter than product reviews. They use more slang and abbreviations. They include emojis and hashtags. Methods that work well on formal reviews may fail on social media text.

This project addresses some of these gaps. It focuses on product reviews but uses a practical approach. It compares multiple models to find the best option. It builds a complete system from data processing to web deployment. It provides both batch processing and real-time prediction. The methods used here can be adapted to other domains with minimal changes.

## 2.7 Summary of Literature Survey

The literature survey shows that sentiment analysis has evolved significantly. Early rule-based systems gave way to machine learning methods. Traditional algorithms like Naive Bayes, Logistic Regression, and SVM remain effective for many tasks. They work well when combined with proper text preprocessing and feature extraction.

Text preprocessing is crucial for good results. It includes cleaning, tokenization, stopword removal, and lemmatization. These steps prepare text for analysis. Feature extraction converts text into numbers. TF-IDF with n-grams has proven effective across many studies.

E-commerce review analysis is an important application of sentiment analysis. Reviews contain valuable information for businesses and customers. Automated analysis saves time and provides quick insights. Three-class classification handles the variety of opinions found in reviews.

Research gaps exist in cross-domain analysis, multilingual support, and real-time processing. This project contributes by building a complete, practical system. It demonstrates how traditional machine learning can solve real business problems effectively. The system achieves high accuracy while remaining fast and easy to deploy.

The next chapter will describe the system requirements and design. It will explain how the literature review findings influenced the technical choices made in this project.

---

**End of Chapter 2**
