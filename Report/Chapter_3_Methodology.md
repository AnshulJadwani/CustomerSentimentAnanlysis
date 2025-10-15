# Chapter 3: Methodology and Implementation

## 3.1 System Overview

This project builds an end-to-end sentiment analysis system for e-commerce product reviews. The system takes raw text reviews as input and produces sentiment predictions as output. It classifies each review into one of three categories: positive, neutral, or negative.

The system follows a standard machine learning pipeline. First, it loads and explores the data. Next, it cleans and prepares the text. Then, it extracts features from the cleaned text. After that, it trains multiple models and compares their performance. Finally, it saves the best model and deploys it through a web interface.

The entire system is built using Python. Python offers excellent libraries for data processing, machine learning, and web development. The code is organized into modules for easy maintenance. Each module handles a specific task. This separation makes the code cleaner and easier to understand.

The system is designed for both technical and non-technical users. Data scientists can work with the Jupyter notebook to explore and experiment. Business users can use the web application to get instant predictions. The command-line tool supports batch processing for large datasets.

## 3.2 Dataset Description

The dataset contains customer reviews of mobile phones. It was collected from e-commerce platforms. Each row in the dataset represents one review. The dataset has two main columns: the review text and the sentiment label.

The review text is what customers wrote about the product. Reviews vary in length from a few words to several paragraphs. Some reviews focus on specific features like camera or battery. Others give overall impressions. The language is mostly informal. Customers use everyday words to express their thoughts.

The sentiment label shows whether the review is positive, neutral, or negative. Originally, reviews had star ratings from one to five. These ratings were converted to sentiment labels. Ratings of four and five stars became positive. Ratings of one and two stars became negative. Three-star ratings became neutral. This conversion makes sense because customers usually give high ratings when happy and low ratings when unhappy.

The dataset contains 3,316 reviews in total. This is a moderate-sized dataset. It is large enough to train good models but small enough to process quickly. The distribution across sentiment classes is reasonably balanced. This prevents models from being biased toward one class.

The data was cleaned before use. Duplicate reviews were removed. Reviews with missing text were deleted. Very short reviews with less than three words were filtered out. These reviews do not contain enough information for analysis. After cleaning, the dataset was ready for the next steps.

## 3.3 Text Preprocessing Pipeline

Text preprocessing transforms raw reviews into a clean format suitable for machine learning. This step is crucial for good model performance. The preprocessing pipeline includes several stages.

The first stage converts all text to lowercase. This ensures that "Good" and "good" are treated identically. It reduces the vocabulary size and helps the model generalize better. Next, URLs and HTML tags are removed. Some reviews contain links to websites or formatted text. These elements do not contribute to sentiment and should be deleted.

Special characters and punctuation are removed in the next stage. Characters like exclamation marks and question marks are deleted. However, spaces between words are preserved. Numbers are also removed because they rarely indicate sentiment. For example, "3GB RAM" becomes "GB RAM," which is sufficient for sentiment analysis.

The text is then split into individual words. This process is called tokenization. Each word becomes a separate token. Tokenization handles contractions correctly. For example, "don't" is split into "do" and "not." This ensures that negations are captured properly.

Stopwords are removed after tokenization. Stopwords are common words like "the," "is," "and," and "of." They appear frequently but carry little meaning. Removing them reduces noise and speeds up training. However, some stopwords like "not" are kept because they affect sentiment.

The final stage is lemmatization. This reduces words to their base form. For example, "running" becomes "run," and "better" becomes "good." Lemmatization uses language rules to find the correct base form. It is more accurate than stemming, which simply cuts off word endings.

After preprocessing, each review is a clean list of meaningful words. These words are ready for feature extraction. The preprocessing pipeline is implemented as a reusable function. It can be applied to both training data and new reviews.

## 3.4 Feature Extraction Using TF-IDF

Machine learning models cannot work with text directly. Text must be converted to numerical features. This project uses TF-IDF for feature extraction. TF-IDF stands for Term Frequency-Inverse Document Frequency.

TF-IDF measures how important a word is to a document. It considers two factors. The first factor is term frequency. This is how often a word appears in a document. Words that appear more often are more important. The second factor is inverse document frequency. This measures how rare a word is across all documents. Rare words that appear in few documents are more distinctive.

The TF-IDF score is calculated by multiplying these two factors. A word gets a high score if it appears often in a document but rarely in other documents. This helps identify words that characterize each document. Common words that appear everywhere get low scores. Rare but distinctive words get high scores.

This project uses TF-IDF with n-grams. N-grams are sequences of words. Unigrams are single words like "good" or "bad." Bigrams are word pairs like "very good" or "not bad." Using both unigrams and bigrams helps capture context. The phrase "not good" has different meaning than just "good."

The TF-IDF vectorizer is configured with specific parameters. The maximum number of features is set to 5,000. This means the model uses the 5,000 most important words and word pairs. This limit prevents overfitting and speeds up training. The minimum document frequency is set to 2. This means a word must appear in at least two reviews to be included. This filters out typos and very rare words.

The vectorizer is fitted on the training data. It learns which words appear in the corpus and calculates their IDF values. Once fitted, it can transform any text into a TF-IDF vector. Each review becomes a vector of 5,000 numbers. These vectors are fed to machine learning models for training.

## 3.5 Model Selection and Training

Three machine learning models are implemented and compared: Logistic Regression, Naive Bayes, and Support Vector Machine. Each model has different characteristics and performance.

Logistic Regression is the first model. It is a linear classifier that predicts probabilities. For each feature, it learns a weight. Positive weights support positive sentiment. Negative weights support negative sentiment. The model combines all weights to make a prediction. Logistic Regression is fast and interpretable. It provides probability scores that indicate confidence. Training takes about one minute on the dataset.

Naive Bayes is the second model. It calculates the probability of each sentiment class given the words in a review. It assumes that words are independent. This assumption simplifies calculations and makes training very fast. Naive Bayes trains in about ten seconds. Despite its simplicity, it often performs well on text data. It serves as a good baseline.

Support Vector Machine is the third model. It finds the best boundary between sentiment classes. The boundary maximizes the margin between different classes. SVM handles high-dimensional data well. With 5,000 features, the feature space is very large. SVM can work in this space without overfitting. A linear kernel is used because it works well for text classification. Training takes three to four minutes.

All models are trained on the same training set. The training set contains 80 percent of the data. The remaining 20 percent is reserved for testing. The split is stratified to maintain class balance. This ensures that each set has similar proportions of positive, neutral, and negative reviews.

During training, each model learns patterns that distinguish sentiments. Logistic Regression learns feature weights. Naive Bayes learns word probabilities. SVM learns support vectors that define the decision boundary. After training, each model is evaluated on the test set.

## 3.6 Model Evaluation

Model evaluation measures how well each model performs. Several metrics are used to get a complete picture. The first metric is accuracy. This is the percentage of correct predictions. It shows overall performance but can be misleading if classes are imbalanced.

Precision is the second metric. It measures how many predicted positives are actually positive. High precision means few false positives. This is important when false positives are costly. For example, classifying a negative review as positive could mislead customers.

Recall is the third metric. It measures how many actual positives are correctly predicted. High recall means few false negatives. This is important when missing positives is costly. For example, failing to detect negative reviews could hide customer problems.

F1-score combines precision and recall into a single number. It is the harmonic mean of the two metrics. F1-score is useful when you want to balance precision and recall. A high F1-score means both metrics are good.

Confusion matrices provide detailed insight. They show how many reviews of each class were predicted correctly or incorrectly. For example, a confusion matrix reveals if the model confuses neutral reviews with positive ones. This helps identify specific weaknesses.

All three models are evaluated using these metrics. The results are compared in a table. The Support Vector Machine achieves the highest accuracy at 94.69 percent. It also has the best precision, recall, and F1-score. Logistic Regression comes second with 90.78 percent accuracy. Naive Bayes is third with 85.63 percent accuracy.

The high accuracy of SVM makes it the best choice for this task. However, Logistic Regression offers a good balance of speed and performance. If training time is critical, Logistic Regression is a solid alternative. Naive Bayes is fastest but least accurate. It works well as a baseline but not for production use.

## 3.7 Web Application Development

The web application provides an easy way for users to interact with the model. It is built using Streamlit, a Python library for creating web apps. Streamlit turns Python scripts into interactive web interfaces with minimal code.

The application has a clean and simple layout. The main page displays the project title and description. Users see a text input box where they can enter a review. They can type their own review or choose from sample reviews. Sample reviews demonstrate the system with pre-written examples.

After entering text, users click a button to analyze sentiment. The application preprocesses the text using the same pipeline as training. It then transforms the text into a TF-IDF vector. The trained model makes a prediction and returns the sentiment label.

The result is displayed prominently. A colored box shows the predicted sentiment. Green indicates positive sentiment. Red indicates negative sentiment. Yellow indicates neutral sentiment. This visual feedback makes results easy to understand at a glance.

The application also shows confidence scores. These are probabilities for each sentiment class. For example, a review might be 85 percent positive, 10 percent neutral, and 5 percent negative. These scores help users understand how certain the model is. Low confidence suggests the review is ambiguous.

A bar chart visualizes the confidence scores. The chart makes it easy to compare probabilities across classes. Users can see which sentiment is dominant and by how much. This transparency builds trust in the predictions.

The application includes model selection. Users can choose which model to use for prediction. They can compare results from different models on the same review. This feature is useful for understanding model differences.

The web app is responsive and works on different screen sizes. It runs locally on the user's machine. Users start it with a simple command. The app opens in a web browser. No deployment to external servers is needed. This makes it easy to use for personal projects and demos.

## 3.8 Command-Line Interface

The command-line interface provides an alternative way to use the system. It is designed for users who prefer working in a terminal. It also supports automation and batch processing.

The interface is implemented in the main.py file. Users run commands by typing them in a terminal. The most common command is training. It loads the data, preprocesses text, trains models, and saves results. Users can specify which models to train and where to save them.

Another command is prediction. Users provide a text file with reviews. The system loads the trained model and predicts sentiment for each review. Results are written to an output file. This batch processing is faster than using the web app for many reviews.

The interface provides clear messages during execution. It shows progress for each step. For example, it prints "Loading data..." when loading the dataset. It prints "Training Logistic Regression..." when training that model. These messages help users track what is happening.

Error handling is included. If a file is missing, the system displays a helpful error message. If a model file is not found, it suggests training first. This makes the interface user-friendly even when things go wrong.

The command-line interface is powerful for advanced users. It allows scripting and integration with other tools. Users can schedule daily sentiment analysis on new reviews. They can combine it with other data processing pipelines. This flexibility makes it suitable for production environments.

## 3.9 Implementation Summary

This chapter described the complete methodology and implementation. The system follows a clear pipeline from data to deployment. Each step is carefully designed and implemented.

The dataset provides realistic product reviews with sentiment labels. Text preprocessing cleans the reviews and prepares them for analysis. TF-IDF extracts numerical features that capture word importance. Three machine learning models learn to classify sentiment. Evaluation metrics compare performance and select the best model.

The web application makes the system accessible to all users. The command-line interface supports automation and batch processing. Together, these components form a complete sentiment analysis solution.

The implementation uses best practices from software engineering. Code is modular and reusable. Functions have clear purposes. Variables have descriptive names. Comments explain complex logic. This makes the code easy to understand and maintain.

The next chapter will present the results in detail. It will show model performance metrics, confusion matrices, and visualization. It will analyze what the models learned and where they succeed or struggle.

---

**End of Chapter 3**
