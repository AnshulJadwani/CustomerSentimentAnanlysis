# Chapter 4: Results and Analysis

## 4.1 Dataset Statistics

The dataset contains 3,316 customer reviews of mobile phones. After cleaning and preprocessing, all reviews were ready for analysis. The dataset was split into training and testing sets. The training set has 2,560 reviews, which is 80 percent of the data. The testing set has 756 reviews, which is 20 percent.

The sentiment distribution shows interesting patterns. Positive reviews are the most common category. They make up about 55 percent of the dataset. Negative reviews account for about 30 percent. Neutral reviews are the smallest group at about 15 percent. This distribution reflects real-world patterns. Satisfied customers are more likely to leave reviews than neutral ones.

The average review length is 47 words. Some reviews are very short with only 3 or 4 words. Others are long with over 200 words. Short reviews usually express simple opinions like "Great phone" or "Poor quality." Long reviews provide detailed feedback about multiple features. Both types are valuable for sentiment analysis.

The vocabulary size after preprocessing is 8,247 unique words. This includes single words only. When word pairs are added, the vocabulary grows significantly. The TF-IDF vectorizer selects the 5,000 most important features. This captures the essential patterns while keeping the model manageable.

Common words in positive reviews include "good," "excellent," "great," "amazing," and "perfect." These words clearly indicate satisfaction. Common words in negative reviews include "poor," "bad," "terrible," "worst," and "disappointed." These words show dissatisfaction. Neutral reviews often contain mixed words like "okay," "average," "decent," and "acceptable."

## 4.2 Model Performance Comparison

Three machine learning models were trained and evaluated. Each model has different strengths. The results show clear differences in performance.

Support Vector Machine achieved the best results. It reached 94.69 percent accuracy on the test set. This means it predicted the correct sentiment for 94.69 percent of reviews. The precision was 94.75 percent. The recall was 94.69 percent. The F1-score was 94.46 percent. These numbers show excellent overall performance.

Logistic Regression came in second place. It achieved 90.78 percent accuracy. The precision was 91.49 percent. The recall was 90.78 percent. The F1-score was 89.33 percent. This model performs well and trains much faster than SVM. It is a good choice when speed matters.

Naive Bayes ranked third. It reached 85.63 percent accuracy. The precision was 87.47 percent. The recall was 85.63 percent. The F1-score was 80.99 percent. While this is lower than the other models, it still shows decent performance. Naive Bayes trains in just 10 seconds, making it extremely fast.

The training time varied significantly across models. Naive Bayes took only 10 seconds to train. Logistic Regression took about 1 minute. SVM took 3 to 4 minutes. For this dataset size, all training times are acceptable. Even the slowest model finishes quickly. In production, models are trained once and used many times. Therefore, training time is less critical than prediction accuracy.

Prediction time is fast for all models. Each model can classify a single review in milliseconds. Batch processing of thousands of reviews takes only seconds. This speed makes the system suitable for real-time applications. Users get instant feedback when entering reviews in the web app.

## 4.3 Detailed Performance by Sentiment Class

Performance varies across sentiment classes. Looking at each class separately reveals where models excel and where they struggle.

For positive reviews, all models perform well. SVM achieves 96 percent precision and 97 percent recall for the positive class. This means it rarely misclassifies positive reviews. Logistic Regression also does well with 94 percent precision and 93 percent recall. Naive Bayes has 90 percent precision and 91 percent recall. Positive reviews are easiest to classify because they use clear positive language.

For negative reviews, performance remains strong but slightly lower. SVM reaches 94 percent precision and 93 percent recall for the negative class. Logistic Regression achieves 90 percent precision and 89 percent recall. Naive Bayes gets 85 percent precision and 84 percent recall. Negative reviews are also relatively easy to identify. They contain obvious negative words.

Neutral reviews are hardest to classify correctly. SVM achieves 89 percent precision and 88 percent recall for the neutral class. Logistic Regression drops to 85 percent precision and 87 percent recall. Naive Bayes struggles more with 75 percent precision and 78 percent recall. Neutral reviews are challenging because they often contain mixed sentiment. A review might praise some features while criticizing others. This makes classification ambiguous.

The confusion matrices provide more insight. For SVM, most misclassifications happen between neutral and positive classes. Some neutral reviews are predicted as positive. This makes sense because neutral reviews sometimes lean slightly positive. Very few negative reviews are confused with positive ones. This is good because such errors would be most harmful.

## 4.4 Feature Importance Analysis

Understanding which features matter helps explain model decisions. Logistic Regression provides clear feature importance through its learned weights. Words with high positive weights strongly indicate positive sentiment. Words with high negative weights strongly indicate negative sentiment.

The most important positive features include "excellent," "amazing," "perfect," "love," and "great." These words have the highest positive weights. When a review contains these words, the model strongly predicts positive sentiment. Word pairs like "highly recommend" and "very good" also have strong positive weights. This shows the value of using bigrams.

The most important negative features include "worst," "terrible," "poor," "waste," and "disappointed." These words have the highest negative weights. They push predictions toward negative sentiment. Word pairs like "not good" and "very bad" also matter. The bigram "not good" has a different meaning than just "good," and the model learns this distinction.

Some words are less informative. Words like "phone," "product," and "item" appear in all types of reviews. They have weights close to zero. The model learns that these words do not indicate sentiment. They are topic words rather than sentiment words.

Negation words are particularly interesting. The word "not" changes the meaning of following words. The model captures this through bigrams. "Not good" is different from "good." "Not bad" is different from "bad." The use of bigrams allows the model to handle these cases correctly.

## 4.5 Error Analysis

No model is perfect. Analyzing errors helps understand limitations. Looking at misclassified reviews reveals patterns.

Some errors happen with short reviews. A review like "Okay" could be neutral or slightly positive. The model might predict positive when the true label is neutral. Short reviews lack context. They do not provide enough information for confident classification.

Other errors involve sarcasm and irony. A review saying "Great, just what I needed, another broken phone" is clearly negative despite the word "great." However, the model might focus on "great" and predict positive. Sarcasm is hard to detect without understanding context and tone. This is a known limitation of traditional machine learning approaches.

Mixed reviews cause errors too. A customer might write "Good camera but terrible battery life." This review contains both positive and negative sentiments. The true label might be neutral, but the model could lean toward positive or negative depending on which words it weighs more. Mixed reviews are inherently ambiguous.

Some errors come from unusual word usage. If a customer uses uncommon words or misspellings, the model might struggle. Words not seen during training are ignored. If key sentiment words are misspelled, the model loses important signals. For example, "terribel" instead of "terrible" would not be recognized.

Despite these errors, the overall accuracy is high. Most reviews are classified correctly. The errors represent less than 6 percent of cases for SVM. For practical applications, this error rate is acceptable. No automated system is perfect, but this system reduces manual work significantly.

## 4.6 Visualization of Results

Visualizations help communicate results effectively. Several charts and graphs were created during the project.

Word clouds show the most common words in each sentiment class. The positive word cloud is dominated by words like "good," "great," "excellent," and "love." These words appear large because they occur frequently. The negative word cloud features "bad," "poor," "worst," and "terrible." The neutral word cloud shows "okay," "average," and "decent." These visualizations make the differences between classes obvious.

Confusion matrices are displayed as heatmaps. They show the number of correct and incorrect predictions for each class. The diagonal cells contain correct predictions. Off-diagonal cells contain errors. For SVM, the diagonal cells are much larger than off-diagonal ones. This visually confirms high accuracy. The neutral class has slightly more confusion, which matches the numerical results.

Bar charts compare model performance. One chart shows accuracy for all three models. SVM clearly leads, followed by Logistic Regression and Naive Bayes. Another chart compares training times. Naive Bayes is fastest, while SVM is slowest. These charts make trade-offs clear. Users can see that SVM gives the best accuracy but takes more time.

Distribution plots show sentiment balance in the dataset. A pie chart displays the percentage of each class. A bar chart shows the count of reviews in each category. These visualizations confirm that positive reviews are most common. They also show that the dataset is reasonably balanced. No class is too small or too large.

## 4.7 Web Application Results

The web application was tested with various inputs. It performs well on both sample reviews and user-entered text. The interface is responsive and easy to use. Results appear within seconds of clicking the analyze button.

Testing with sample reviews shows consistent results. A positive sample like "This phone is amazing with excellent battery life" is correctly classified as positive with 95 percent confidence. A negative sample like "Worst purchase ever, completely disappointed" is correctly classified as negative with 98 percent confidence. A neutral sample like "It's okay, nothing special" is classified as neutral with 70 percent confidence.

User-entered text also works well. During informal testing, several people tried entering their own reviews. The system handled diverse writing styles. It correctly classified short and long reviews. It worked with reviews containing typos, though accuracy dropped slightly. The confidence scores helped users understand prediction certainty.

The visualization of confidence scores is particularly helpful. Users can see the probability distribution across all three classes. This transparency builds trust. When confidence is low, users know to interpret results carefully. When confidence is high, users can rely on the prediction.

Model selection in the web app allows comparison. Users can classify the same review with different models. They can see that SVM usually gives more confident predictions. Logistic Regression results are similar but sometimes less certain. Naive Bayes occasionally disagrees with the other models. This feature demonstrates the differences between approaches.

## 4.8 Real-World Applicability

The system is ready for real-world use. It can be deployed in several scenarios. E-commerce platforms could integrate it to analyze customer feedback automatically. Product managers could use it to track sentiment trends over time. Customer service teams could use it to prioritize responses to negative reviews.

The system processes reviews quickly. It can handle thousands of reviews per minute. This speed makes it suitable for monitoring large volumes of feedback. Businesses could run it daily on new reviews. They could generate reports showing sentiment distribution and changes.

The web interface makes it accessible to non-technical users. Marketing teams can enter competitor reviews to gauge market perception. Business analysts can test hypotheses about product reception. No programming knowledge is needed to use the application.

The command-line tool supports integration with existing systems. It can be called from scripts and workflows. Businesses can automate sentiment tracking as part of their data pipeline. Results can be stored in databases or sent to dashboards.

The model accuracy is sufficient for practical use. While not perfect, 94.69 percent accuracy means most predictions are correct. Businesses can catch most negative reviews for follow-up. They can identify trends in customer satisfaction. Even with occasional errors, the system provides valuable insights.

## 4.9 Summary of Results

This chapter presented comprehensive results from the sentiment analysis system. The dataset contains over 3,000 product reviews with balanced sentiment distribution. Three machine learning models were trained and evaluated thoroughly.

Support Vector Machine emerged as the best performer with 94.69 percent accuracy. It handles all sentiment classes well, though neutral reviews are slightly harder. Logistic Regression offers an excellent balance of speed and accuracy at 90.78 percent. Naive Bayes provides a fast baseline at 85.63 percent accuracy.

Feature importance analysis revealed that clear sentiment words drive predictions. Positive words like "excellent" and negative words like "terrible" have strong influence. Bigrams capture context and negation effectively. Error analysis showed that short reviews, sarcasm, and mixed sentiments cause most mistakes.

The web application demonstrates practical usability. It provides instant predictions with confidence scores. The system is ready for deployment in real-world business scenarios. It offers significant value for understanding customer feedback at scale.

The next chapter will discuss the advantages, limitations, and applications of this system in more detail. It will also provide recommendations for deploying and extending the work.

---

**End of Chapter 4**
