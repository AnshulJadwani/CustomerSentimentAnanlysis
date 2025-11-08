# Customer Sentiment Analysis on E-Commerce Product Reviews

### Project Report

**Submitted in partial fulfillment of**  
**Bachelor of Technology**  
**in**  
**Artificial Intelligence & Data Science**

---

**By:**

- Anshul Jadwani (S002) - SAP ID: 70562200050
- Harshil Patni (S007) - SAP ID: 70562200083  
- Dhruv Hirani (S021) - SAP ID: 70562300107

---

**Under the supervision of:**  
**Dr. Ankur Ratmele**  
*(Assistant Professor, STME, MPSTME)*

---

**SVKM's NMIMS University**  
*(Deemed-to-be University)*

**MUKESH PATEL SCHOOL OF TECHNOLOGY MANAGEMENT & ENGINEERING**  
**(MPSTME)**

Vile Parle (W), Mumbai-56  
2025-2026

---

## CERTIFICATE

This is to certify that the project entitled **"Customer Sentiment Analysis on E-Commerce Product Reviews"** has been done by Mr. Anshul Jadwani, Mr. Harshil Patni, and Mr. Dhruv Hirani under my guidance and supervision. This work has been submitted in partial fulfillment of the degree of Bachelor of Technology in Artificial Intelligence & Data Science of MPSTME, SVKM's NMIMS (Deemed-to-be University), Mumbai, India.

**Dr. Ankur Ratmele**  
Assistant Professor

**Date:** October 15, 2025  
**Place:** Indore

---

**Examiner**

---

**Head of Department (HoD)**

---

## ACKNOWLEDGEMENT

We would like to express our sincere gratitude to all those who contributed to the successful completion of this project. First and foremost, we thank our project supervisor, Dr. Ankur Ratmele, for his constant guidance, valuable suggestions, and encouragement throughout the project duration. His expertise in machine learning and natural language processing helped us navigate technical challenges effectively.

We are grateful to the faculty members of the Department of Artificial Intelligence & Data Science at MPSTME for providing us with the necessary knowledge and skills. The courses in machine learning, data mining, and programming laid a strong foundation for this work.

We acknowledge SVKM's NMIMS University for providing excellent infrastructure and laboratory facilities. Access to computing resources and software tools was essential for implementing and testing our system.

We also thank our families and friends for their support and understanding during the project period. Their encouragement motivated us to give our best effort.

Finally, we express our appreciation to all the researchers whose work we reviewed during our literature survey. Their contributions to the field of sentiment analysis provided valuable insights that shaped our approach.

**Team Members:**

| Name | Roll No. | SAP ID |
|------|----------|---------|
| Anshul Jadwani | S002 | 70562200050 |
| Harshil Patni | S007 | 70562200083 |
| Dhruv Hirani | S021 | 70562300107 |

---

## ABSTRACT

Businesses receive large volumes of customer feedback through online reviews and social media. Reading all this feedback manually is difficult and time-consuming. This makes it hard for companies to understand customer opinions in real time. This project provides an automated solution to this challenge.

We built a web-based application that uses machine learning to analyze customer sentiment. The application quickly determines if a piece of text expresses positive, negative, or neutral sentiment. The core of the system is built on three machine learning models: Logistic Regression, Naive Bayes, and Support Vector Machine. We compared these models and found that Support Vector Machine performs best with 94.69 percent accuracy.

Natural Language Processing techniques are used to clean and prepare text data for training. The application was developed using Python, Flask, and Streamlit. The tool features a simple interface where users can enter review text and receive instant sentiment predictions displayed on a visual dashboard.

This project offers a practical tool for businesses to monitor customer opinions efficiently. It helps companies make better decisions, respond to customer concerns quickly, and improve their products and services based on actual customer feedback. The system processes thousands of reviews in minutes, making large-scale sentiment analysis feasible and cost-effective.

---

## TABLE OF CONTENTS

| Chapter | Topic | Page |
|---------|-------|------|
| | **List of Figures** | i |
| | **List of Tables** | ii |
| | **Abbreviations** | iii |
| | | |
| **1** | **Introduction** | |
| 1.1 | Background of the Project Topic | |
| 1.2 | Motivation and Scope of the Report | |
| 1.3 | Problem Statement | |
| 1.4 | Salient Contributions | |
| 1.5 | Organization of Report | |
| | | |
| **2** | **Literature Survey** | |
| 2.1 | Introduction to Sentiment Analysis | |
| 2.2 | Evolution of Text Classification Methods | |
| 2.3 | Natural Language Processing for Text Preprocessing | |
| 2.4 | Machine Learning Approaches for Sentiment Analysis | |
| 2.5 | E-Commerce Review Analysis | |
| 2.6 | Related Work and Research Gaps | |
| 2.7 | Summary of Literature Survey | |
| | | |
| **3** | **Methodology and Implementation** | |
| 3.1 | System Overview | |
| 3.2 | Dataset Description | |
| 3.3 | Text Preprocessing Pipeline | |
| 3.4 | Feature Extraction Using TF-IDF | |
| 3.5 | Model Selection and Training | |
| 3.6 | Model Evaluation | |
| 3.7 | Web Application Development | |
| 3.8 | Command-Line Interface | |
| 3.9 | Implementation Summary | |
| | | |
| **4** | **Results and Analysis** | |
| 4.1 | Dataset Statistics | |
| 4.2 | Model Performance Comparison | |
| 4.3 | Detailed Performance by Sentiment Class | |
| 4.4 | Feature Importance Analysis | |
| 4.5 | Error Analysis | |
| 4.6 | Visualization of Results | |
| 4.7 | Web Application Results | |
| 4.8 | Real-World Applicability | |
| 4.9 | Summary of Results | |
| | | |
| **5** | **Advantages, Limitations, and Applications** | |
| 5.1 | Advantages of the System | |
| 5.2 | Limitations of the System | |
| 5.3 | Applications of the System | |
| 5.4 | Summary | |
| | | |
| **6** | **Conclusion and Future Scope** | |
| 6.1 | Project Summary | |
| 6.2 | Key Achievements | |
| 6.3 | Lessons Learned | |
| 6.4 | Future Scope and Enhancements | |
| 6.5 | Recommendations for Deployment | |
| 6.6 | Final Thoughts | |
| 6.7 | Conclusion | |
| | | |
| | **References** | |
| | **Appendix A: Code Samples** | |
| | **Appendix B: User Manual** | |
| | **Appendix C: Glossary** | |

---

## LIST OF FIGURES

| Figure No. | Name of the Figure | Page No. |
|------------|-------------------|----------|
| 3.1 | System Architecture Diagram | |
| 3.2 | Text Preprocessing Pipeline Flowchart | |
| 3.3 | TF-IDF Feature Extraction Process | |
| 4.1 | Sentiment Distribution in Dataset | |
| 4.2 | Model Accuracy Comparison Chart | |
| 4.3 | Confusion Matrix for SVM Model | |
| 4.4 | Word Cloud for Positive Reviews | |
| 4.5 | Word Cloud for Negative Reviews | |
| 4.6 | Word Cloud for Neutral Reviews | |
| 4.7 | Feature Importance Bar Chart | |
| 4.8 | Training Time Comparison | |
| 4.9 | Web Application Interface Screenshot | |
| 4.10 | Confidence Score Visualization | |

---

## LIST OF TABLES

| Table No. | Name of the Table | Page No. |
|-----------|------------------|----------|
| 3.1 | Dataset Statistics Summary | |
| 3.2 | TF-IDF Vectorizer Parameters | |
| 3.3 | Model Training Parameters | |
| 4.1 | Overall Model Performance Comparison | |
| 4.2 | Class-wise Performance Metrics | |
| 4.3 | Top Positive Features | |
| 4.4 | Top Negative Features | |
| 4.5 | Error Distribution Analysis | |

---

## ABBREVIATIONS

| Abbreviation | Full Form |
|--------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| BERT | Bidirectional Encoder Representations from Transformers |
| CLI | Command-Line Interface |
| CPU | Central Processing Unit |
| CSV | Comma-Separated Values |
| EDA | Exploratory Data Analysis |
| F1 | F1-Score (Harmonic Mean of Precision and Recall) |
| GDPR | General Data Protection Regulation |
| GPU | Graphics Processing Unit |
| HTML | HyperText Markup Language |
| IDF | Inverse Document Frequency |
| JSON | JavaScript Object Notation |
| LIME | Local Interpretable Model-agnostic Explanations |
| LR | Logistic Regression |
| ML | Machine Learning |
| NB | Naive Bayes |
| NLP | Natural Language Processing |
| NLTK | Natural Language Toolkit |
| RAM | Random Access Memory |
| SHAP | SHapley Additive exPlanations |
| SVM | Support Vector Machine |
| TF | Term Frequency |
| TF-IDF | Term Frequency-Inverse Document Frequency |
| UI | User Interface |
| URL | Uniform Resource Locator |

---

