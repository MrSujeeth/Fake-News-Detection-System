Fake News Detection System Model
# 📰 Fake News Detection System using Machine Learning

This project is a **Machine Learning–based Fake News Detection System** that classifies news articles as **Fake** or **Genuine** using **Natural Language Processing (NLP)** and multiple supervised learning algorithms.

---

## 🚀 Project Overview

Fake news spreads rapidly through online platforms and can mislead people.  
This project analyzes news text data and predicts whether a given news article is **Fake** or **Real (Genuine)** using machine learning models trained on labeled datasets.

---

## 🎯 Objectives

- Detect fake news automatically  
- Apply NLP techniques for text preprocessing  
- Train and compare multiple ML models  
- Allow manual testing of news articles  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Regular Expressions (re)

---

## 📂 Dataset

- **True.csv** → Contains genuine news articles  
- **Fake.csv** → Contains fake news articles  

Each dataset includes:
- `title`
- `text`
- `subject`
- `date`

Labels assigned:
- **1 → Genuine News**
- **0 → Fake News**

---

## 🔄 Data Preprocessing

- Merged True and Fake datasets  
- Removed unnecessary columns (`title`, `subject`, `date`)  
- Shuffled data randomly  
- Cleaned text by:
  - Converting to lowercase  
  - Removing URLs, HTML tags  
  - Removing punctuation, digits, and new lines  

---

## 🧠 Feature Extraction

- **TF-IDF Vectorization**  
  - Converts text into numerical features  
  - Captures important words while reducing noise  

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**

---

## 📊 Model Evaluation

- Dataset split:
  - **70% Training**
  - **30% Testing**
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Used `classification_report` for performance comparison

---

## 🧪 Manual Testing Feature

Users can manually enter a news article, and the system predicts whether it is **Fake** or **Genuine** using multiple models.

### Example Output:
