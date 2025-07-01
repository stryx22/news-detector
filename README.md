# 📰 Fake News Detection System with LIME Explanation



## 🧰 Tools & Concepts Used

**Tools & Libraries:**

* Python
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn
* Streamlit
* LIME (Local Interpretable Model-Agnostic Explanations)
* NLTK
* WordCloud

**Concepts Applied:**

* Natural Language Processing (NLP)
* Text preprocessing & vectorization (TF-IDF)
* Binary classification using Logistic Regression
* Model interpretation and explainability
* Exploratory Data Analysis (EDA)

---

## 🔍 Overview

This is a complete **Fake News Detection System** that uses **Machine Learning** (Logistic Regression with TF-IDF) to classify news articles as **real** or **fake**. It features:

* ✅ End-to-end pipeline: EDA → Preprocessing → Training → Web App
* 🎯 Model Explainability using **LIME**
* 🌐 Interactive UI built with **Streamlit**

---

## 🗂️ Project Structure

```
fake-news-detector/
├── app/
│   └── app.py                # Streamlit Web App
├── data/
│   ├── Fake.csv              # Fake news dataset
│   └── True.csv              # Real news dataset
├── models/
│   ├── fake_news_model.pkl   # Trained ML model
│   └── vectorizer.pkl        # TF-IDF Vectorizer
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
├── utils/
│   ├── preprocess.py         # Text cleaning
│   └── explain.py            # LIME explanation logic
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
└── README.md                 # Project info and usage
```

---

## 📊 EDA Preview (`eda.ipynb`)

* Class balance
* Word count distributions
* Word clouds for fake vs real

---

## 🧠 Model

* **TF-IDF** text vectorization
* **Logistic Regression** classifier
* **LIME** for interpretability

---

## 🖼️ App Features

* Input raw news article
* Get immediate prediction
* View explainability with LIME in an interactive panel

---

## 🛠 Built With

* Python 🐍
* Scikit-learn ⚙️
* Streamlit 🌐
* LIME 🧠
* NLTK + WordCloud 📊

---

## 🤝 Contributing

Feel free to fork and submit pull requests. Suggestions and improvements are welcome!

---

## 📄 License

MIT License © 2025 Shubhayu Roy
