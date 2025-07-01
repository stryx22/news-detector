# fake-news-detector/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from utils.preprocess import clean_text

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Label the data
fake["label"] = 1
real["label"] = 0

# Combine and shuffle
data = pd.concat([fake, real], axis=0).sample(frac=1, random_state=42)
data["text"] = data["text"].apply(clean_text)

# Split
tf = TfidfVectorizer(max_features=5000)
X = tf.fit_transform(data["text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(tf, "models/vectorizer.pkl")