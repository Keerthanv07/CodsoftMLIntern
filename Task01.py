# MOVIE GENRE CLASSIFICATION

import kagglehub
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
print("Dataset downloaded to:", path)

train_file = os.path.join(path, "Genre Classification Dataset", "train_data.txt")

data = []
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) >= 4:
            genre = parts[2]
            plot = parts[3]
            data.append((genre, plot))

df = pd.DataFrame(data, columns=["genre", "plot"])
print("Loaded data sample:")
print(df.head())

if df.empty:
    raise ValueError("No valid data loaded. Please check train_data.txt format.")

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['plot'])
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.show()