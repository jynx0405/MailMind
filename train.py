import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("data/emails.csv")

X = df['text']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("model/logistic_model.pkl", "wb"))

print("Model training complete.")
