import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')
df['text'] = df['text'].astype(str).str.lower()

X_text = df['text']
y = df['label']

# Split for evaluation
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorize
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2),max_features=10000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# Simple Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=['Not Bullying','Bullying']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save model for app.py
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
print("\nModel saved as model.pkl and vectorizer.pkl")
