from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils import load_csv_dataset

# Load data from CSV
X, y = load_csv_dataset("train_data.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF + Classifier pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save the model
joblib.dump(model, 'model/resume_classifier.pkl')
