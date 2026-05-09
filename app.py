import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\bhawa\OneDrive\Documents\Downloads\archive (3)\UpdatedResumeDataSet.csv")

# Show first 5 rows
print(df.head())

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Clean resume text
df['Resume'] = df['Resume'].apply(clean_text)

# Features and labels
X = df['Resume']
y = df['Category']

# Convert text into vectors
tfidf = TfidfVectorizer(stop_words='english')

X_vectorized = tfidf.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Sample resume prediction
sample_resume = """
Python SQL Machine Learning Deep Learning
Pandas NumPy Data Analysis
"""

sample_vector = tfidf.transform([sample_resume])

prediction = model.predict(sample_vector)

print("\nPredicted Category:", prediction[0])