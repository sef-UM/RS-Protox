import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os

# Download NLTK data if not already present
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Example training data (for demonstration)
data = {
    'text': [
        'I hate you',
        'You are so stupid',
        'Have a nice day',
        'I love this',
        'You are an idiot',
        'What a wonderful world',
        'You are disgusting',
        'Such a pleasant surprise',
        'I despise your actions',
        'You are amazing',
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = hatespeech, 0 = not
}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(preprocess)

# Vectorizer and model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text_clean'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

def predict_hatespeech(text):
    text_clean = preprocess(text)
    X_test = vectorizer.transform([text_clean])
    pred = model.predict(X_test)[0]
    return 'Hatespeech' if pred == 1 else 'Not Hatespeech'
