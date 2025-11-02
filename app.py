import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📰 Fake News Detector")


fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['label'] = 'Fake'
true['label'] = 'Real'

data = pd.concat([fake, true])


X = data['text']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)


accuracy = accuracy_score(y_test, model.predict(X_test_vec))
st.success(f"Model Accuracy: {accuracy*100:.1f}%")

st.write("### Enter news article:")
text = st.text_area("", height=150)

if st.button("Check"):
    if text:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        
        if prediction == 'Fake':
            st.error("🚨 This is FAKE NEWS")
        else:
            st.success("✅ This is REAL NEWS")
    else:
        st.warning("Please enter some text")