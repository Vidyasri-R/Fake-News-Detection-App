import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📰 Fake News Detector")

# --- Load Dataset from Google Drive ---
@st.cache_data
def load_data():
    # Convert Google Drive links to direct download format
    fake_url = "https://drive.google.com/uc?id=1VjBU1-tZtJMsc17LZb8x4yTt50H0-tQD"
    true_url = "https://drive.google.com/uc?id=1LuuN8DWzVsuij-Ckel3cjchBkpPth4gi"


    # Read CSVs directly from Drive
    fake = pd.read_csv(fake_url)
    true = pd.read_csv(true_url)

    # Add labels
    fake['label'] = 'Fake'
    true['label'] = 'Real'

    # Combine datasets
    data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
    return data

# --- Load data ---
data = load_data()

# --- Feature and Label selection ---
X = data['text']
y = data['label']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Text Vectorization ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Model Training ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- Model Accuracy ---
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# --- User Input Section ---
st.write("### 📰 Enter a News Article Below:")
text = st.text_area("", height=150)

# --- Prediction Button ---
if st.button("Check"):
    if text.strip():
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        if prediction == 'Fake':
            st.error("🚨 This is FAKE NEWS")
        else:
            st.success("✅ This is REAL NEWS")
    else:
        st.warning("Please enter some text to analyze.")
