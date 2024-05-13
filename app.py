import sys
print(sys.executable)

import re

import streamlit as st

import pickle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TOKENIZER_PATH = "models/tokenizer.pickle"

MODELS = [
    {
        "name": "Recurrent Neural Network",
        "path": "models/rnn.h5"
    },
    {
        "name": "Recurrent Neural Network with GloVe embeddings",
        "path": "models/rnn_glove.h5"
    },
    {
        "name": "Convolutional Neural Network",
        "path": "models/cnn_model.keras"
    },
    {
        "name": "Convolutional Neural Network with GloVe embeddings",
        "path": "models/cnn_model_glove.keras"
    },
    {
        "name": "Long Short Term Memory Network",
        "path": "models/lstm_l2_model.keras"
    },
    {
        "name": "Long Short Term Memory Networks with GloVe embeddings",
        "path": "models/lstm_glove_model.keras"
    },
    {
        "name": "Logistic Regression with Bag-of-Words",
        "path": "models/logReg_bow.joblib"
    },
    {
        "name": "Logistic Regression with TF-IDF",
        "path": "models/logReg_tfidf.joblib"
    },
    {
        "name": "Naive Bayes with TF-IDF",
        "path": "models/naive_tfidf.joblib"
    },
    {
        "name": "Naive Bayes with Bag-of-Words",
        "path": "models/naive_bow.joblib"
    },
    {
        "name": "DenseNet",
        "path": "models/denseNet_model.h5"
    },
]

emotion_to_emoji = {
    'admiration': '🤩',
    'amusement': '😄',
    'anger': '😡',
    'annoyance': '😑',
    'approval': '👍',
    'caring': '🥰',
    'confusion': '😕',
    'curiosity': '🤔',
    'desire': '😏',
    'disappointment': '😞',
    'disapproval': '👎',
    'disgust': '🤢',
    'embarrassment': '😳',
    'excitement': '😃',
    'fear': '😨',
    'gratitude': '🙏',
    'joy': '😀',
    'love': '❤️',
    'neutral': '😐',
    'optimism': '😊',
    'realization': '😲',
    'sadness': '😢',
    'surprise': '😮'
}

def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

def remove_special_characters(sentence, remove_digits=False):
    print(f'Removing special characters from sentence: {sentence}')
    pattern = r'/[^\w-]|_/' if not remove_digits else r'[^a-zA-Z\s]'  
    clean_text = re.sub(pattern, '', sentence)
    print(f'Cleaned sentence: {clean_text}')
    return clean_text

def preprocess_input(text, maxlen=18):
    # Download the NLTK resources and initialize the lemmatizer
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Remove special characters
    text = remove_special_characters(text, remove_digits=True)

    # Apply lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Text to lowercase
    text = text.lower()

    # Tokenize the input text
    tokenizer = load_tokenizer()
    text = tokenizer.texts_to_sequences([text])

    # Pad the input text
    text = pad_sequences(text, maxlen=maxlen)

    return text

def predict_sentiment_joblib(text, model_used):
    if isinstance(text, str):
        text = [text]
    print(f'text: {text}')
    print(f"Predicting sentiment for tokenized text: {text}")
    prediction = model_used.predict_proba(text)[0]
    print(f"Prediction: {prediction}")
    # prediction is a list of probabilities for each class
    # return the top 3 classes with the highest probabilities
    # as well as the corresponding emojis
    # as a list of tuples [(emoji, emotion, probability), ...]
    # Obtener las clases del modelo
    classes = model_used.classes_
    print("Classes:", classes)

    # Encontrar los índices de las tres mayores probabilidades
    top_indices = prediction.argsort()[-3:][::-1]
    print(f"Top classes indices: {top_indices}")

    # Recolectar la información de las tres mejores clases
    top_classes_info = [(emotion_to_emoji[classes[i]], classes[i], prediction[i]) for i in top_indices]
    print("Top classes information:", top_classes_info)

    return top_classes_info

def predict_sentiment(text, model_used):
    print(f"Predicting sentiment for tokenized text: {text}")
    prediction = model_used.predict(text)[0]
    print(f"Prediction: {prediction}")
    # prediction is a list of probabilities for each class
    # return the top 3 classes with the highest probabilities
    # as well as the corresponding emojis
    # as a list of tuples [(emoji, emotion, probability), ...]
    top_classes = prediction.argsort()[-3:][::-1]
    print(f"Top classes: {top_classes}")
    emotion_labels = list(emotion_to_emoji.keys())
    print(emotion_labels)
    top_classes_info = [(emotion_to_emoji[emotion_labels[top_class]], emotion_labels[top_class], prediction[top_class]) for top_class in top_classes]
    print(top_classes_info)
    return top_classes_info

# Streamlit app
def main():

    # Favicon
    st.set_page_config(page_title="moodAI", page_icon="img/logo_moodai.png")
    
    # Title
    st.image("img/moodAI.png", use_column_width=True)
    
    # Subheader and description
    st.subheader("Helping you understand the emotions behind text messages")
    # Description
    st.markdown("moodAI is designed to **empower** individuals, especially those with special needs, in **deciphering the emotions of others**.")

    # Text input for user input
    text = st.text_input("Enter a text message below to see the sentiment analysis result:")

    # Dropdown for selecting the model
    model_selected = st.selectbox("Select a model:", [model["name"] for model in MODELS])

    # Load the selected model
    model_path = [model["path"] for model in MODELS if model["name"] == model_selected][0]
    print(f'Loading model from path: {model_path}')

    if model_path.endswith(".joblib"):
        model_loaded_joblib = joblib.load(model_path)
        print(f'Model loaded: {model_loaded_joblib}')

    else:
        model_loaded = load_model(model_path)
        print(f'Model loaded: {model_loaded}')

    # Add a button to trigger the classification
    if st.button("Analyze"):
        if len(text.split()) > 18:
            st.error("The input text is too long. Please enter a text with less than 18 words.")
        else:
            # Determine the correct maximum length for padding based on the model selected
            if "Convolutional" in model_selected or "Long" in model_selected:
                processed_text = preprocess_input(text, maxlen=19)
            else:
                processed_text = preprocess_input(text, maxlen=18)

            # Predict the sentiment
            if model_path.endswith(".joblib"):
                result = predict_sentiment_joblib(text, model_loaded_joblib)
            else:
                result = predict_sentiment(processed_text, model_loaded)

            # Display the sentiment analysis results
            st.markdown("<h4>Sentiment analysis result:</h4>", unsafe_allow_html=True)
            for emotion in result:
                st.markdown(f"<h4>{emotion[0]} {emotion[1]} - {emotion[2]*100:.0f} %</h4>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
