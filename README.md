<div style="display: flex; align-items: center;">
  <img src="img/logo_moodai.png" alt="Image" width="100" style="margin-right: 20px;">
  <h1>moodAI - Emotion detection with AI</h1>
</div>

This project shows how we built a sentiment analysis tool that detects emotions in text. It is based on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) and uses NLP techniques and Deep Learning models to detect emotions in text. It is part of the Unstructured Data Analysis course at Universidad Pontificia Comillas.

The goal of this tool is to help people who are not able to detect emotions in text on their own, such as people with autism or other mental health issues.

The tool is available as a [Streamlit web application](https://moodai.streamlit.app/), where users can input text, select the model they want to use and get the emotions detected in the text.

## 🔍 Exploratory Data Analysis and Data Preprocessing
The data used in this project is the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) from Google Research. This dataset contains 58,000 English Reddit comments labeled with 28 emotions. We have leaved out the comments with more than one emotion and the ones with no emotion labeled. Also, we have dropped the comments with rare emotions, so we have ended up with 23 emotions.

The data has been preprocessed by removing special characters, numbers, and stopwords. We have also tokenized the text and lemmatized it.

An exhaustive Exploratory Data Analysis has been carried out to understand the data and the emotions in it and is available in the [EDA.ipynb notebook](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/EDA.ipynb).

## 🤖 AI Models
We first trained two simpler models to use as a baseline:
- [Linear Regression](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/LinearReg_Naive.ipynb)
- [Naive Bayes](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/LinearReg_Naive.ipynb)

Then, we trained some Deep Learning models from scratch and using GloVe embeddings from the [GloVe project](https://nlp.stanford.edu/projects/glove/):
- [Recurrent Neural Network (RNN)](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/RNN.ipynb)
- [RNN with GloVe embeddings](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/RNN_GloVe.ipynb)
- [Long Short-Term Memory (LSTM)](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/LSTM.ipynb)
- [Long Short-Term Memory (LSTM) with GloVe embeddings](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/LSTM%2BGloVe.ipynb)
- [Convolutional Neural Network (CNN)](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/CNN.ipynb)
- [Convolutional Neural Network (CNN) with GloVe embeddings](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/CNN%2BGloVe.ipynb)

We have also trained a model using pre-trained models:
- [DenseNet](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/DenseNet.ipynb)
- [BERT](https://github.com/MARIABELENSB/sentimentAnalysis/blob/main/PretrainedModel.ipynb)


## 🚀 Deployment
The tool is available as a Streamlit web app, by running `streamlit run app.py`, where users can input text, select the model they want to use, and get the emotions detected in the text.
[![Watch the video](img/Captura%20de%20pantalla%202024-05-14%20175840.png)](img/moodai.mp4)

## 🛠️ Technologies used
- pandas 🐼
- numpy 🧮
- matplotlib 📊
- nltk 📚
- scikit-learn 🧠
- tensorflow keras 🤖
- streamlit 🚀
