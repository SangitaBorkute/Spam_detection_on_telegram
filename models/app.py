import numpy as np
import joblib
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
import pandas as pd

class SpamDetection:
    def __init__(self, review):
        self.review = review
        self.load_model()

    def load_model(self):
        try:
            with open(r"C:\Users\sangi\OneDrive\Desktop\spam detection project\models\count_mnb_model.pkl", 'rb') as f:
                self.model = joblib.load(f)  # Load the trained model
            
            self.count_model = joblib.load(r"C:\Users\sangi\OneDrive\Desktop\spam detection project\models\count_vectorizer.pkl")  # Load CountVectorizer
        except Exception as e:
            print(f"Error loading model or vectorizer: {e}")
            exit()

    def remove_spaces(self, data):
        return data.replace('\n', ' ').replace("\t", ' ').replace('\\', ' ')

    def expand_text(self, data):
        return contractions.fix(data)

    def handling_accented(self, data):
        return unidecode(data)

    @staticmethod
    def clean_data(data):
        stopword_list = stopwords.words("english")
        tokens = word_tokenize(data)
        return [word.lower() for word in tokens if (word not in punctuation) and (word.lower() not in stopword_list) and (len(word) > 2) and (word.isalpha())]

    def autocorrection(self, data):
        spell = Speller(lang='en')
        return spell(data)

    def lemmatization(self, data):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in data])

    def message_prediction(self):
        clean_text = self.remove_spaces(self.review)
        clean_text = self.expand_text(clean_text)
        clean_text = self.handling_accented(clean_text)
        clean_text = self.clean_data(clean_text)
        clean_text = self.lemmatization(clean_text)
        
        vector_input = self.count_model.transform([clean_text])
        result = self.model.predict(vector_input)[0]
        return "ham" if result == 1 else "spam"

if __name__ == "__main__":
    while True:
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        spam = SpamDetection(user_input)
        prediction = spam.message_prediction()
        print(f"Result: {prediction}\n")
