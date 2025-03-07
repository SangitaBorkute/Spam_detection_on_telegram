import logging
import joblib
import asyncio
import re
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Telegram Bot Token
TOKEN = "8190103208:AAFJ0_RSYqt9mjvu4pXKkEd5YB77BT4jhVE"

# Load trained models and vectorizers
message_model = joblib.load("count_mnb_model.pkl")
message_vectorizer = joblib.load("count_vectorizer.pkl")

link_model = joblib.load("linkmodel.pkl")
link_vectorizer = joblib.load("tfidf_vectorizer.pkl")

fake_news_model = joblib.load("fake_news_model.pkl")
news_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Preprocessing Class for Message Spam Detection
class SpamDetection:
    def __init__(self, review):
        self.review = review

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
        
        vector_input = message_vectorizer.transform([" ".join(clean_text)])
        result = message_model.predict(vector_input)[0]
        return "ham" if result == 1 else "spam"

# Function to classify URLs
async def classify_url(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    transformed_text = link_vectorizer.transform([user_message])
    prediction = link_model.predict(transformed_text)[0]
    await update.message.reply_text(f"The URL is classified as: {prediction}")

# Function to classify messages (spam or ham)
async def detect_spam(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    spam = SpamDetection(user_message)
    prediction = spam.message_prediction()
    await update.message.reply_text(f"Message Result: {prediction}")

# Function to classify news articles
async def classify_news(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    transformed_text = news_vectorizer.transform([user_message])
    prediction = fake_news_model.predict(transformed_text)[0]
    result = "Fake News" if prediction == 1 else "True News"
    await update.message.reply_text(f"The news is classified as: {result}")

# Function to determine if input is a link, spam, or news
async def classify_input(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text

    # Regex to detect if the message contains a URL
    url_pattern = r"(https?://\S+|www\.\S+)"
    if re.search(url_pattern, user_message):
        await classify_url(update, context)
    elif len(user_message.split()) > 30:  # Assuming long text is a news article
        await classify_news(update, context)
    else:
        await detect_spam(update, context)

# Start Command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! Send me a message, link, or news article, and I'll classify it.")

# Main Function
def main():
    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_input))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
