import logging
import joblib
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram.error import TimedOut, NetworkError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Telegram Bot Token
TOKEN = "8190103208:AAFJ0_RSYqt9mjvu4pXKkEd5YB77BT4jhVE"

# Load trained models and vectorizers
try:
    # Spam detection models
    message_model = joblib.load("count_mnb_model.pkl")
    message_vectorizer = joblib.load("count_vectorizer.pkl")

    # URL classification models
    link_model = joblib.load("linkmodel.pkl")
    link_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Fake news detection models (Updated - Separate Models)
    title_model = joblib.load("naive_bayes_title_model.pkl")
    text_model = joblib.load("naive_bayes_text_model.pkl")
    title_vectorizer = joblib.load("tfidf_vectorizer_title.pkl")
    text_vectorizer = joblib.load("tfidf_vectorizer_text.pkl")

    logger.info("Models and vectorizers loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or vectorizers: {e}")
    raise

# Function to clean text (Consistent with training)
def clean_text(text):
    text = text.lower().strip()  # Convert to lowercase
    text = re.sub(r"\W+", " ", text)  # Remove special characters
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text.strip()

# Function to classify URLs
async def classify_url(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    try:
        transformed_text = link_vectorizer.transform([user_message])
        prediction = link_model.predict(transformed_text)[0]
        await update.message.reply_text(f"The URL is classified as: {prediction}")
    except Exception as e:
        logger.error(f"Error classifying URL: {e}")
        await update.message.reply_text("Sorry, I couldn't classify the URL. Please try again.")

# Function to classify messages (spam or ham)
async def detect_spam(update: Update, context: CallbackContext) -> None:
    user_message = clean_text(update.message.text)  # Use consistent cleaning
    try:
        transformed_text = message_vectorizer.transform([user_message])
        prediction = message_model.predict(transformed_text)[0]
        result = "✅ Ham" if prediction == 1 else "🚨 Spam ❌"
        await update.message.reply_text(f"Message Result: {result}")
    except Exception as e:
        logger.error(f"Error detecting spam: {e}")
        await update.message.reply_text("Sorry, I couldn't detect spam. Please try again.")

# Function to classify news articles using separate title & text models
async def classify_news(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    cleaned_text = clean_text(user_message)  # Use consistent cleaning

    try:
        # Predict using both title and text models
        title_transformed = title_vectorizer.transform([cleaned_text])
        text_transformed = text_vectorizer.transform([cleaned_text])

        title_prediction = title_model.predict(title_transformed)[0]
        text_prediction = text_model.predict(text_transformed)[0]

        # Decision Logic: If either predicts Fake News, classify as Fake
        if title_prediction == 1 or text_prediction == 1:
            result = "🚨 Fake News ❌"
        else:
            result = "📰 True News ✅"

        await update.message.reply_text(f"The news is classified as: {result}")

    except Exception as e:
        logger.error(f"Error classifying news: {e}")
        await update.message.reply_text("Sorry, I couldn't classify the news. Please try again.")

# Function to determine if input is a link, spam, or news
async def classify_input(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    url_pattern = r"(https?://\S+|www\.\S+)"

    # Keywords to detect news
    news_keywords = ["breaking", "report", "news", "government", "politics", "media", "journalist"]

    if re.search(url_pattern, user_message):  # Check if message is a URL
        await classify_url(update, context)
    elif any(word in user_message.lower() for word in news_keywords) or len(user_message.split()) > 30:
        await classify_news(update, context)  # Improved news detection logic
    else:
        await detect_spam(update, context)  # Default to spam detection

# Start Command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! Send me a message, link, or news article, and I'll classify it.")

# Error Handler
async def error_handler(update: Update, context: CallbackContext) -> None:
    logger.error(f"Update {update} caused error: {context.error}")
    if isinstance(context.error, (TimedOut, NetworkError)):
        await update.message.reply_text("Sorry, I encountered a network issue. Please try again later.")
    else:
        await update.message.reply_text("An unexpected error occurred. Please try again.")

# Main Function
def main():
    app = Application.builder().token(TOKEN).read_timeout(30).write_timeout(30).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_input))
    app.add_error_handler(error_handler)
    
    logger.info("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
