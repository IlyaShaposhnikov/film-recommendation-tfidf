import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Telegram Bot Token
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Пути к данным
DATA_PATH = 'data'
MOVIES_CSV = os.path.join(DATA_PATH, 'tmdb_5000_movies.csv')
CREDITS_CSV = os.path.join(DATA_PATH, 'tmdb_5000_credits.csv')

# Настройки рекомендаций
GENRES_NUM_RECOMMENDATIONS = 5
ACTORS_NUM_RECOMMENDATIONS = 10
SEARCH_RESULTS_LIMIT = 10
