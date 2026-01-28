"""
Конфигурационный файл для Telegram бота
"""
import os
from dotenv import load_dotenv

# Находим корневую папку проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Загружаем переменные окружения из .env в корне проекта
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Telegram Bot Token
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Настройки рекомендаций
GENRES_NUM_RECOMMENDATIONS = 5
ACTORS_NUM_RECOMMENDATIONS = 10
