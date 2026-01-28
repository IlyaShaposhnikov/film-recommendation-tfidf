"""
Telegram бот для рекомендаций фильмов
"""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
import pandas as pd

from interfaces.config import (
    TELEGRAM_TOKEN, GENRES_NUM_RECOMMENDATIONS, ACTORS_NUM_RECOMMENDATIONS
)
from core.genres_recommender import GenresRecommender
from core.actors_recommender import ActorsRecommender

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация рекомендательных систем
genres_recommender = None
actors_recommender = None

# Состояния пользователей
user_states = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    welcome_text = (
        f"Привет, {user.first_name}!\n\n"
        "Я бот для рекомендаций фильмов!\n"
        "Я могу рекомендовать фильмы двумя способами:\n\n"
        "1. По жанрам и ключевым словам - "
        "ищу фильмы с похожим содержанием\n"
        "2. По актерскому составу - ищу фильмы с похожими актерами\n\n"
        "Выберите тип рекомендаций:"
    )

    keyboard = [
        [InlineKeyboardButton("По жанрам", callback_data='genres')],
        [InlineKeyboardButton("По актерам", callback_data='actors')],
        [InlineKeyboardButton("Помощь", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(
            welcome_text, reply_markup=reply_markup
        )
    elif update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_text,
            reply_markup=reply_markup
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = (
        "Помощь по командам:\n\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать это сообщение\n"
        "/genres - Рекомендации по жанрам\n"
        "/actors - Рекомендации по актерам\n\n"
        "Как пользоваться:\n"
        "1. Выберите тип рекомендаций\n"
        "2. Введите часть названия фильма\n"
        "3. Выберите фильм из списка\n"
        "4. Получите рекомендации!\n\n"
        "Важно: Названия фильмов нужно вводить на английском языке"
    )

    if update.message:
        await update.message.reply_text(help_text)
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text)


async def genres_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /genres"""
    await handle_recommendation_type(update, context, 'genres')


async def actors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /actors"""
    await handle_recommendation_type(update, context, 'actors')


async def handle_recommendation_type(
        update: Update, context: ContextTypes.DEFAULT_TYPE, rec_type: str
):
    """Обработчик выбора типа рекомендаций"""
    user_id = update.effective_user.id
    user_states[user_id] = {'type': rec_type, 'step': 'search'}

    text = "Введите часть названия фильма на английском (минимум 3 буквы):"

    if rec_type == 'genres':
        text = "Рекомендации по жанрам\n\n" + text
    else:
        text = "Рекомендации по актерам\n\n" + text

    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик поиска фильмов"""
    user_id = update.effective_user.id

    if user_id not in user_states or user_states[user_id]['step'] != 'search':
        await update.message.reply_text(
            "Пожалуйста, сначала выберите тип рекомендаций через /start"
        )
        return

    search_term = update.message.text.strip()

    if len(search_term) < 3:
        await update.message.reply_text(
            "Для поиска необходимо ввести минимум 3 символа."
        )
        return

    rec_type = user_states[user_id]['type']

    # Инициализация рекомендательной системы при первом использовании
    global genres_recommender, actors_recommender

    if rec_type == 'genres':
        if genres_recommender is None:
            await update.message.reply_text("Загружаю данные о фильмах...")
            genres_recommender = GenresRecommender()

        results = genres_recommender.search_movies(search_term)
    else:
        if actors_recommender is None:
            await update.message.reply_text("Загружаю данные об актерах...")
            actors_recommender = ActorsRecommender()

        results = actors_recommender.search_movies(search_term)

    if (isinstance(results, pd.DataFrame) and results.empty) or (
        isinstance(results, list) and not results
    ):
        await update.message.reply_text(
            f"Фильмы с названием, содержащим '{search_term}', не найдены."
        )
        return

    # Сохраняем результаты поиска
    user_states[user_id]['search_results'] = results
    user_states[user_id]['step'] = 'select'

    # Формируем сообщение с результатами
    if isinstance(results, pd.DataFrame):
        # Для жанров
        text = f"Найдено фильмов: {len(results)}\n\n"
        for i, (_, row) in enumerate(results.iterrows(), 1):
            text += f"{i}. {row['title']}"
            if 'release_year' in row and row['release_year'] > 0:
                text += f" ({row['release_year']})"
            text += "\n"
    else:
        # Для актеров
        text = f"Найдено фильмов: {len(results)}\n\n"
        for i, title in enumerate(results, 1):
            text += f"{i}. {title}\n"

    text += "\nВыберите номер фильма для получения рекомендаций:"

    # Создаем клавиатуру с номерами
    keyboard = []
    max_per_row = 3
    for i in range(0, min(len(results), 10), max_per_row):
        row = []
        for j in range(max_per_row):
            idx = i + j + 1
            if idx <= min(len(results), 10):
                row.append(InlineKeyboardButton(
                    str(idx), callback_data=f'select_{idx-1}')
                )
        keyboard.append(row)

    keyboard.append([InlineKeyboardButton(
        "Новый поиск", callback_data=rec_type
    )])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup)


async def handle_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора фильма из списка"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    if user_id not in user_states:
        await query.edit_message_text(
            "Сессия устарела. Пожалуйста, начните заново с /start"
        )
        return

    data = query.data

    if data.startswith('select_'):
        idx = int(data.split('_')[1])
        rec_type = user_states[user_id]['type']
        results = user_states[user_id]['search_results']

        # Получаем выбранный фильм
        if isinstance(results, pd.DataFrame):
            selected_movie = results.iloc[idx]['title']
        else:
            selected_movie = results[idx]

        await query.edit_message_text(
            f"Ищу рекомендации для '{selected_movie}'..."
        )

        # Получаем рекомендации
        if rec_type == 'genres':
            requested_movie, recommendations = genres_recommender.recommend(
                selected_movie,
                num_recommendations=GENRES_NUM_RECOMMENDATIONS
            )
        else:
            requested_movie, recommendations = actors_recommender.recommend(
                selected_movie,
                num_recommendations=ACTORS_NUM_RECOMMENDATIONS
            )

        # Формируем ответ
        if isinstance(recommendations, str):
            text = f"{recommendations}"
        else:
            if rec_type == 'genres':
                text = (
                    "Рекомендации для фильма\n"
                    f"{requested_movie['title']}"
                )
                if requested_movie['release_year'] > 0:
                    text += f" ({requested_movie['release_year']})"
                text += "\n\n"

                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    text += f"{i}. {row['title']}"
                    if row['release_year'] > 0:
                        text += f" ({row['release_year']})"
                    text += f" - схожесть: {row['similarity_score']}\n"
            else:
                text = (
                    "Фильмы с похожим актерским составом\n"
                    f"{requested_movie}\n\n"
                )

                for i, rec in enumerate(recommendations, 1):
                    text += (
                        f"{i}. {rec['title']} - "
                        f"схожесть: {rec['similarity_score']}\n"
                    )

        text += "\nЧто вы хотите сделать дальше?"

        keyboard = [
            [InlineKeyboardButton("Новый поиск", callback_data=rec_type)],
            [InlineKeyboardButton("Сменить тип", callback_data='change_type')],
            [InlineKeyboardButton("В начало", callback_data='start')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(text, reply_markup=reply_markup)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data in ['genres', 'actors']:
        await handle_recommendation_type(update, context, data)
    elif data == 'help':
        await help_command(update, context)
    elif data == 'start':
        await start(update, context)
    elif data == 'change_type':
        await start(update, context)
    elif data.startswith('select_'):
        await handle_selection(update, context)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка при обработке обновления {update}: {context.error}")

    # Отправляем сообщение об ошибке пользователю
    try:
        if update and update.message:
            await update.message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте еще раз или "
                "используйте команду /start для начала работы."
            )
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение об ошибке: {e}")


def main():
    """Основная функция запуска бота"""
    if not TELEGRAM_TOKEN:
        print("Ошибка: Токен бота не найден!")
        print("Создайте файл .env и добавьте TELEGRAM_BOT_TOKEN")
        return

    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("genres", genres_command))
    application.add_handler(CommandHandler("actors", actors_command))

    # Обработчик текстовых сообщений
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_search)
    )

    # Обработчик нажатий на кнопки
    application.add_handler(CallbackQueryHandler(button_handler))

    # Добавляем обработчик ошибок
    application.add_error_handler(error_handler)

    # Запуск бота
    print("Бот запускается...")
    print("Перейдите в Telegram и найдите вашего бота")
    print("Для остановки нажмите Ctrl+C")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
