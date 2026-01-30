"""
Модуль для загрузки и подготовки данных из CSV файлов TMDB.
Обеспечивает базовые функции для работы с данными о фильмах и актерах.
Основные задачи:
1. Загрузка CSV файлов в DataFrame
2. Извлечение структурированных данных из JSON-строк
3. Преобразование данных для использования в рекомендательных системах
"""
import pandas as pd
import json


def load_movies_data():
    """
    Загрузка данных о фильмах из файла tmdb_5000_movies.csv.

    Возвращает:
        DataFrame с колонками:
        - title: название фильма
        - genres: JSON-строка с жанрами
        - keywords: JSON-строка с ключевыми словами
        - release_date: дата выхода
        - release_year: год выхода (вычисляется из release_date)

    Примечания:
        - release_year преобразуется в целое число
        - Пропущенные значения заполняются 0
        - Ошибки парсинга даты игнорируются (errors='coerce')
    """
    df = pd.read_csv('data/tmdb_5000_movies.csv')
    df['release_year'] = pd.to_datetime(
        df['release_date'], errors='coerce'
    ).dt.year
    df['release_year'] = df['release_year'].fillna(0).astype(int)
    return df


def load_credits_data():
    """
    Загрузка данных об актерах и съемочной группе
    из файла tmdb_5000_credits.csv.

    Возвращает:
        DataFrame с колонками:
        - movie_id: уникальный идентификатор фильма
        - title: название фильма
        - cast: JSON-строка с информацией об актерах
        - crew: JSON-строка с информацией о съемочной группе
    """
    df = pd.read_csv('data/tmdb_5000_credits.csv')
    return df


def extract_genres_and_keywords(row):
    """
    Извлечение жанров и ключевых слов из JSON-строк в единую текстовую строку.
    Эта функция выполняет преобразование данных из структурированного
    JSON-формата в плоский текстовый формат, который может быть
    обработан моделью TF-IDF.

    Аргументы:
        row: строка DataFrame с колонками 'genres' и 'keywords'

    Возвращает:
        Единую строку, содержащую все жанры и ключевые слова,
        разделенные пробелами, склеив все состаные названия

    Алгоритм:
        1. Парсинг JSON из колонок 'genres' и 'keywords'
        2. Извлечение названий жанров и ключевых слов
        3. Удаление пробелов внутри составных названий
           (например, "Science Fiction" → "ScienceFiction")
        4. Склейка всех слов в одну строку

    Пример преобразования:
        Вход: {"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}
        Выход: "Action Adventure"

    Обработка ошибок:
        При ошибках парсинга возвращается пустая строка
    """

    try:
        genres = json.loads(row['genres'])
        # j['name'] - получаем название жанра из словаря
        # .replace(' ', '') - удаляем все пробелы в названии
        # (например, "Science Fiction" → "ScienceFiction")
        # ' '.join(...) - объединяем все обработанные названия жанров
        # в одну строку, разделяя пробелами
        genres_str = ' '.join(j['name'].replace(' ', '') for j in genres)

        keywords = json.loads(row['keywords'])
        keywords_str = ' '.join(j['name'].replace(' ', '') for j in keywords)

        return f"{genres_str} {keywords_str}"
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return ""


def extract_weighted_actors(cast_json, max_actors=10):
    """
    Извлекает актеров с весами на основе порядка в титрах.

    Аргументы:
        cast_json: JSON-строка с информацией об актерах
        max_actors: максимальное количество актеров для учета (по умолчанию 10)

    Возвращает:
        Строку с именами актеров, повторенными пропорционально их весу

    Логика весов:
        - Актеры сортируются по порядку в титрах (order)
        - Вес вычисляется по формуле: weight = 1 / (order + 1)
        - Количество повторений: repeats = max(1, int(weight * 5))
        - Главный актер (order=0): weight=1, repeats=5
        - Второй актер (order=1): weight=0.5, repeats=2
        - и т.д.

    Пример:
        Вход: [{"name": "Actor1", "order": 0}, {"name": "Actor2", "order": 1}]
        Выход: "Actor1 Actor1 Actor1 Actor1 Actor1 Actor2 Actor2"

    Обработка ошибок:
        При ошибках парсинга возвращается пустая строка
    """
    try:
        cast = json.loads(cast_json)
        # если order отсутствует, устанавливаем значение 100 (окажутся в конце)
        cast.sort(key=lambda x: x.get('order', 100))
        top_cast = cast[:max_actors]

        weighted_names = []
        for i, actor in enumerate(top_cast):
            name = actor['name']
            weight = 1 / (i + 1)
            # max(1, ...) гарантирует хотя бы одно повторение
            repeats = max(1, int(weight * 5))
            # Добавляем имя repeats раз в список
            weighted_names.extend([name] * repeats)

        return ' '.join(weighted_names)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return ""
