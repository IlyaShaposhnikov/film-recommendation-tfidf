"""
Модуль для загрузки и подготовки данных
"""
import pandas as pd
import json


def load_movies_data():
    """Загрузка данных о фильмах"""
    df = pd.read_csv('data/tmdb_5000_movies.csv')
    df['release_year'] = pd.to_datetime(
        df['release_date'], errors='coerce'
    ).dt.year
    df['release_year'] = df['release_year'].fillna(0).astype(int)
    return df


def load_credits_data():
    """Загрузка данных об актерах"""
    df = pd.read_csv('data/tmdb_5000_credits.csv')
    return df


def extract_genres_and_keywords(row):
    """Извлечение жанров и ключевых слов из JSON"""
    try:
        genres = json.loads(row['genres'])
        genres = ' '.join(' '.join(j['name'].split()) for j in genres)

        keywords = json.loads(row['keywords'])
        keywords = ' '.join(' '.join(j['name'].split()) for j in keywords)

        return f"{genres} {keywords}"
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return ""


def extract_weighted_actors(cast_json, max_actors=10):
    """
    Извлекает актеров с весами на основе порядка в титрах.
    Простая формула: weight = 1 / (order + 1)
    """
    try:
        cast = json.loads(cast_json)
        cast.sort(key=lambda x: x.get('order', 100))
        top_cast = cast[:max_actors]

        weighted_names = []
        for i, actor in enumerate(top_cast):
            name = actor['name']
            weight = 1 / (i + 1)
            repeats = max(1, int(weight * 5))
            weighted_names.extend([name] * repeats)

        return ' '.join(weighted_names)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return ""
