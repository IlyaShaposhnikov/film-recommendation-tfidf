"""
Система рекомендаций фильмов по схожести актерского состава.
Учитывает порядок в титрах через простую весовую функцию.
"""

import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_prepare_data():
    """Загрузка и подготовка данных."""
    df = pd.read_csv('data/tmdb_5000_credits.csv')
    return df


def extract_weighted_actors(cast_json, max_actors=10):
    """
    Извлекает актеров с весами на основе порядка в титрах.
    Простая формула: weight = 1 / (order + 1)
    """
    try:
        cast = json.loads(cast_json)
        # Сортируем по order и берем первых max_actors
        cast.sort(key=lambda x: x.get('order', 100))
        top_cast = cast[:max_actors]

        # Создаем строку с повторением имен пропорционально весу
        weighted_names = []
        for i, actor in enumerate(top_cast):
            name = actor['name']
            weight = 1 / (i + 1)  # Простая весовая функция
            # Повторяем имя пропорционально весу
            repeats = max(1, int(weight * 5))
            weighted_names.extend([name] * repeats)

        return ' '.join(weighted_names)
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return ""


def interactive_recommender():
    """Интерактивная система рекомендаций."""
    print("=" * 60)
    print("РЕКОМЕНДАЦИИ ПО СХОЖЕСТИ АКТЕРСКОГО СОСТАВА")
    print("=" * 60)

    # Загрузка данных
    print("\nЗагрузка данных...")
    df = load_and_prepare_data()

    # Подготовка данных для TF-IDF
    print("Подготовка данных об актерах...")
    df['weighted_actors'] = df['cast'].apply(extract_weighted_actors)

    # Создание TF-IDF матрицы
    print("Создание модели схожести...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['weighted_actors'])

    # Создание индекса для поиска
    movie_titles = df['title'].tolist()
    movie2idx = {title: idx for idx, title in enumerate(movie_titles)}

    # Функция поиска фильмов по подстроке
    def search_movies(search_term, min_length=3, max_results=10):
        """Поиск фильмов по подстроке в названии."""
        if len(search_term) < min_length:
            return []

        results = []
        for title in movie_titles:
            if search_term.lower() in title.lower():
                results.append(title)
                if len(results) >= max_results:
                    break

        return results

    # Основной цикл программы
    while True:
        print("\n" + "=" * 60)
        print("МЕНЮ:")
        print("1. Найти фильм и получить рекомендации")
        print("2. Примеры рекомендаций")
        print("3. Выход")

        choice = input("\nВыберите действие (1-3): ").strip()

        if choice == '1':
            # Поиск фильма
            search_term = input("\nВведите часть названия фильма на английском (минимум 3 буквы): ").strip()

            if len(search_term) < 3:
                print("Ошибка: для поиска необходимо ввести минимум 3 символа.")
                continue

            found_movies = search_movies(search_term)

            if not found_movies:
                print(f"Фильмы с названием, содержащим '{search_term}', не найдены.")
                continue

            print(f"\nНайдено фильмов: {len(found_movies)}")
            for i, title in enumerate(found_movies, 1):
                print(f"{i}. {title}")

            # Выбор фильма для рекомендаций
            try:
                selection = input("\nВведите номер фильма (или 0 для нового поиска): ").strip()

                if selection == '0':
                    continue

                selection = int(selection) - 1
                if 0 <= selection < len(found_movies):
                    selected_movie = found_movies[selection]
                    movie_idx = movie2idx[selected_movie]

                    # Вычисляем схожесть
                    print(f"\nПоиск фильмов, похожих на '{selected_movie}'...")

                    # Получаем вектор для выбранного фильма
                    movie_vector = tfidf_matrix[movie_idx]

                    # Вычисляем схожесть со всеми фильмами
                    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()

                    # Получаем индексы самых похожих фильмов (исключая сам фильм)
                    similar_indices = np.argsort(-similarities)[1:11]  # Топ-10

                    print(f"\nТоп-10 фильмов, похожих по актерскому составу на '{selected_movie}':")
                    print("-" * 60)

                    for i, idx in enumerate(similar_indices, 1):
                        sim_score = similarities[idx]
                        if sim_score > 0.01:  # Минимальный порог схожести
                            print(f"{i}. {movie_titles[idx]} (схожесть: {sim_score:.3f})")
                        else:
                            print(f"{i}. {movie_titles[idx]} (очень низкая схожесть)")

                else:
                    print("Некорректный номер фильма.")

            except (ValueError, IndexError):
                print("Пожалуйста, введите корректный номер.")

        elif choice == '2':
            # Примеры рекомендаций
            examples = [
                "The Dark Knight",
                "Avatar",
                "The Shawshank Redemption",
                "Pulp Fiction"
            ]

            print("\nПримеры работы системы:")
            print("=" * 60)

            for example in examples:
                if example in movie2idx:
                    movie_idx = movie2idx[example]
                    movie_vector = tfidf_matrix[movie_idx]
                    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
                    similar_indices = np.argsort(-similarities)[1:4]  # Топ-3

                    print(f"\nДля '{example}':")
                    for i, idx in enumerate(similar_indices, 1):
                        print(f"  {i}. {movie_titles[idx]} ({similarities[idx]:.3f})")

            input("\nНажмите Enter для продолжения...")

        elif choice == '3':
            print("\nСпасибо за использование системы рекомендаций!")
            break

        else:
            print("Некорректный выбор. Пожалуйста, выберите 1, 2 или 3.")


if __name__ == "__main__":
    interactive_recommender()
