# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Создаем папку для визуализаций, если она не существует
vis_path = Path('visualizations')
vis_path.mkdir(exist_ok=True)

# Загрузка данных из папки data
print("Загрузка данных...")
df = pd.read_csv('data/tmdb_5000_movies.csv')

# Извлекаем год из даты релиза для удобства пользователя
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_year'] = df['release_year'].fillna(0).astype(int)


# Функция для преобразования жанров и ключевых слов в единую строку
def genres_and_keywords_to_string(row):
    """
    Преобразует JSON-строки с жанров и ключевых слов в единую текстовую строку.
    Функция склеивает наименования жанров и ключевых слов, состоящие из нескольких слов,
    в одно цельное наименование (например, "Science Fiction" становится "ScienceFiction").
    """
    genres = json.loads(row['genres'])
    genres = ' '.join(' '.join(j['name'].split()) for j in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(' '.join(j['name'].split()) for j in keywords)

    return "%s %s" % (genres, keywords)


# Создаем новую колонку 'string' в датафрейме
print("Обработка жанров и ключевых слов...")
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

# Создаем объект TF-IDF векторизатора с английскими стоп-словами
print("Создание TF-IDF матрицы...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X = tfidf.fit_transform(df['string'])

# Создаем отображение названия фильма -> индекс в датафрейме
movie2idx = pd.Series(df.index, index=df['title'])

print(f"\nДанные загружены: {len(df)} фильмов, {X.shape[1]} признаков")
print("="*60)

# Визуализация: распределение количества слов в описаниях фильмов
word_counts = df['string'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12, 6))
plt.hist(word_counts, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Количество слов в строке (жанры + ключевые слова)')
plt.ylabel('Количество фильмов')
plt.title('Распределение количества слов в описаниях фильмов')
plt.axvline(word_counts.mean(), color='red', linestyle='dashed', linewidth=2,
            label=f'Среднее: {word_counts.mean():.1f} слов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(vis_path / 'word_count_distribution.png', dpi=100)
plt.close()

# Визуализация: топ-20 самых важных слов после добавления стоп-слов
feature_names = tfidf.get_feature_names_out()
word_importances = X.sum(axis=0).A1
word_importance_df = pd.DataFrame({
    'word': feature_names,
    'importance': word_importances
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
bars = plt.barh(word_importance_df['word'], word_importance_df['importance'], color='green')
plt.xlabel('Сумма TF-IDF значений по всем фильмам')
plt.title('Топ-20 самых важных слов для рекомендаций (без стоп-слов)')
plt.gca().invert_yaxis()

for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', ha='left', va='center')

plt.tight_layout()
plt.savefig(vis_path / 'top_20_words_no_stopwords.png', dpi=100)
plt.close()

print("Визуализации сохранены в папке 'visualizations/'")


# Функция для поиска фильмов по подстроке
def search_movies(search_term, df, num_results=10):
    """
    Ищет фильмы по подстроке в названии.

    Аргументы:
        search_term: подстрока для поиска (минимум 3 символа)
        df: DataFrame с фильмами
        num_results: максимальное количество результатов для возврата

    Возвращает:
        DataFrame с найденными фильмами или сообщение об ошибке
    """
    if len(search_term) < 3:
        return "Для поиска необходимо ввести минимум 3 символа."

    # Ищем фильмы, содержащие подстроку в названии (без учета регистра)
    mask = df['title'].str.contains(search_term, case=False, na=False)
    results = df[mask][['title', 'release_year']].copy()

    if len(results) == 0:
        return f"Фильмы с названием, содержащим '{search_term}', не найдены."

    # Добавляем нумерацию для удобства выбора
    results.reset_index(drop=True, inplace=True)
    results.index = results.index + 1  # Начинаем с 1, а не с 0

    return results.head(num_results)


# Функция для получения рекомендаций
def recommend(title, df, X, movie2idx, num_recommendations=5):
    """
    Генерирует рекомендации фильмов на основе косинусной схожести TF-IDF векторов.
    """
    try:
        # Получаем индекс фильма в датафрейме
        idx = movie2idx[title]

        # Обрабатываем случай, если title соответствует нескольким фильмам
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Получаем TF-IDF вектор для запрошенного фильма
        query = X[idx]

        # Вычисляем косинусную схожесть со всеми фильмами
        scores = cosine_similarity(query, X)
        scores = scores.flatten()

        # Получаем индексы рекомендованных фильмов
        recommended_idx = (-scores).argsort()[1:num_recommendations+1]

        # Создаем DataFrame с рекомендациями
        recommendations = df.iloc[recommended_idx][['title', 'release_year']].copy()
        recommendations['similarity_score'] = scores[recommended_idx].round(3)

        # Добавляем информацию о запрошенном фильме
        requested_movie = df.iloc[idx][['title', 'release_year']]

        return requested_movie, recommendations.reset_index(drop=True)

    except KeyError:
        return None, f"Фильм '{title}' не найден в базе данных."
    except Exception as e:
        return None, f"Произошла ошибка: {e}"


# Интерактивный интерфейс для пользователя
def interactive_recommendation_system():
    """
    Основная функция для взаимодействия с пользователем.
    Позволяет искать фильмы и получать рекомендации.
    """
    print("\n" + "="*60)
    print("СИСТЕМА РЕКОМЕНДАЦИЙ ФИЛЬМОВ")
    print("="*60)

    while True:
        print("\nМЕНЮ:")
        print("1. Поиск фильмов для получения рекомендаций")
        print("2. Примеры работы системы")
        print("3. Выход")

        choice = input("\nВыберите действие (1-3): ").strip()

        if choice == '1':
            # Поиск фильмов
            search_term = input("\nВведите часть названия фильма (минимум 3 буквы): ").strip()

            if len(search_term) < 3:
                print("Ошибка: для поиска необходимо ввести минимум 3 символа.")
                continue

            results = search_movies(search_term, df)

            if isinstance(results, str):
                print(results)
                continue

            print(f"\nНайдено фильмов: {len(results)}")
            print("Результаты поиска:")
            print(results.to_string())

            if len(results) > 0:
                # Запрос точного названия для рекомендаций
                try:
                    selection = input("\nВведите номер фильма для получения рекомендаций (или 0 для нового поиска): ").strip()

                    if selection == '0':
                        continue

                    selection = int(selection)
                    if 1 <= selection <= len(results):
                        selected_movie = results.iloc[selection-1]['title']

                        print(f"\nВыбран фильм: {selected_movie}")
                        print("Формирование рекомендаций...")

                        requested_movie, recommendations = recommend(selected_movie, df, X, movie2idx)

                        if recommendations is None or isinstance(recommendations, str):
                            print(recommendations)
                        else:
                            print(f"\nРекомендации для фильма '{requested_movie['title']}' ({requested_movie['release_year']}):")
                            print("="*60)
                            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                                print(f"{i}. {row['title']} ({row['release_year']}) - схожесть: {row['similarity_score']}")
                    else:
                        print("Некорректный номер фильма.")
                except ValueError:
                    print("Пожалуйста, введите число.")

        elif choice == '2':
            # Примеры работы системы
            example_movies = ['Avatar', 'The Dark Knight', 'Toy Story', 'The Shawshank Redemption']

            print("\nПримеры работы системы рекомендаций:")
            print("="*60)

            for movie in example_movies:
                requested_movie, recommendations = recommend(movie, df, X, movie2idx)

                if recommendations is None or isinstance(recommendations, str):
                    print(f"\n{movie}: {recommendations}")
                else:
                    print(f"\nРекомендации для '{requested_movie['title']}' ({requested_movie['release_year']}):")
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        print(f"  {i}. {row['title']} ({row['release_year']})")

            input("\nНажмите Enter для продолжения...")

        elif choice == '3':
            print("\nСпасибо за использование системы рекомендаций фильмов!")
            break

        else:
            print("Некорректный выбор. Пожалуйста, выберите 1, 2 или 3.")


# Основной блок выполнения
if __name__ == "__main__":
    try:
        interactive_recommendation_system()
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем.")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")
