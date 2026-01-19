"""
Файл с учебными примерами и пояснениями к проекту рекомендаций фильмов.
Здесь собраны вспомогательные функции и примеры, которые помогают понять
работу алгоритма рекомендаций на основе TF-IDF и косинусной схожести.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Создаем папку для визуализаций обучения
learning_vis_path = Path('visualizations/learning')
learning_vis_path.mkdir(parents=True, exist_ok=True)

# Загрузка данных
df = pd.read_csv('data/tmdb_5000_movies.csv')

# 1. Пример: как выглядят исходные данные в колонках genres и keywords
print("="*60)
print("ПРИМЕР 1: Структура исходных данных")
print("="*60)

first_movie = df.iloc[0]
print(f"Фильм: {first_movie['title']}")
print("\nСырые данные жанров (JSON строка):")
print(first_movie['genres'])
print("\nСырые данные ключевых слов (JSON строка):")
print(first_movie['keywords'])

# Парсим JSON
genres_json = json.loads(first_movie['genres'])
keywords_json = json.loads(first_movie['keywords'])

print("\nПарсированные жанры (первые 3):")
for i, genre in enumerate(genres_json[:3], 1):
    print(f"  {i}. ID: {genre['id']}, Название: {genre['name']}")

print("\nПарсированные ключевые слова (первые 3):")
for i, keyword in enumerate(keywords_json[:3], 1):
    print(f"  {i}. ID: {keyword['id']}, Название: {keyword['name']}")

# Визуализация: распределение количества жанров у фильмов
df['genres_count'] = df['genres'].apply(lambda x: len(json.loads(x)))
plt.figure(figsize=(10, 6))
plt.hist(df['genres_count'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Количество жанров у фильма')
plt.ylabel('Количество фильмов')
plt.title('Распределение количества жанров у фильмов')
plt.grid(True, alpha=0.3)
plt.savefig(learning_vis_path / 'genres_count_distribution.png', dpi=100)
plt.close()
print("\nГрафик распределения количества жанров сохранен")

# 2. Пример: как работает функция преобразования в строку
print("\n" + "="*60)
print("ПРИМЕР 2: Преобразование жанров и ключевых слов в строку")
print("="*60)


def genres_and_keywords_to_string_simple(row):
    """Упрощенная версия функции для демонстрации."""
    genres = json.loads(row['genres'])
    genres_str = ' '.join(' '.join(j['name'].split()) for j in genres)
    return genres_str


# Тестируем на нескольких фильмах
test_indices = [0, 10, 100]
for idx in test_indices:
    movie = df.iloc[idx]
    original_genres = json.loads(movie['genres'])
    genre_names = [g['name'] for g in original_genres]
    processed_string = genres_and_keywords_to_string_simple(movie)

    print(f"\nФильм: {movie['title']}")
    print(f"  Оригинальные жанры: {', '.join(genre_names)}")
    print(f"  После обработки: {processed_string}")
    print(f"  Изменения: {'Science Fiction' in ' '.join(genre_names) and 'ScienceFiction' in processed_string}")

# 3. Пример: как работает TF-IDF
print("\n" + "="*60)
print("ПРИМЕР 3: Принцип работы TF-IDF")
print("="*60)

from sklearn.feature_extraction.text import TfidfVectorizer

# Упрощенный пример с небольшим набором документов
documents = [
    "action adventure sciencefiction",
    "comedy romance drama",
    "action thriller mystery",
    "comedy action adventure"
]

simple_tfidf = TfidfVectorizer()
simple_X = simple_tfidf.fit_transform(documents)

print("Документы:")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc}")

print("\nСловарь (признаки):")
print(simple_tfidf.get_feature_names_out())

print("\nTF-IDF матрица (разреженная, показываем как плотную):")
print(simple_X.toarray())

print("\nИнтерпретация:")
print("Каждая строка - документ (фильм), каждый столбец - слово")
print("Значения показывают важность слова для документа")

# 4. Пример: как работает косинусная схожесть
print("\n" + "="*60)
print("ПРИМЕР 4: Косинусная схожесть")
print("="*60)

from sklearn.metrics.pairwise import cosine_similarity

# Берем два первых документа
doc1 = simple_X[0:1]
doc2 = simple_X[1:2]

similarity = cosine_similarity(doc1, doc2)
print(f"Документ 1: {documents[0]}")
print(f"Документ 2: {documents[1]}")
print(f"Косинусная схожесть: {similarity[0][0]:.3f}")
print("\nОбъяснение:")
print("Косинусная схожесть измеряет угол между векторами")
print("1.0 - векторы сонаправлены (идеальная схожесть)")
print("0.0 - векторы перпендикулярны (нет схожести)")
print("-1.0 - векторы противоположно направлены")

# Визуализация: пример векторов в 2D
plt.figure(figsize=(8, 8))
plt.arrow(0, 0, 0.8, 0.6, head_width=0.05, head_length=0.05, fc='blue', ec='blue', label='Вектор 1')
plt.arrow(0, 0, 0.3, 0.9, head_width=0.05, head_length=0.05, fc='red', ec='red', label='Вектор 2')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Ось X (признак 1)')
plt.ylabel('Ось Y (признак 2)')
plt.title('Визуализация косинусной схожести')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', alpha=0.3)
plt.axvline(x=0, color='k', alpha=0.3)
plt.savefig(learning_vis_path / 'cosine_similarity_visualization.png', dpi=100)
plt.close()
print("\nГрафик визуализации косинусной схожести сохранен")

# 5. Пример: влияние стоп-слов
print("\n" + "="*60)
print("ПРИМЕР 5: Влияние стоп-слов на рекомендации")
print("="*60)

documents_with_stopwords = [
    "the action movie with adventure and science fiction",
    "a comedy film about romance and drama",
    "an action thriller with mystery elements",
    "the comedy action adventure movie"
]

# Без стоп-слов
tfidf_no_stop = TfidfVectorizer(stop_words='english')
X_no_stop = tfidf_no_stop.fit_transform(documents_with_stopwords)

# Со стоп-словами
tfidf_with_stop = TfidfVectorizer()
X_with_stop = tfidf_with_stop.fit_transform(documents_with_stopwords)

print("Без стоп-слов (словарь):")
print(tfidf_no_stop.get_feature_names_out())

print("\nСо стоп-словами (первые 10 слов словаря):")
print(tfidf_with_stop.get_feature_names_out()[:10])

print("\nВывод: стоп-слова (the, a, an, with, about) удаляются,")
print("оставляя только значимые слова для определения тематики.")

print("\n" + "="*60)
print("ЗАКЛЮЧЕНИЕ")
print("="*60)
print("Эта система рекомендаций работает следующим образом:")
print("1. Извлекает жанры и ключевые слова из JSON-формата")
print("2. Преобразует их в текстовые строки")
print("3. Создает TF-IDF матрицу (оценивает важность слов)")
print("4. Вычисляет косинусную схожесть между фильмами")
print("5. Рекомендует фильмы с наибольшей схожестью")
print("\nСистема игнорирует стоп-слова и учитывает только смысловые слова.")
