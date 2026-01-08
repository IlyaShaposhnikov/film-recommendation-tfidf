# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Загрузка данных из папки data
df = pd.read_csv('data/tmdb_5000_movies.csv')


# Функция для преобразования жанров и ключевых слов в единую строку
def genres_and_keywords_to_string(row):
    """
    Преобразует JSON-строки с жанрами и ключевых слов в единую
    текстовую строку.
    Функция склеивает наименования жанров и ключевых слов,
    состоящие из нескольких слов,
    в одно цельное наименование
    (например, "Science Fiction" становится "ScienceFiction").

    Аргументы:
        row: строка датафрейма

    Возвращает:
        Строку, содержащую все жанры и ключевые слова, разделенные пробелами
    """
    # Парсим JSON-строку с жанрами
    genres = json.loads(row['genres'])
    # Преобразуем список жанров в строку, удаляя пробелы внутри названий жанров
    genres = ' '.join(' '.join(j['name'].split()) for j in genres)

    # Парсим JSON-строку с ключевыми словами
    keywords = json.loads(row['keywords'])
    # Преобразуем список ключевых слов в строку, удаляя пробелы внутри названий
    keywords = ' '.join(' '.join(j['name'].split()) for j in keywords)

    # Возвращаем объединенную строку жанров и ключевых слов
    return "%s %s" % (genres, keywords)


# Создаем новую колонку 'string' в датафрейме
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

# Создаем объект TF-IDF векторизатора
# max_features=2000 ограничивает количество признаков
# до 2000 самых частотных слов
tfidf = TfidfVectorizer(max_features=2000)

# Преобразуем текстовые строки в матрицу TF-IDF признаков
# fit_transform() обучает векторизатор на данных и сразу преобразует их
x = tfidf.fit_transform(df['string'])

print("TF-IDF матрица создана:")
print(f"Размерность матрицы: {x.shape}")
print(f"Количество фильмов: {x.shape[0]}")
print(f"Количество признаков (слов): {x.shape[1]}")
print("\nТип объекта матрицы:", type(x))
print("\nПример первых 5 строк и 10 признаков:")
print(x[:5, :10].toarray())

# Создаем отображение названия фильма -> индекс в датафрейме
# pd.Series создает серию, где индексы - названия фильмов,
# значения - индексы строк
movie2idx = pd.Series(df.index, index=df['title'])

print("\nОтображение movie2idx создано:")
print(f"Количество фильмов в отображении: {len(movie2idx)}")
print("\nПримеры отображения (первые 5 записей):")
print(movie2idx.head())

# Тестируем отображение для конкретного фильма
idx = movie2idx['Scream 3']
print(f"\nИндекс фильма 'Scream 3': {idx}")
print("Информация о фильме 'Scream 3':")
print(df.loc[idx, ['title', 'genres', 'keywords']])

# Визуализация: топ-20 самых частых слов в TF-IDF матрице
# Получаем имена признаков (слова)
feature_names = tfidf.get_feature_names_out()

# Вычисляем сумму TF-IDF значений по всем документам для каждого слова
word_importances = x.sum(axis=0).A1  # A1 преобразует в одномерный массив
word_importance_df = pd.DataFrame({
    'word': feature_names,
    'importance': word_importances
}).sort_values('importance', ascending=False).head(20)

print("\nТоп-20 самых важных слов (по сумме TF-IDF):")
print(word_importance_df)

# Создаем график
plt.figure(figsize=(12, 8))
bars = plt.barh(word_importance_df['word'], word_importance_df['importance'])
plt.xlabel('Сумма TF-IDF значений по всем фильмам')
plt.title('Топ-20 самых важных слов для рекомендаций фильмов')
plt.gca().invert_yaxis()  # чтобы самое важное слово было сверху

# Добавляем значения на график
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('top_20_words_tfidf.png', dpi=100)
print("\nГрафик сохранен как 'top_20_words_tfidf.png'")

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
plt.savefig('word_count_distribution.png', dpi=100)
print("График распределения сохранен как 'word_count_distribution.png'")

print("\nПодготовка данных завершена!")
print("Следующий шаг - вычисление схожести между фильмами.")
