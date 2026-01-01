# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Загрузка данных из папки data
df = pd.read_csv('data/tmdb_5000_movies.csv')

print("Первые 5 строк датафрейма:")
print(df.head())

x = df.iloc[0]
print("\nПервая строка датафрейма:")
print(x)

print("\nЖанры первой строки (сырые данные):")
print(x['genres'])

print("\nКлючевые слова первой строки (сырые данные):")
print(x['keywords'])

# Преобразуем JSON-строку с жанрами в список словарей
j = json.loads(x['genres'])
print("\nРаспарсенные жанры (JSON):")
print(j)

# Преобразуем список жанров в строку с названиями жанров
# 1. Для каждого элемента jj в списке j берем значение по ключу 'name'
# 2. Разделяем имя жанра на слова (split()) и снова соединяем (join()) - это убирает лишние пробелы
# 3. Собираем все имена жанров в одну строку, разделенную пробелами
genres_str = ' '.join(' '.join(jj['name'].split()) for jj in j)
print("\nСтрока с жанрами первого фильма:")
print(genres_str)

# Функция для преобразования жанров и ключевых слов в единую строку
def genres_and_keywords_to_string(row):
    """
    Преобразует JSON-строки с жанрами и ключевыми словами в единую текстовую строку.
    
    Аргументы:
        row: строка датафрейма
    
    Возвращает:
        Строку, содержащую все жанры и ключевые слова, разделенные пробелами
    """
    # Парсим JSON-строку с жанрами
    genres = json.loads(row['genres'])
    # Преобразуем список жанров в строку (например, "Action Adventure ScienceFiction")
    genres = ' '.join(' '.join(j['name'].split()) for j in genres)
    
    # Парсим JSON-строку с ключевыми словами
    keywords = json.loads(row['keywords'])
    # Преобразуем список ключевых слов в строку
    keywords = ' '.join(' '.join(j['name'].split()) for j in keywords)
    
    # Возвращаем объединенную строку жанров и ключевых слов
    return "%s %s" % (genres, keywords)

# Создаем новую колонку 'string' в датафрейме, содержащую объединенные строки жанров и ключевых слов
# axis=1 означает, что функция применяется к каждой строке датафрейма
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

print("\nПервые 5 строк датафрейма с новой колонкой 'string':")
print(df[['original_title', 'string']].head())

print("\nПример строки для первого фильма:")
print(df.iloc[0]['string'])
