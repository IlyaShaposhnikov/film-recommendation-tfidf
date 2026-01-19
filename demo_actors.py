"""
Учебные примеры и пояснения к системе рекомендаций фильмов
на основе схожести актерского состава с использованием TF-IDF.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Создаем папку для визуализаций
vis_path = Path('visualizations/actor_based_learning')
vis_path.mkdir(parents=True, exist_ok=True)


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


print("=" * 60)
print("УЧЕБНЫЕ ПРИМЕРЫ: РЕКОМЕНДАЦИИ ПО АКТЕРСКОГО СОСТАВУ")
print("=" * 60)

# Загрузка данных
print("\n1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("-" * 40)

df = pd.read_csv('data/tmdb_5000_credits.csv')
print(f"Загружено {len(df)} фильмов")
print(f"Колонки: {df.columns.tolist()}")

# Анализ структуры данных
first_movie = df.iloc[0]
print(f"\nПример данных для фильма '{first_movie['title']}':")

try:
    cast_data = json.loads(first_movie['cast'])
    print(f"Количество актеров: {len(cast_data)}")
    print("Первые 3 актера:")
    for i, actor in enumerate(cast_data[:3], 1):
        print(f"  {i}. {actor['name']} (порядок в титрах: {actor.get('order', 'N/A')})")
except ValueError:
    print("Ошибка при чтении данных об актерах")

# 2. Как работает функция извлечения актеров с весами
print("\n\n2. КАК РАБОТАЕТ ФУНКЦИЯ ВЕСОВОЙ ОБРАБОТКИ")
print("-" * 40)


def explain_weighted_actors():
    """Пояснение работы весовой функции."""
    example_cast = [
        {"name": "Actor1", "order": 0},
        {"name": "Actor2", "order": 1},
        {"name": "Actor3", "order": 2},
        {"name": "Actor4", "order": 3},
        {"name": "Actor5", "order": 4}
    ]

    print("Пример актерского состава (5 актеров с порядком в титрах):")
    for actor in example_cast:
        print(f"  {actor['name']} - order: {actor['order']}")

    print("\nВесовая функция: weight = 1 / (order + 1)")
    print("Количество повторений: repeats = max(1, int(weight * 5))")
    print("\nРасчет:")

    weighted_names = []
    for i, actor in enumerate(example_cast):
        order = actor['order']
        weight = 1 / (order + 1)
        repeats = max(1, int(weight * 5))

        print(f"  {actor['name']} (order={order}):")
        print(f"    weight = 1 / ({order} + 1) = {weight:.3f}")
        print(f"    repeats = max(1, int({weight:.3f} * 5)) = {repeats}")

        weighted_names.extend([actor['name']] * repeats)

    print(f"\nРезультирующая строка: {' '.join(weighted_names)}")
    print("\nИнтерпретация:")
    print("Главный актер (order=0) повторяется 5 раз")
    print("Второстепенные актеры повторяются меньше раз")
    print("Это дает им больший вес в TF-IDF модели")


explain_weighted_actors()

# 3. Анализ распределения количества актеров в фильмах
print("\n\n3. АНАЛИЗ РАСПРЕДЕЛЕНИЯ АКТЕРОВ В ФИЛЬМАХ")
print("-" * 40)


def count_actors(cast_json):
    """Подсчет количества актеров в фильме."""
    try:
        cast = json.loads(cast_json)
        return len(cast)
    except (json.JSONDecodeError, TypeError, KeyError):
        return 0


df['actor_count'] = df['cast'].apply(count_actors)

print(f"Среднее количество актеров на фильм: {df['actor_count'].mean():.1f}")
print(f"Минимальное количество актеров: {df['actor_count'].min()}")
print(f"Максимальное количество актеров: {df['actor_count'].max()}")

# Визуализация распределения количества актеров
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['actor_count'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Количество актеров в фильме')
plt.ylabel('Количество фильмов')
plt.title('Распределение количества актеров в фильмах')
plt.grid(True, alpha=0.3)

# Коробчатая диаграмма
plt.subplot(1, 2, 2)
plt.boxplot(df['actor_count'], vert=False, widths=0.7)
plt.xlabel('Количество актеров')
plt.title('Распределение количества актеров (коробчатая диаграмма)')
plt.yticks([])

plt.tight_layout()
plt.savefig(vis_path / 'actor_count_distribution.png', dpi=100)
plt.close()
print("График распределения сохранен")

# 4. Пример работы TF-IDF на простых данных
print("\n\n4. ПРИНЦИП РАБОТЫ TF-IDF НА ПРИМЕРЕ")
print("-" * 40)

# Создаем простой пример с тремя фильмами
example_films = [
    {"title": "Фильм A", "actors": "TomHanks TomHanks TomHanks MegRyan MegRyan"},
    {"title": "Фильм B", "actors": "TomHanks TomHanks MegRyan MegRyan BillyCrystal"},
    {"title": "Фильм C", "actors": "BillyCrystal BillyCrystal JuliaRoberts JuliaRoberts"}
]

example_df = pd.DataFrame(example_films)

print("Пример трех фильмов с актерами:")
for _, row in example_df.iterrows():
    print(f"  {row['title']}: {row['actors']}")

# Применяем TF-IDF
tfidf_example = TfidfVectorizer()
X_example = tfidf_example.fit_transform(example_df['actors'])

print("\nTF-IDF матрица:")
print("(строки: фильмы, столбцы: актеры, значения: TF-IDF вес)")
print(X_example.toarray())

feature_names = tfidf_example.get_feature_names_out()
print(f"\nПризнаки (актеры): {feature_names}")

# Вычисляем схожести
similarities = cosine_similarity(X_example)
print("\nМатрица схожести между фильмами:")
for i in range(len(example_films)):
    for j in range(len(example_films)):
        if i != j:
            print(f"  {example_films[i]['title']} -> {example_films[j]['title']}: {similarities[i][j]:.3f}")

print("\nИнтерпретация:")
print("Фильмы A и B имеют высокую схожесть (общие актеры: TomHanks, MegRyan)")
print("Фильм C имеет низкую схожесть с A и B (нет общих актеров)")

# 5. Анализ реальных данных
print("\n\n5. АНАЛИЗ РЕАЛЬНЫХ ДАННЫХ И ВИЗУАЛИЗАЦИЯ")
print("-" * 40)

# Подготовка данных для реального анализа
df['weighted_actors'] = df['cast'].apply(
    lambda x: extract_weighted_actors(x) if isinstance(x, str) else ""
)

# Создаем TF-IDF модель для небольшого набора данных для визуализации
sample_size = 500
sample_df = df.head(sample_size).copy()

tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(sample_df['weighted_actors'])

# Вычисляем схожести для пары известных фильмов
test_movies = ["Avatar", "Titanic", "The Dark Knight", "Pulp Fiction"]
similarity_results = {}

for movie in test_movies:
    if movie in sample_df['title'].values:
        idx = sample_df[sample_df['title'] == movie].index[0]
        movie_vector = X[idx]
        similarities = cosine_similarity(movie_vector, X).flatten()

        # Находим топ-5 похожих фильмов
        similar_indices = np.argsort(-similarities)[1:6]

        similarity_results[movie] = {
            "similarities": similarities,
            "similar_movies": sample_df.iloc[similar_indices]['title'].tolist(),
            "similar_scores": similarities[similar_indices].tolist()
        }

# Визуализация схожести
plt.figure(figsize=(15, 10))

for i, movie in enumerate(test_movies[:4], 1):
    if movie in similarity_results:
        plt.subplot(2, 2, i)

        # Получаем схожести для этого фильма
        similarities = similarity_results[movie]["similarities"]

        # Сортируем по убыванию
        sorted_indices = np.argsort(-similarities)
        sorted_similarities = similarities[sorted_indices]

        # Строим график
        plt.plot(sorted_similarities[:50], marker='o', markersize=3, alpha=0.7)
        plt.xlabel('Ранк похожести (0 = самый похожий)')
        plt.ylabel('Коэффициент схожести')
        plt.title(f'Схожесть фильмов с "{movie}"')
        plt.grid(True, alpha=0.3)

        # Добавляем аннотацию для топ-3
        top_3_movies = similarity_results[movie]["similar_movies"][:3]
        top_3_scores = similarity_results[movie]["similar_scores"][:3]

        for j in range(3):
            plt.annotate(
                f'{top_3_movies[j][:15]}...',
                xy=(j, top_3_scores[j]),
                xytext=(j, top_3_scores[j] + 0.05),
                ha='center', fontsize=8
            )

plt.tight_layout()
plt.savefig(vis_path / 'movie_similarity_patterns.png', dpi=100)
plt.close()
print("Графики схожести сохранены")

# 6. Анализ самых частых актеров
print("\n\n6. АНАЛИЗ САМЫХ ЧАСТЫХ АКТЕРОВ")
print("-" * 40)

# Извлекаем всех актеров из данных
all_actors = []
for cast_json in df['cast']:
    try:
        cast = json.loads(cast_json)
        for actor in cast[:10]:  # Берем только топ-10 актеров из каждого фильма
            all_actors.append(actor['name'])
    except (json.JSONDecodeError, TypeError, KeyError):
        continue

# Считаем частоту появления
actor_counts = pd.Series(all_actors).value_counts().head(15)

print("Топ-15 самых частых актеров (по количеству фильмов):")
for i, (actor, count) in enumerate(actor_counts.items(), 1):
    print(f"{i:2}. {actor:30} - {count} фильмов")

# Визуализация топ актеров
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(actor_counts)), actor_counts.values[::-1])
plt.yticks(range(len(actor_counts)), actor_counts.index[::-1])
plt.xlabel('Количество фильмов')
plt.title('Топ-15 самых частых актеров в базе данных')
plt.gca().invert_yaxis()

# Добавляем значения на график
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{int(width)}', ha='left', va='center')

plt.tight_layout()
plt.savefig(vis_path / 'top_15_actors.png', dpi=100)
plt.close()
print("График топ-15 актеров сохранен")

# 7. Как работает поиск по подстроке
print("\n\n7. ПРИНЦИП РАБОТЫ ПОИСКА ПО ПОДСТРОКЕ")
print("-" * 40)

search_examples = ["bat", "star", "love", "dark"]

print("Примеры поиска по подстроке:")
for search_term in search_examples:
    found_movies = []
    for title in df['title']:
        if search_term.lower() in title.lower():
            found_movies.append(title)
            if len(found_movies) >= 3:
                break

    if found_movies:
        print(f"\nПоиск '{search_term}':")
        for i, title in enumerate(found_movies, 1):
            print(f"  {i}. {title}")
    else:
        print(f"\nПоиск '{search_term}': не найдено")

# 8. Пример полного цикла рекомендаций
print("\n\n8. ПОЛНЫЙ ЦИКЛ РЕКОМЕНДАЦИЙ НА ПРИМЕРЕ")
print("-" * 40)


def demonstrate_recommendation_process(movie_title):
    """Демонстрация полного цикла рекомендаций для одного фильма."""
    print(f"\nДемонстрация для фильма: '{movie_title}'")

    if movie_title not in df['title'].values:
        print(f"  Фильм '{movie_title}' не найден в базе данных.")
        return

    # Подготовка данных
    df_sample = df.copy()
    df_sample['weighted_actors'] = df_sample['cast'].apply(
        lambda x: extract_weighted_actors(x) if isinstance(x, str) else ""
    )

    # TF-IDF
    tfidf_full = TfidfVectorizer(max_features=1000, stop_words='english')
    X_full = tfidf_full.fit_transform(df_sample['weighted_actors'])

    # Поиск индекса фильма
    movie_idx = df_sample[df_sample['title'] == movie_title].index[0]

    # Вычисление схожести
    movie_vector = X_full[movie_idx]
    similarities = cosine_similarity(movie_vector, X_full).flatten()

    # Топ-5 рекомендаций
    similar_indices = np.argsort(-similarities)[1:6]

    print(f"\n  Топ-5 рекомендаций для '{movie_title}':")
    for i, idx in enumerate(similar_indices, 1):
        similar_movie = df_sample.iloc[idx]['title']
        sim_score = similarities[idx]
        print(f"  {i}. {similar_movie} (схожесть: {sim_score:.3f})")

    # Анализ почему рекомендованы эти фильмы
    print("\n  Почему эти фильмы рекомендованы:")

    # Извлекаем актеров исходного фильма
    try:
        original_cast = json.loads(df_sample.iloc[movie_idx]['cast'])
        original_actors = [actor['name'] for actor in original_cast[:5]]
        print(f"    Главные актеры в '{movie_title}': {', '.join(original_actors)}")

        # Для каждого рекомендованного фильма находим общих актеров
        for i, idx in enumerate(similar_indices[:3], 1):
            similar_movie = df_sample.iloc[idx]['title']
            try:
                similar_cast = json.loads(df_sample.iloc[idx]['cast'])
                similar_actors = [actor['name'] for actor in similar_cast[:5]]

                # Находим общих актеров
                common_actors = set(original_actors).intersection(set(similar_actors))
                if common_actors:
                    print(f"    {i}. '{similar_movie}': общие актеры - {', '.join(common_actors)}")
                else:
                    print(f"    {i}. '{similar_movie}': общие актеры не найдены в топ-5")
            except (json.JSONDecodeError, TypeError, KeyError):
                print(f"    {i}. '{similar_movie}': ошибка при чтении данных об актерах")
    except (json.JSONDecodeError, TypeError, KeyError):
        print("    Ошибка при анализе актерского состава")


# Демонстрация для нескольких фильмов
demonstration_movies = ["The Dark Knight", "Avatar", "The Godfather"]
for movie in demonstration_movies:
    demonstrate_recommendation_process(movie)

print("\n" + "=" * 60)
print("ЗАКЛЮЧЕНИЕ")
print("=" * 60)
print("\nЭта система рекомендаций работает следующим образом:")
print("1. Извлекает актеров из JSON-формата и сортирует их по порядку в титрах")
print("2. Применяет весовую функцию для учета 'звездности' актеров")
print("3. Создает TF-IDF матрицу, где каждый фильм представлен взвешенным набором актеров")
print("4. Вычисляет косинусную схожесть между фильмами")
print("5. Рекомендует фильмы с наибольшей схожестью актерского состава")
print("\nПреимущества подхода:")
print("- Учитывает не только наличие актеров, но и их важность в фильме")
print("- Находит скрытые связи между фильмами через общих актеров")
print("- Быстрая работа даже с большими наборами данных")

print("\nВсе визуализации сохранены в папке 'visualizations/actor_based_learning/'")
print("\nАнализ завершен!")
