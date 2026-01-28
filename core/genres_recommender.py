"""
Рекомендации фильмов по схожести жанров и ключевых слов
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path

from core.data_loader import load_movies_data, extract_genres_and_keywords


class GenresRecommender:
    def __init__(self):
        """Инициализация рекомендательной системы по жанрам"""
        print("Загрузка данных для рекомендаций по жанрам...")
        self.df = load_movies_data()
        self.df['string'] = self.df.apply(extract_genres_and_keywords, axis=1)

        print("Создание TF-IDF матрицы...")
        self.tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
        self.X = self.tfidf.fit_transform(self.df['string'])

        self.movie2idx = pd.Series(self.df.index, index=self.df['title'])
        print(f"Данные загружены: {len(self.df)} фильмов")

    def search_movies(self, search_term, num_results=10):
        """Поиск фильмов по подстроке в названии"""
        if len(search_term) < 3:
            return []

        mask = self.df['title'].str.contains(search_term, case=False, na=False)
        results = self.df[mask][['title', 'release_year']].copy()

        if len(results) == 0:
            return []

        results.reset_index(drop=True, inplace=True)
        results.index = results.index + 1
        return results.head(num_results)

    def recommend(self, title, num_recommendations=5):
        """Получение рекомендаций для указанного фильма"""
        try:
            idx = self.movie2idx[title]

            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

            query = self.X[idx]
            scores = cosine_similarity(query, self.X).flatten()
            recommended_idx = (-scores).argsort()[1:num_recommendations+1]

            recommendations = self.df.iloc[recommended_idx][
                ['title', 'release_year']
            ].copy()
            recommendations['similarity_score'] = (
                scores[recommended_idx].round(3)
            )

            requested_movie = self.df.iloc[idx][['title', 'release_year']]

            return requested_movie, recommendations.reset_index(drop=True)
        except KeyError:
            return None, f"Фильм '{title}' не найден в базе данных."
        except Exception as e:
            return None, f"Произошла ошибка: {e}"

    def create_visualizations(self):
        """Создание визуализаций (для консольного интерфейса)"""
        vis_path = Path('visualizations')
        vis_path.mkdir(exist_ok=True)

        # Распределение количества слов
        word_counts = self.df['string'].apply(lambda x: len(x.split()))
        plt.figure(figsize=(12, 6))
        plt.hist(word_counts, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Количество слов в строке')
        plt.ylabel('Количество фильмов')
        plt.title('Распределение количества слов в описаниях фильмов')
        plt.axvline(
            word_counts.mean(), color='red',
            linestyle='dashed', linewidth=2,
            label=f'Среднее: {word_counts.mean():.1f} слов'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_path / 'word_count_distribution.png', dpi=100)
        plt.close()

        # Топ-20 важных слов
        feature_names = self.tfidf.get_feature_names_out()
        word_importances = self.X.sum(axis=0).A1
        word_importance_df = pd.DataFrame({
            'word': feature_names,
            'importance': word_importances
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            word_importance_df['word'],
            word_importance_df['importance'],
            color='green'
        )
        plt.xlabel('Сумма TF-IDF значений')
        plt.title('Топ-20 самых важных слов для рекомендаций')
        plt.gca().invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                     f'{width:.2f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig(vis_path / 'top_20_words_no_stopwords.png', dpi=100)
        plt.close()

        print("Визуализации сохранены в папке 'visualizations/'")
