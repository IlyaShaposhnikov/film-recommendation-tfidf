"""
Рекомендации фильмов по схожести актерского состава
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.data_loader import load_credits_data, extract_weighted_actors


class ActorsRecommender:
    def __init__(self):
        """Инициализация рекомендательной системы по актерам"""
        print("Загрузка данных для рекомендаций по актерам...")
        self.df = load_credits_data()

        print("Подготовка данных об актерах...")
        self.df['weighted_actors'] = self.df['cast'].apply(
            extract_weighted_actors
        )

        print("Создание модели схожести...")
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.df['weighted_actors']
        )

        self.movie_titles = self.df['title'].tolist()
        self.movie2idx = {
            title: idx for idx, title in enumerate(self.movie_titles)
        }
        print(f"Данные загружены: {len(self.df)} фильмов")

    def search_movies(self, search_term, num_results=10):
        """Поиск фильмов по подстроке в названии"""
        if len(search_term) < 3:
            return []

        results = []
        for title in self.movie_titles:
            if search_term.lower() in title.lower():
                results.append(title)
                if len(results) >= num_results:
                    break

        return results

    def recommend(self, title, num_recommendations=10):
        """Получение рекомендаций для указанного фильма"""
        if title not in self.movie2idx:
            return None, f"Фильм '{title}' не найден в базе данных."

        movie_idx = self.movie2idx[title]
        movie_vector = self.tfidf_matrix[movie_idx]

        similarities = cosine_similarity(
            movie_vector, self.tfidf_matrix
        ).flatten()
        similar_indices = np.argsort(-similarities)[1:num_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            sim_score = similarities[idx]
            if sim_score > 0.01:
                recommendations.append({
                    'title': self.movie_titles[idx],
                    'similarity_score': round(sim_score, 3)
                })

        return title, recommendations
