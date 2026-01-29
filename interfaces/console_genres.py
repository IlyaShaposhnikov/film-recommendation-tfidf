"""
Консольный интерфейс для рекомендаций по жанрам
(Содержит визуализации)
"""
import sys
import os

# Добавляем корневую папку проекта в путь Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.genres_recommender import GenresRecommender


def interactive_recommendation_system():
    """Интерактивная система рекомендаций по жанрам"""
    print("\n" + "="*60)
    print("СИСТЕМА РЕКОМЕНДАЦИЙ ФИЛЬМОВ ПО ЖАНРАМ")
    print("="*60)

    recommender = GenresRecommender()
    recommender.create_visualizations()

    while True:
        print("\nМЕНЮ:")
        print("1. Поиск фильмов для получения рекомендаций")
        print("2. Примеры работы системы")
        print("3. Выход")

        choice = input("\nВыберите действие (1-3): ").strip()

        if choice == '1':
            search_term = input(
                "\nВведите часть названия фильма на английском "
                "(минимум 3 буквы): "
            ).strip()

            if len(search_term) < 3:
                print(
                    "Ошибка: для поиска необходимо ввести минимум 3 символа."
                )
                continue

            results = recommender.search_movies(search_term)

            if len(results) == 0:
                print(
                    f"Фильмы с названием, содержащим '{search_term}', "
                    "не найдены."
                )
                continue

            print(f"\nНайдено фильмов: {len(results)}")
            print("Результаты поиска:")
            print(results.to_string())

            try:
                selection = input(
                    "\nВведите номер фильма для получения рекомендаций "
                    "(или 0 для нового поиска): "
                ).strip()

                if selection == '0':
                    continue

                selection = int(selection)
                if 1 <= selection <= len(results):
                    selected_movie = results.iloc[selection-1]['title']

                    print(f"\nВыбран фильм: {selected_movie}")
                    print("Формирование рекомендаций...")

                    requested_movie, recommendations = recommender.recommend(
                        selected_movie
                    )

                    if recommendations is None or isinstance(
                        recommendations, str
                    ):
                        print(recommendations)
                    else:
                        print(
                            "\nРекомендации для фильма "
                            f"'{requested_movie['title']}' "
                            f"({requested_movie['release_year']}):"
                        )
                        print("="*60)
                        for i, (_, row) in enumerate(
                            recommendations.iterrows(), 1
                        ):
                            print(
                                f"{i}. {row['title']} "
                                f"({row['release_year']}) - схожесть: "
                                f"{row['similarity_score']}"
                            )
                else:
                    print("Некорректный номер фильма.")
            except ValueError:
                print("Пожалуйста, введите число.")

        elif choice == '2':
            example_movies = [
                'Avatar',
                'The Dark Knight',
                'Toy Story',
                'The Shawshank Redemption'
            ]

            print("\nПримеры работы системы рекомендаций:")
            print("="*60)

            for movie in example_movies:
                requested_movie, recommendations = recommender.recommend(movie)

                if recommendations is None or isinstance(recommendations, str):
                    print(f"\n{movie}: {recommendations}")
                else:
                    print(
                        f"\nРекомендации для '{requested_movie['title']}' "
                        f"({requested_movie['release_year']}):"
                    )
                    for i, (_, row) in enumerate(
                        recommendations.iterrows(), 1
                    ):
                        print(f"  {i}. {row['title']} ({row['release_year']})")

            input("\nНажмите Enter для продолжения...")

        elif choice == '3':
            print("\nСпасибо за использование системы рекомендаций фильмов!")
            break

        else:
            print("Некорректный выбор. Пожалуйста, выберите 1, 2 или 3.")


if __name__ == "__main__":
    try:
        interactive_recommendation_system()
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем.")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")
