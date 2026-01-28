"""
Консольный интерфейс для рекомендаций по актерам
"""
from core.actors_recommender import ActorsRecommender


def interactive_recommender():
    """Интерактивная система рекомендаций по актерам"""
    print("=" * 60)
    print("РЕКОМЕНДАЦИИ ПО СХОЖЕСТИ АКТЕРСКОГО СОСТАВА")
    print("=" * 60)

    recommender = ActorsRecommender()

    while True:
        print("\n" + "=" * 60)
        print("МЕНЮ:")
        print("1. Найти фильм и получить рекомендации")
        print("2. Примеры рекомендаций")
        print("3. Выход")

        choice = input("\nВыберите действие (1-3): ").strip()

        if choice == '1':
            search_term = input(
                "\nВведите часть названия фильма "
                "на английском (минимум 3 буквы): "
            ).strip()

            if len(search_term) < 3:
                print(
                    "Ошибка: для поиска необходимо ввести минимум 3 символа."
                )
                continue

            found_movies = recommender.search_movies(search_term)

            if not found_movies:
                print(
                    "Фильмы с названием, содержащим "
                    f"'{search_term}', не найдены."
                )
                continue

            print(f"\nНайдено фильмов: {len(found_movies)}")
            for i, title in enumerate(found_movies, 1):
                print(f"{i}. {title}")

            try:
                selection = input(
                    "\nВведите номер фильма (или 0 для нового поиска): "
                ).strip()

                if selection == '0':
                    continue

                selection = int(selection) - 1
                if 0 <= selection < len(found_movies):
                    selected_movie = found_movies[selection]

                    print(f"\nПоиск фильмов, похожих на '{selected_movie}'...")

                    requested_movie, recommendations = recommender.recommend(
                        selected_movie
                    )

                    if isinstance(recommendations, str):
                        print(recommendations)
                    else:
                        print(
                            f"\nТоп-{len(recommendations)} фильмов, "
                            "похожих по актерскому составу на "
                            f"'{requested_movie}':"
                        )
                        print("-" * 60)

                        for i, rec in enumerate(recommendations, 1):
                            print(
                                f"{i}. {rec['title']} (схожесть: "
                                f"{rec['similarity_score']})"
                            )
                else:
                    print("Некорректный номер фильма.")

            except (ValueError, IndexError):
                print("Пожалуйста, введите корректный номер.")

        elif choice == '2':
            examples = [
                "The Dark Knight",
                "Avatar",
                "The Shawshank Redemption",
                "Pulp Fiction"
            ]

            print("\nПримеры работы системы:")
            print("=" * 60)

            for example in examples:
                requested_movie, recommendations = recommender.recommend(
                    example, num_recommendations=3
                )

                if isinstance(recommendations, str):
                    print(f"\nДля '{example}': {recommendations}")
                else:
                    print(f"\nДля '{requested_movie}':")
                    for i, rec in enumerate(recommendations, 1):
                        print(
                            f"  {i}. {rec['title']} "
                            f"({rec['similarity_score']})"
                        )

            input("\nНажмите Enter для продолжения...")

        elif choice == '3':
            print("\nСпасибо за использование системы рекомендаций!")
            break

        else:
            print("Некорректный выбор. Пожалуйста, выберите 1, 2 или 3.")


if __name__ == "__main__":
    interactive_recommender()
