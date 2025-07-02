from recsys.EvaluationDataset import EvaluationDataset
from recsys.AlgorithmEvaluator import AlgorithmEvaluator


class Evaluator:
    algorithms: list[AlgorithmEvaluator] = []

    def __init__(self, dataset, rankings):
        self.dataset = EvaluationDataset(dataset, rankings)

    def add_algorithm(self, algorithm, name):
        alg = AlgorithmEvaluator(algorithm, name)
        self.algorithms.append(alg)

    def evaluate(self, top_n_metrics=False, minimum_rating=0.0):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.name, "...")
            results[algorithm.name] = algorithm.evaluate(
                self.dataset, top_n_metrics, minimum_rating
            )

        # Print results
        print("\n")

        if top_n_metrics:
            print(
                "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm",
                    "RMSE",
                    "MAE",
                    "HR",
                    "cHR",
                    "ARHR",
                    "Coverage",
                    "Diversity",
                    "Novelty",
                )
            )
            for name, metrics in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name,
                        metrics["RMSE"],
                        metrics["MAE"],
                        metrics["HR"],
                        metrics["cHR"],
                        metrics["ARHR"],
                        metrics["Coverage"],
                        metrics["Diversity"],
                        metrics["Novelty"],
                    )
                )
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for name, metrics in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"]
                    )
                )

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if top_n_metrics:
            print(
                "HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better."
            )
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better."
            )
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better."
            )
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better."
            )
            print(
                "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations"
            )
            print("           for a given user. Higher means more diverse.")
            print(
                "Novelty:   Average popularity rank of recommended items. Higher means more novel."
            )

    def sample_top_n_recs(self, ml, test_subject=85, k=10):
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.name)

            print("\nBuilding recommendation model...")
            train_set = self.dataset.GetFullTrainSet()
            algo.algorithm.fit(train_set)

            print("Computing recommendations...")
            test_set = self.dataset.GetAntiTestSetForUser(test_subject)

            predictions = algo.algorithm.test(test_set)

            recommendations = []

            print("\nWe recommend:")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ml.get_movie_name(ratings[0]), ratings[1])
