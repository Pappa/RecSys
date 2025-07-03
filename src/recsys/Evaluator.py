from recsys.EvaluationDataset import EvaluationDataset
from recsys.AlgorithmEvaluator import AlgorithmEvaluator
import logging


class Evaluator:
    algorithms: list[AlgorithmEvaluator] = []

    def __init__(self, dataset, rankings, verbose=False) -> None:
        self.dataset = EvaluationDataset(dataset, rankings)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def add_algorithm(self, algorithm, name):
        alg = AlgorithmEvaluator(algorithm, name)
        self.algorithms.append(alg)

    def evaluate(self, top_n_metrics=False, minimum_rating=0.0):
        results = {}
        for algorithm in self.algorithms:
            self._logger.info(f"Evaluating: {algorithm.name}")
            results[algorithm.name] = algorithm.evaluate(
                self.dataset, top_n_metrics, minimum_rating
            )

        if top_n_metrics:
            self._logger.info(f"Top-N metrics: {algorithm.name}")
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

    def sample_top_n_recs(self, ml, test_subject=85, k=10):
        for algo in self.algorithms:
            self._logger.info(f"Using recommender: {algo.name}")

            self._logger.info("Training model")
            train_set = self.dataset.full_train_set
            algo.algorithm.fit(train_set)

            self._logger.info("Computing recommendations")
            test_set = self.dataset.get_anti_test_set_for_user(test_subject)

            predictions = algo.algorithm.test(test_set)

            recommendations = []

            print("\nWe recommend:")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ml.get_movie_name(ratings[0]), ratings[1])
