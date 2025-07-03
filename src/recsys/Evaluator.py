from recsys.EvaluationDataset import EvaluationDataset
from recsys.AlgorithmEvaluator import AlgorithmEvaluator
import logging


class Evaluator:
    algorithms: list[AlgorithmEvaluator]

    def __init__(self, dataset, rankings, verbose=False) -> None:
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.dataset = EvaluationDataset(dataset, rankings, verbose=verbose)
        self.algorithms = []

    def add_algorithm(self, algorithm, name):
        alg = AlgorithmEvaluator(algorithm, name, verbose=self._verbose)
        self.algorithms.append(alg)

    def evaluate(self, top_n_metrics=False, minimum_rating=0.0):
        results = []
        metrics = []
        for algorithm in self.algorithms:
            self._logger.info(f"Evaluating: {algorithm.name}")
            algorithm_results = algorithm.evaluate(
                evaluation_dataset=self.dataset,
                top_n_metrics=top_n_metrics,
                minimum_rating=minimum_rating,
            )
            metrics.append([result[0] for result in algorithm_results])
            results.append([result[1] for result in algorithm_results])

        names = [a.name for a in self.algorithms]
        return names, metrics[0], results

    def sample_top_n_recs(self, ml, test_subject=85, n=10):
        for algo in self.algorithms:
            self._logger.info(f"Using recommender: {algo.name}")

            self._logger.info("Training model")
            train_set = self.dataset.full_train_set
            algo.algorithm.fit(train_set)

            self._logger.info("Generate recommendations")
            test_set = self.dataset.get_anti_test_set_for_user(test_subject)

            predictions = algo.algorithm.test(test_set)

            recommendations = []

            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            samples = [
                (ml.get_movie_name(ratings[0]), ratings[1])
                for ratings in recommendations[:n]
            ]
            return samples
