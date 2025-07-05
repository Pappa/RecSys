from recsys.EvaluationDataset import EvaluationDataset
from recsys.AlgorithmEvaluator import AlgorithmEvaluator
import logging


class Evaluator:
    _verbose: bool
    _logger: logging.Logger
    _algorithms: list[AlgorithmEvaluator]
    _dataset: EvaluationDataset

    def __init__(self, dataset, rankings, verbose=False) -> None:
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._dataset = EvaluationDataset(dataset, rankings, verbose=verbose)
        self._algorithms = []

    def add_algorithm(self, algorithm, name):
        alg = AlgorithmEvaluator(algorithm, name, verbose=self._verbose)
        self._algorithms.append(alg)

    def evaluate(self, top_n_metrics=False, minimum_rating=0.0):
        names, metrics, results = [], [], []
        
        for algorithm in self._algorithms:
            self._logger.info(f"Evaluating: {algorithm.name}")
            algorithm_metrics, algorithm_results = algorithm.evaluate(
                evaluation_dataset=self._dataset,
                top_n_metrics=top_n_metrics,
                minimum_rating=minimum_rating,
            )

            names.append(algorithm.name)
            metrics.append(algorithm_metrics)
            results.append(algorithm_results)

        return names, metrics[0], results

    def sample_top_n_recs(self, lens, test_uid=85, n=10):
        for algorithm in self._algorithms:
            self._logger.info(f"Using recommender: {algorithm.name}")

            self._logger.info("Training model")
            trainset = self._dataset.full_trainset
            algorithm.algorithm.fit(trainset)

            self._logger.info("Generate recommendations")
            testset = self._dataset.get_anti_testset_for_user(test_uid)

            predictions = algorithm.algorithm.test(testset)

            recommendations = []

            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            samples = [
                (lens.get_movie_name(ratings[0]), ratings[1])
                for ratings in recommendations[:n]
            ]
            return samples
