from recsys.RecommenderMetrics import RecommenderMetrics
import logging


class AlgorithmEvaluator:
    def __init__(self, algorithm, name, verbose=False):
        self._algorithm = algorithm
        self._name = name
        self._accuracy_metrics = {}
        self._top_n_metrics = {}

        self._logger = logging.getLogger(
            f"{self.__class__.__name__}({algorithm.__class__.__name__}, {name})"
        )
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def evaluate(
        self, evaluation_dataset, top_n_metrics=False, minimum_rating=0.4, n=10
    ):
        self._logger.info(
            f"Evaluating: {self.__class__.__name__}({self._algorithm.__class__.__name__}, {self._name})"
        )
        accuracy_results = self._evaluate_accuracy(evaluation_dataset)
        top_n_results = []

        if top_n_metrics:
            top_n_results = self._evaluate_top_n_metrics(
                evaluation_dataset, n, minimum_rating
            )

        self._logger.info("Evaluation complete")

        return accuracy_results + top_n_results

    def _evaluate_accuracy(self, evaluation_dataset):
        self._logger.info("Evaluating accuracy")
        self._algorithm.fit(evaluation_dataset.train_set)
        predictions = self._algorithm.test(evaluation_dataset.test_set)
        rmse = RecommenderMetrics.rmse(predictions)
        mae = RecommenderMetrics.mae(predictions)
        return [
            ("RMSE", rmse),
            ("MAE", mae),
        ]

    def _evaluate_top_n_metrics(self, evaluation_dataset, n, minimum_rating):
        self._logger.info("Evaluating top-N metrics with Leave One Out validation")

        self._algorithm.fit(evaluation_dataset.loo_train_set)
        loo_validation_set = self._algorithm.test(evaluation_dataset.loo_test_set)
        anti_test_predictions = self._algorithm.test(
            evaluation_dataset.loo_anti_test_set
        )
        top_n_predictions = RecommenderMetrics.get_top_n(
            anti_test_predictions, n, minimum_rating
        )

        top_n_results = []

        hit_rate = RecommenderMetrics.hit_rate(top_n_predictions, loo_validation_set)
        top_n_results.append(("HR", hit_rate))

        cumulative_hit_rate = RecommenderMetrics.cumulative_hit_rate(
            top_n_predictions, loo_validation_set
        )
        top_n_results.append(("cHR", cumulative_hit_rate))

        arhr = RecommenderMetrics.average_reciprocal_hit_rank(
            top_n_predictions, loo_validation_set
        )
        top_n_results.append(("ARHR", arhr))

        self._logger.info("Evaluating top-N metrics with full dataset")
        self._algorithm.fit(evaluation_dataset.full_train_set)
        anti_test_predictions = self._algorithm.test(
            evaluation_dataset.full_anti_test_set
        )
        top_n_predictions = RecommenderMetrics.get_top_n(
            anti_test_predictions, n, minimum_rating
        )

        coverage = RecommenderMetrics.user_coverage(
            top_n_predictions,
            evaluation_dataset.full_train_set.n_users,
            minimum_rating=minimum_rating,
        )
        top_n_results.append(("Coverage", coverage))

        diversity = RecommenderMetrics.diversity(
            top_n_predictions, evaluation_dataset.similarities
        )
        top_n_results.append(("Diversity", diversity))

        novelty = RecommenderMetrics.novelty(
            top_n_predictions, evaluation_dataset.popularity_rankings
        )
        top_n_results.append(("Novelty", novelty))

        return top_n_results

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
