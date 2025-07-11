from recsys.RecommenderMetrics import RecommenderMetrics, HitRateMetrics
import logging


class AlgorithmEvaluator:
    def __init__(self, algorithm, name, verbose=False):
        self._algorithm = algorithm
        self._name = name

        self._logger = logging.getLogger(
            f"{self.__class__.__name__}({algorithm.__class__.__name__}, {name})"
        )
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def evaluate(
        self, evaluation_dataset, top_n_metrics=False, minimum_rating=1e-5, coverage_threshold=1e-5, n=10
    ):
        self._logger.info(
            f"Evaluating: {self.__class__.__name__}({self._algorithm.__class__.__name__}, {self._name})"
        )
        accuracy_metrics, accuracy_results = self._evaluate_accuracy(evaluation_dataset)
        top_n_metrics_loo, top_n_results_loo = [], []
        top_n_metrics_full, top_n_results_full = [], []

        if top_n_metrics:
            top_n_metrics_loo, top_n_results_loo = self._evaluate_top_n_metrics_loo(evaluation_dataset, n, minimum_rating)
            top_n_metrics_full, top_n_results_full = self._evaluate_top_n_metrics_full(evaluation_dataset, n, minimum_rating, coverage_threshold)

        self._logger.info("Evaluation complete")

        return (
            accuracy_metrics + list(top_n_metrics_loo[:-1]) + top_n_metrics_full,
            accuracy_results + list(top_n_results_loo[:-1]) + top_n_results_full,
        )

    def _evaluate_accuracy(self, evaluation_dataset):
        self._logger.info("Evaluating accuracy")
        self._algorithm.fit(evaluation_dataset.trainset)
        predictions = self._algorithm.test(evaluation_dataset.testset)
        rmse = RecommenderMetrics.rmse(predictions)
        mae = RecommenderMetrics.mae(predictions)

        return ["RMSE", "MAE"], [rmse, mae]

    def _evaluate_top_n_metrics_loo(self, evaluation_dataset, n, minimum_rating):
        self._logger.info("Evaluating top-N metrics with Leave One Out validation")

        self._algorithm.fit(evaluation_dataset.loo_trainset)
        loo_predictions = self._algorithm.test(evaluation_dataset.loo_testset)
        anti_test_predictions = self._algorithm.test(
            evaluation_dataset.loo_anti_testset
        )
        top_n_predictions = RecommenderMetrics.get_top_n(
            anti_test_predictions, n, minimum_rating
        )

        hit_rate_metrics = RecommenderMetrics.hit_rate_metrics(top_n_predictions, loo_predictions, minimum_rating)


        return ["HR", "cHR", "ARHR", "rHR"], hit_rate_metrics

    def _evaluate_top_n_metrics_full(self, evaluation_dataset, n, minimum_rating, coverage_threshold):
        self._logger.info("Evaluating top-N metrics with full dataset")
        self._algorithm.fit(evaluation_dataset.full_trainset)
        all_predictions = self._algorithm.test(
            evaluation_dataset.full_anti_testset
        )
        top_n_predictions = RecommenderMetrics.get_top_n(
            all_predictions, n, minimum_rating
        )

        coverage = RecommenderMetrics.user_coverage(
            top_n_predictions,
            evaluation_dataset.full_trainset.n_users,
            minimum_rating=coverage_threshold,
        )

        diversity = RecommenderMetrics.diversity(
            top_n_predictions, evaluation_dataset.similarity_model
        )

        novelty = RecommenderMetrics.novelty(
            top_n_predictions, evaluation_dataset.popularity_rankings
        )

        return ["Coverage", "Diversity", "Novelty"], [coverage, diversity, novelty]

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
