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
        top_n_results_loo = []
        top_n_results_full = []

        if top_n_metrics:
            top_n_results_loo = self._evaluate_top_n_metrics_loo(evaluation_dataset, n, minimum_rating)
            top_n_results_full = self._evaluate_top_n_metrics_full(evaluation_dataset, n, minimum_rating)

        self._logger.info("Evaluation complete")

        return accuracy_results + top_n_results_loo + top_n_results_full

    def _evaluate_accuracy(self, evaluation_dataset):
        self._logger.info("Evaluating accuracy")
        self._algorithm.fit(evaluation_dataset.trainset)
        predictions = self._algorithm.test(evaluation_dataset.testset)
        rmse = RecommenderMetrics.rmse(predictions)
        mae = RecommenderMetrics.mae(predictions)
        return [
            ("RMSE", rmse),
            ("MAE", mae),
        ]

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


        hit_rate = RecommenderMetrics.hit_rate(top_n_predictions, loo_predictions)

        cumulative_hit_rate = RecommenderMetrics.cumulative_hit_rate(
            top_n_predictions, loo_predictions
        )

        arhr = RecommenderMetrics.average_reciprocal_hit_rank(
            top_n_predictions, loo_predictions
        )

        return [
            ("HR", hit_rate),
            ("cHR", cumulative_hit_rate),
            ("ARHR", arhr),
        ]

    def _evaluate_top_n_metrics_full(self, evaluation_dataset, n, minimum_rating):
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
            minimum_rating=minimum_rating,
        )

        diversity = RecommenderMetrics.diversity(
            top_n_predictions, evaluation_dataset.similarities
        )

        novelty = RecommenderMetrics.novelty(
            top_n_predictions, evaluation_dataset.popularity_rankings
        )

        return [
            ("Coverage", coverage),
            ("Diversity", diversity),
            ("Novelty", novelty),
        ]

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
