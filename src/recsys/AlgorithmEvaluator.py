from recsys.RecommenderMetrics import RecommenderMetrics
import logging


class AlgorithmEvaluator:
    def __init__(self, algorithm, name, verbose=True):
        self._algorithm = algorithm
        self._name = name
        self._accuracy_metrics = {}
        self._top_n_metrics = {}

        self._logger = logging.getLogger(f"{self.__class__.__name__}({name})")
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)


    def evaluate(
        self, evaluation_dataset, top_n_metrics=False, minimum_rating=0.4, n=10
    ):
        self._evaluate_accuracy(evaluation_dataset)

        if top_n_metrics:
            self._evaluate_top_n_metrics(evaluation_dataset, n, minimum_rating)

        self._logger.info("Analysis complete.")

        return {**self._accuracy_metrics, **self._top_n_metrics}
    
    def _evaluate_accuracy(self, evaluation_dataset):
        self._logger.info("Evaluating accuracy.")
        self._algorithm.fit(evaluation_dataset.train_set)
        predictions = self._algorithm.test(evaluation_dataset.test_set)
        self._accuracy_metrics["RMSE"] = RecommenderMetrics.rmse(predictions)
        self._accuracy_metrics["MAE"] = RecommenderMetrics.mae(predictions)

    def _evaluate_top_n_metrics(self, evaluation_dataset, n, minimum_rating):
        # Evaluate top-10 with Leave One Out validation
        self._logger.info("Evaluating top-N with leave-one-out validation.")
        self._algorithm.fit(evaluation_dataset.loo_train_set)
        loo_validation_set = self._algorithm.test(evaluation_dataset.loo_test_set)
        # Build predictions for all ratings not in the training set
        all_preds = self._algorithm.test(
            evaluation_dataset.loo_anti_test_set
        )
        # Compute top 10 recs for each user
        top_n_preds = RecommenderMetrics.get_top_n(
            all_preds, n, minimum_rating
        )
        self._logger.info("Computing hit-rate and rank metrics.")
        # See how often we recommended a movie the user actually rated
        self._top_n_metrics["HR"] = RecommenderMetrics.hit_rate(
            top_n_preds, loo_validation_set
        )
        # See how often we recommended a movie the user actually liked
        self._top_n_metrics["cHR"] = RecommenderMetrics.cumulative_hit_rate(
            top_n_preds, loo_validation_set
        )
        # Compute ARHR
        self._top_n_metrics["ARHR"] = RecommenderMetrics.average_reciprocal_hit_rank(
            top_n_preds, loo_validation_set
        )

        # Evaluate properties of recommendations on full training set
        self._logger.info("Computing recommendations with full data set.")
        self._algorithm.fit(evaluation_dataset.full_train_set)
        all_preds = self._algorithm.test(evaluation_dataset.full_anti_test_set)
        top_n_preds = RecommenderMetrics.get_top_n(
            all_preds, n, minimum_rating
        )
        self._logger.info("Analyzing coverage, diversity, and novelty.")
        # self._logger.info user coverage with a minimum predicted rating of 4.0:
        self._top_n_metrics["Coverage"] = RecommenderMetrics.user_coverage(
            top_n_preds,
            evaluation_dataset.full_train_set.n_users,
            rating_threshold=4.0,
        )
        # Measure diversity of recommendations:
        self._top_n_metrics["Diversity"] = RecommenderMetrics.diversity(
            top_n_preds, evaluation_dataset.similarities
        )

        # Measure novelty (average popularity rank of recommendations):
        self._top_n_metrics["Novelty"] = RecommenderMetrics.novelty(
            top_n_preds, evaluation_dataset.popularity_rankings
        )

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
