from recsys.RecommenderMetrics import RecommenderMetrics
import logging


class AlgorithmEvaluator:
    def __init__(self, algorithm, name, verbose=True):
        self._algorithm = algorithm
        self._name = name
        self._logger = logging.getLogger(name)
        level = logging.INFO if verbose else logging.WARNING
        self._logger.setLevel(level)

    def evaluate(
        self, evaluation_dataset, top_n_metrics=False, minimum_rating=0.4, n=10
    ):
        metrics = {}
        # Compute accuracy
        self._logger.info("Evaluating accuracy...")
        self._algorithm.fit(evaluation_dataset.train_set)
        predictions = self._algorithm.test(evaluation_dataset.test_set)
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if top_n_metrics:
            # Evaluate top-10 with Leave One Out testing
            self._logger.info("Evaluating top-N with leave-one-out...")
            self._algorithm.fit(evaluation_dataset.LOOCV_train_set)
            leftOutPredictions = self._algorithm.test(evaluation_dataset.LOOCV_test_set)
            # Build predictions for all ratings not in the training set
            allPredictions = self._algorithm.test(
                evaluation_dataset.LOOCV_anti_test_set
            )
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(
                allPredictions, n, minimum_rating
            )
            self._logger.info("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(
                topNPredicted, leftOutPredictions
            )
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(
                topNPredicted, leftOutPredictions
            )
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(
                topNPredicted, leftOutPredictions
            )

            # Evaluate properties of recommendations on full training set
            self._logger.info("Computing recommendations with full data set...")
            self._algorithm.fit(evaluation_dataset.full_train_set)
            allPredictions = self._algorithm.test(evaluation_dataset.full_anti_test_set)
            topNPredicted = RecommenderMetrics.GetTopN(
                allPredictions, n, minimum_rating
            )
            self._logger.info("Analyzing coverage, diversity, and novelty...")
            # self._logger.info user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(
                topNPredicted,
                evaluation_dataset.full_train_set.n_users,
                ratingThreshold=4.0,
            )
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(
                topNPredicted, evaluation_dataset.similarities
            )

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(
                topNPredicted, evaluation_dataset.popularity_rankings
            )

        self._logger.info("Analysis complete.")

        return metrics

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
