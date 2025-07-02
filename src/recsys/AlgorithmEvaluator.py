from recsys.RecommenderMetrics import RecommenderMetrics
import logging


class AlgorithmEvaluator:
    def __init__(self, algorithm, name, verbose=True):
        self._algorithm = algorithm
        self._name = name
        self._logger = logging.getLogger(name)
        level = logging.INFO if verbose else logging.WARNING
        self._logger.setLevel(level)

    def evaluate(self, evaluationData, doTopN, minimumRating=0.4, n=10):
        metrics = {}
        # Compute accuracy
        self._logger.info("Evaluating accuracy...")
        self._algorithm.fit(evaluationData.GetTrainSet())
        predictions = self._algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if doTopN:
            # Evaluate top-10 with Leave One Out testing
            self._logger.info("Evaluating top-N with leave-one-out...")
            self._algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self._algorithm.test(evaluationData.GetLOOCVTestSet())
            # Build predictions for all ratings not in the training set
            allPredictions = self._algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n, minimumRating)
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
            self._algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self._algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n, minimumRating)
            self._logger.info("Analyzing coverage, diversity, and novelty...")
            # self._logger.info user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(
                topNPredicted,
                evaluationData.GetFullTrainSet().n_users,
                ratingThreshold=4.0,
            )
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(
                topNPredicted, evaluationData.GetSimilarities()
            )

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(
                topNPredicted, evaluationData.GetPopularityRankings()
            )

        self._logger.info("Analysis complete.")

        return metrics

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return self._algorithm
