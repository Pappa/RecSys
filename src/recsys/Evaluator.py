from recsys.EvaluationDataset import EvaluationDataset
from recsys.AlgorithmEvaluator import AlgorithmEvaluator
import logging
from dataclasses import dataclass
import pandas as pd

@dataclass
class EvaluationResult:
    """Evaluation result for a single algorithm."""
    algorithm: str
    metrics: list[str]
    values: list[float]

    def show(self):
        print(f"{self.algorithm}:")
        for metric, result in zip(self.metrics, self.values):
            print(f"  {metric}: {result}")

@dataclass
class EvaluationResultSet:
    """Evaluation result for a set of algorithms."""
    results: list[EvaluationResult]

    def show(self):
        for result in self.results:
            result.show()

    def to_df(self) -> pd.DataFrame:
        algorithms = [result.algorithm for result in self.results]
        values = [result.values for result in self.results]
        metrics = self.results[0].metrics

        return pd.DataFrame(
            values, columns=metrics, index=pd.Index(algorithms, name="Algorithm")
        )

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

    def evaluate(self, top_n_metrics=False, minimum_rating=1e-5) -> EvaluationResultSet:
        evaluation_results = []
        
        for algorithm in self._algorithms:
            self._logger.info(f"Evaluating: {algorithm.name}")
            algorithm_metrics, algorithm_results = algorithm.evaluate(
                evaluation_dataset=self._dataset,
                top_n_metrics=top_n_metrics,
                minimum_rating=minimum_rating,
            )
            evaluation_results.append(EvaluationResult(
                algorithm=algorithm.name,
                metrics=algorithm_metrics,
                values=algorithm_results,
            ))

        return EvaluationResultSet(results=evaluation_results)

    def sample_top_n_recs(self, uid, n=10):
        if not uid:
            raise ValueError("uid is required")

        results = {}
        for algorithm in self._algorithms:
            self._logger.info(f"Using recommender: {algorithm.name}")

            self._logger.info("Training model")
            trainset = self._dataset.full_trainset
            algorithm.algorithm.fit(trainset)

            self._logger.info("Generate recommendations")
            testset = self._dataset.get_anti_testset_for_user(uid)
            predictions = algorithm.algorithm.test(testset)

            top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
            results[algorithm.name] = [p.iid for p in top_n_predictions]

        return results