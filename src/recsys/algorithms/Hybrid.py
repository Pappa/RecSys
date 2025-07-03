from surprise import AlgoBase
import logging


class Hybrid(AlgoBase):
    def __init__(self, algorithms, weights, sim_options={}, verbose=False):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        for algorithm in self.algorithms:
            algorithm.fit(trainset)

        return self

    def estimate(self, u, i):
        sum_scores = 0
        sum_weights = 0

        for idx in range(len(self.algorithms)):
            sum_scores += self.algorithms[idx].estimate(u, i) * self.weights[idx]
            sum_weights += self.weights[idx]

        return sum_scores / sum_weights
