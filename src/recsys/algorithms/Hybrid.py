from surprise import AlgoBase

class Hybrid(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights

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

    