from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline
import logging


class EvaluationDataset:
    def __init__(self, data, popularity_rankings, verbose=False):
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self._rankings = popularity_rankings

        # Build a full training set for evaluating overall properties
        self._full_train_set = data.build_full_trainset()
        self._full_anti_test_set = self._full_train_set.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self._train_set, self._test_set = train_test_split(
            data, test_size=0.25, random_state=1
        )

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        loo_iterator = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loo_iterator.split(data):
            self._loo_train = train
            self._loo_test = test

        self._loo_anti_test_set = self._loo_train.build_anti_testset()

        # Compute similarty matrix between items so we can measure diversity
        sim_options = {"name": "cosine", "user_based": False}
        self._sims_algo = KNNBaseline(sim_options=sim_options, verbose=self._verbose)
        self._sims_algo.fit(self._full_train_set)

    @property
    def full_train_set(self):
        return self._full_train_set

    @property
    def full_anti_test_set(self):
        return self._full_anti_test_set

    def get_anti_test_set_for_user(self, test_subject):
        trainset = self._full_train_set
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [
            (trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill)
            for i in trainset.all_items()
            if i not in user_items
        ]
        return anti_testset

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def loo_train_set(self):
        return self._loo_train

    @property
    def loo_test_set(self):
        return self._loo_test

    @property
    def loo_anti_test_set(self):
        return self._loo_anti_test_set

    @property
    def similarities(self):
        return self._sims_algo

    @property
    def popularity_rankings(self):
        return self._rankings
