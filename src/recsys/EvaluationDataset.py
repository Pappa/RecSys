from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline, Dataset, Trainset
import logging


class EvaluationDataset:
    _verbose: bool
    _logger: logging.Logger
    _trainset: Trainset
    _testset: list[tuple]
    _loo_trainset: Trainset
    _loo_testset: list[tuple]

    def __init__(self, data: Dataset, popularity_rankings, verbose=False):
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self._rankings = popularity_rankings

        # Build a full training set for evaluating overall properties
        self._full_trainset = data.build_full_trainset()
        self._full_anti_testset = self._full_trainset.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self._trainset, self._testset = train_test_split(
            data, test_size=0.25, random_state=1
        )

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        loo_iterator = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loo_iterator.split(data):
            self._loo_trainset = train
            self._loo_testset = test

        self._loo_anti_testset = self._loo_trainset.build_anti_testset()

        # Compute similarty matrix between items so we can measure diversity
        # TODO: replace this with a more efficient similarity matrix
        sim_options = {"name": "cosine", "user_based": False}
        self._similarity_model = KNNBaseline(sim_options=sim_options, verbose=self._verbose)
        self._similarity_model.fit(self._full_trainset)

    @property
    def full_trainset(self):
        return self._full_trainset

    @property
    def full_anti_testset(self):
        return self._full_anti_testset

    def get_anti_testset_for_user(self, test_uid):
        trainset = self._full_trainset
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_uid))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [
            (trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill)
            for i in trainset.all_items()
            if i not in user_items
        ]
        return anti_testset

    @property
    def trainset(self):
        return self._trainset

    @property
    def testset(self):
        return self._testset

    @property
    def loo_trainset(self):
        return self._loo_trainset

    @property
    def loo_testset(self):
        return self._loo_testset

    @property
    def loo_anti_testset(self):
        return self._loo_anti_testset

    @property
    def similarity_model(self):
        return self._similarity_model

    @property
    def popularity_rankings(self):
        return self._rankings
