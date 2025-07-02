from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline


class EvaluationDataset:
    def __init__(self, data, popularity_rankings):
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
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self._LOOCV_train = train
            self._LOOCV_test = test

        self._LOOCV_anti_test_set = self._LOOCV_train.build_anti_testset()

        # Compute similarty matrix between items so we can measure diversity
        sim_options = {"name": "cosine", "user_based": False}
        self._sims_algo = KNNBaseline(sim_options=sim_options)
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
    def LOOCV_train_set(self):
        return self._LOOCV_train

    @property
    def LOOCV_test_set(self):
        return self._LOOCV_test

    @property
    def LOOCV_anti_test_set(self):
        return self._LOOCV_anti_test_set

    @property
    def similarities(self):
        return self._sims_algo

    @property
    def popularity_rankings(self):
        return self._rankings
