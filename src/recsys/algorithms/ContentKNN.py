from surprise import AlgoBase
from surprise import PredictionImpossible
from recsys.MovieLens import MovieLens
import math
import numpy as np
import heapq
import logging


class ContentKNN(AlgoBase):
    def __init__(self, k=40, sim_options={}, verbose=False):
        AlgoBase.__init__(self)
        self.k = k
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes

        # Load up genre vectors for every movie
        ml = MovieLens()
        genres = ml.get_genres()
        years = ml.get_years()
        mes = ml.get_mis_en_scene()

        self._logger.info("Generate content-based similarity matrix")

        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for rating in range(self.trainset.n_items):
            if rating % 500 == 0:
                self._logger.info(f"{rating} of {self.trainset.n_items}")
            for other_rating in range(rating + 1, self.trainset.n_items):
                movie_id = int(self.trainset.to_raw_iid(rating))
                other_movie_id = int(self.trainset.to_raw_iid(other_rating))
                genre_similarity = self.compute_genre_similarity(
                    movie_id, other_movie_id, genres
                )
                year_similarity = self.compute_year_similarity(
                    movie_id, other_movie_id, years
                )
                # mes_similarity = self.compute_mise_en_scene_similarity(
                #     movie_id, other_movie_id, mes
                # )
                self.similarities[rating, other_rating] = (
                    genre_similarity * year_similarity
                )
                self.similarities[other_rating, rating] = self.similarities[
                    rating, other_rating
                ]

        return self

    def compute_genre_similarity(self, movie_id, other_movie_id, genres):
        genres1 = genres[movie_id]
        genres2 = genres[other_movie_id]
        sum_xx, sum_xy, sum_yy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sum_xx += x * x
            sum_yy += y * y
            sum_xy += x * y

        return sum_xy / math.sqrt(sum_xx * sum_yy)

    def compute_year_similarity(self, movie_id, other_movie_id, years):
        diff = abs(years[movie_id] - years[other_movie_id])
        sim = math.exp(-diff / 10.0)
        return sim

    def compute_mise_en_scene_similarity(self, movie_id, other_movie_id, mes):
        mes1 = mes[movie_id]
        mes2 = mes[other_movie_id]
        if mes1 and mes2:
            shot_length_diff = math.fabs(mes1[0] - mes2[0])
            color_variance_diff = math.fabs(mes1[1] - mes2[1])
            motion_diff = math.fabs(mes1[3] - mes2[3])
            lighting_diff = math.fabs(mes1[5] - mes2[5])
            num_shots_diff = math.fabs(mes1[6] - mes2[6])
            return (
                shot_length_diff
                * color_variance_diff
                * motion_diff
                * lighting_diff
                * num_shots_diff
            )
        else:
            return 0

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unkown.")

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i, rating[0]]
            neighbors.append((genre_similarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        sim_total = weighted_sum = 0
        for sim_score, rating in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating

        if sim_total == 0:
            raise PredictionImpossible("No neighbors")

        predicted_rating = weighted_sum / sim_total

        return predicted_rating
