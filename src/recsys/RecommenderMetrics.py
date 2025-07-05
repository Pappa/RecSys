import itertools
from surprise import accuracy, Prediction
from collections import defaultdict
import logging

_logger = logging.getLogger(__name__)


class RecommenderMetrics:
    @staticmethod
    def mae(predictions: list[Prediction], verbose=False):
        _logger.info("Calculating MAE")
        return accuracy.mae(predictions, verbose=verbose)

    @staticmethod
    def rmse(predictions: list[Prediction], verbose=False):
        _logger.info("Calculating RMSE")
        return accuracy.rmse(predictions, verbose=verbose)

    @staticmethod
    def get_top_n(predictions: list[Prediction], n=10, minimum_rating=4.0):
        _logger.info("Get top-N predictions")
        top_n = defaultdict(list)

        for user_id, movie_id, true_rating, predicted_rating, _ in predictions:
            if predicted_rating >= minimum_rating:
                top_n[int(user_id)].append((int(movie_id), predicted_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n

    @staticmethod
    def hit_rate_metrics(top_n_predictions, loo_testset, minimum_rating=1e-5):
        _logger.info("Calculating hit-rate metrics")
        hits = 0
        cumulative_hits = 0
        reciprocal_hits = 0
        total = len(loo_testset)
        hits_by_rating = defaultdict(int)
        total_by_rating = defaultdict(int)

        # For each left-out rating
        for (
            held_out_user_id,
            held_out_movie_id,
            held_out_true_rating,
            held_out_predicted_rating,
            _,
        ) in loo_testset:
            total_by_rating[held_out_true_rating] += 1

            # Is it in the predicted top-N for this user?
            top_n_predictions_for_user = top_n_predictions[int(held_out_user_id)]

            rank = next((idx for idx, (movie_id, predicted_rating) in enumerate(top_n_predictions_for_user) if int(held_out_movie_id) == int(movie_id)), None)
            if rank is not None:
                hits += 1
                hits_by_rating[held_out_true_rating] += 1
                reciprocal_hits += 1.0 / (rank + 1)
            if rank is not None and held_out_true_rating >= minimum_rating:
                cumulative_hits += 1


        hit_rate = hits / total
        cumulative_hit_rate = cumulative_hits / total
        average_reciprocal_hit_rank = reciprocal_hits / total
        rating_hit_rate = [
            (rating, hits_by_rating[rating] / total_by_rating[rating]) for rating in sorted(hits_by_rating.keys())
        ]   
    
        return hit_rate, cumulative_hit_rate, average_reciprocal_hit_rank, rating_hit_rate

    # What percentage of users have at least one "good" recommendation
    @staticmethod
    def user_coverage(top_n_predictions, n_users, minimum_rating=0):
        _logger.info(
            f"Calculating user coverage with a minimum predicted rating of {minimum_rating}"
        )
        hits = 0
        for user_id in top_n_predictions.keys():
            hit = False
            top_n_predictions_for_user = top_n_predictions[int(user_id)]
            for movie_id, predicted_rating in top_n_predictions_for_user:
                if predicted_rating >= minimum_rating:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / n_users

    @staticmethod
    def diversity(top_n_predictions, similarities_model):
        _logger.info("Calculating diversity")
        n = 0
        total = 0
        similarity_matrix = similarities_model.compute_similarities()
        for user_id in top_n_predictions.keys():
            pairs = itertools.combinations(top_n_predictions[user_id], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                inner_id1 = similarities_model.trainset.to_inner_iid(str(movie1))
                inner_id2 = similarities_model.trainset.to_inner_iid(str(movie2))
                similarity = similarity_matrix[inner_id1][inner_id2]
                total += similarity
                n += 1

        if n > 0:
            S = total / n
            return 1 - S
        else:
            return 0

    @staticmethod
    def novelty(top_n_predictions, rankings):
        _logger.info("Calculating novelty")
        n = 0
        total = 0
        for user_id in top_n_predictions.keys():
            for rating in top_n_predictions[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1
        return total / n
