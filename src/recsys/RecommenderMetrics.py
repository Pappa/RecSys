import itertools

from surprise import accuracy, Prediction
from collections import defaultdict


class RecommenderMetrics:
    @staticmethod
    def mae(predictions: list[Prediction], verbose=False):
        return accuracy.mae(predictions, verbose=verbose)

    @staticmethod
    def rmse(predictions: list[Prediction], verbose=False):
        return accuracy.rmse(predictions, verbose=verbose)

    @staticmethod
    def get_top_n(predictions: list[Prediction], n=10, minimum_rating=4.0):
        top_n = defaultdict(list)

        for user_id, movie_id, true_rating, predicted_rating, _ in predictions:
            if predicted_rating >= minimum_rating:
                top_n[int(user_id)].append((int(movie_id), predicted_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n

    @staticmethod
    def hit_rate(top_n_preds, loo_validation_set):
        hits = 0
        total = 0

        # For each left-out rating
        for held_out_rating in loo_validation_set:
            user_id = held_out_rating[0]
            held_out_movie_id = held_out_rating[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movie_id, predicted_rating in top_n_preds[int(user_id)]:
                if int(held_out_movie_id) == int(movie_id):
                    hit = True
                    break
            if hit:
                hits += 1

            total += 1

        # Compute overall precision
        return hits / total

    @staticmethod
    def cumulative_hit_rate(top_n_preds, loo_validation_set, rating_cutoff=1e-5):
        hits = 0
        total = 0

        # For each left-out rating
        for (
            user_id,
            held_out_movie_id,
            true_rating,
            predicted_rating,
            _,
        ) in loo_validation_set:
            # Only consider ratings that are greater than or equal to the cutoff
            if true_rating >= rating_cutoff:
                # Is it in the predicted top 10 for this user?
                hit = False
                for movie_id, predicted_rating in top_n_preds[int(user_id)]:
                    if int(held_out_movie_id) == movie_id:
                        hit = True
                        break
                if hit:
                    hits += 1

                total += 1

        # Compute overall precision
        return hits / total

    @staticmethod
    def rating_hit_rate(top_n_preds, loo_validation_set):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for (
            user_id,
            held_out_movie_id,
            true_rating,
            predicted_rating,
            _,
        ) in loo_validation_set:
            # Is it in the predicted top N for this user?
            hit = False
            for movie_id, predicted_rating in top_n_preds[int(user_id)]:
                if int(held_out_movie_id) == movie_id:
                    hit = True
                    break
            if hit:
                hits[true_rating] += 1

            total[true_rating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    @staticmethod
    def average_reciprocal_hit_rank(top_n_preds, loo_validation_set):
        summation = 0
        total = 0
        # For each left-out rating
        for (
            user_id,
            held_out_movie_id,
            true_rating,
            predicted_rating,
            _,
        ) in loo_validation_set:
            # Is it in the predicted top N for this user?
            hit_rank = 0
            rank = 0
            for movie_id, predicted_rating in top_n_preds[int(user_id)]:
                rank = rank + 1
                if int(held_out_movie_id) == movie_id:
                    hit_rank = rank
                    break
            if hit_rank > 0:
                summation += 1.0 / hit_rank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    @staticmethod
    def user_coverage(top_n_preds, n_users, rating_threshold=0):
        hits = 0
        for user_id in top_n_preds.keys():
            hit = False
            for movie_id, predicted_rating in top_n_preds[user_id]:
                if predicted_rating >= rating_threshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / n_users

    @staticmethod
    def diversity(top_n_preds, similarities_model):
        n = 0
        total = 0
        similarity_matrix = similarities_model.compute_similarities()
        for user_id in top_n_preds.keys():
            pairs = itertools.combinations(top_n_preds[user_id], 2)
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
    def novelty(top_n_preds, rankings):
        n = 0
        total = 0
        for user_id in top_n_preds.keys():
            for rating in top_n_preds[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1
        return total / n
