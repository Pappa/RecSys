import itertools

from surprise import accuracy
from collections import defaultdict


class RecommenderMetrics:
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimum_rating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if estimatedRating >= minimum_rating:
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(top_n_preds, loo_validation_set):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in loo_validation_set:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in top_n_preds[int(userID)]:
                if int(leftOutMovieID) == int(movieID):
                    hit = True
                    break
            if hit:
                hits += 1

            total += 1

        # Compute overall precision
        return hits / total

    def CumulativeHitRate(top_n_preds, loo_validation_set, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for (
            userID,
            leftOutMovieID,
            actualRating,
            estimatedRating,
            _,
        ) in loo_validation_set:
            # Only look at ability to recommend things the users actually liked...
            if actualRating >= ratingCutoff:
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in top_n_preds[int(userID)]:
                    if int(leftOutMovieID) == movieID:
                        hit = True
                        break
                if hit:
                    hits += 1

                total += 1

        # Compute overall precision
        return hits / total

    def RatingHitRate(top_n_preds, loo_validation_set):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for (
            userID,
            leftOutMovieID,
            actualRating,
            estimatedRating,
            _,
        ) in loo_validation_set:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in top_n_preds[int(userID)]:
                if int(leftOutMovieID) == movieID:
                    hit = True
                    break
            if hit:
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(top_n_preds, loo_validation_set):
        summation = 0
        total = 0
        # For each left-out rating
        for (
            userID,
            leftOutMovieID,
            actualRating,
            estimatedRating,
            _,
        ) in loo_validation_set:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in top_n_preds[int(userID)]:
                rank = rank + 1
                if int(leftOutMovieID) == movieID:
                    hitRank = rank
                    break
            if hitRank > 0:
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(top_n_preds, numUsers, ratingThreshold=0):
        hits = 0
        for userID in top_n_preds.keys():
            hit = False
            for movieID, predictedRating in top_n_preds[userID]:
                if predictedRating >= ratingThreshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / numUsers

    def Diversity(top_n_preds, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in top_n_preds.keys():
            pairs = itertools.combinations(top_n_preds[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        if n > 0:
            S = total / n
            return 1 - S
        else:
            return 0

    def Novelty(top_n_preds, rankings):
        n = 0
        total = 0
        for userID in top_n_preds.keys():
            for rating in top_n_preds[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
