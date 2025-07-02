from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import csv


def load_movie_names():
    movie_id_to_name_map = {}
    with open("../ml-20m/movies.csv", newline="", encoding="ISO-8859-1") as csvfile:
        movie_reader = csv.reader(csvfile)
        next(movie_reader)  # Skip header line
        for row in movie_reader:
            movie_id = int(row[0])
            movie_name = row[1]
            movie_id_to_name_map[movie_id] = movie_name
    return movie_id_to_name_map


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("ALSExample")
        .config("spark.executor.cores", "4")
        .getOrCreate()
    )

    lines = spark.read.option("header", "true").csv("../ml-20m/ratings.csv").rdd

    ratings_rdd = lines.map(
        lambda p: Row(
            userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=int(p[3])
        )
    )

    ratings = spark.createDataFrame(ratings_rdd)

    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
    )
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    user_recs = model.recommendForAllUsers(10)

    sample_user_id = 85

    sample_user_recs = user_recs.filter(user_recs["userId"] == sample_user_id).collect()

    spark.stop()

    movie_id_to_name_map = load_movie_names()

    for row in sample_user_recs:
        for rec in row.recommendations:
            if rec.movieId in movie_id_to_name_map:
                print(movie_id_to_name_map[rec.movieId])
