from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from recsys.MovieLens import MovieLens

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALSExample").getOrCreate()

    lines = spark.read.option("header", "true").csv("../../src/data/ratings.csv").rdd

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

    ml = MovieLens()
    ml.load_movielens_data()

    for row in sample_user_recs:
        for rec in row.recommendations:
            print(ml.get_movie_name(rec.movieId))
