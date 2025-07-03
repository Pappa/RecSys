import os
import csv
import re
from pathlib import Path
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import logging


class MovieLens:
    _names_by_movie_id: dict[int, str] = {}
    _movies_by_name: dict[str, int] = {}

    @classmethod
    def load(cls, *args, **kwargs):
        lens = MovieLens(*args, **kwargs)
        data = lens.load_movielens_data()
        rankings = lens.get_popularity_ranks()
        return (lens, data, rankings)

    def __init__(self, verbose=False) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self._logger.info("Initializing MovieLens")
        self._path_base = os.path.dirname(os.path.realpath(__file__))
        self._ratings_path = Path(self._path_base + "/data/ratings.csv").resolve()
        self._movies_path = Path(self._path_base + "/data/movies.csv").resolve()
        self._visual_features_path = Path(
            self._path_base + "/data/visual_features.csv"
        ).resolve()

    def load_movielens_data(self):
        self._logger.info("Loading MovieLens data")
        self._names_by_movie_id = defaultdict(int)
        self._movies_by_name = defaultdict(str)

        reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

        ratings_dataset = Dataset.load_from_file(self._ratings_path, reader=reader)

        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movie_id = int(row[0])
                movie_name = row[1]
                self._names_by_movie_id[movie_id] = movie_name
                self._movies_by_name[movie_name] = movie_id

        return ratings_dataset

    def get_new_movies(self):
        new_movies = []
        years = self.get_years()
        # What's the newest year in our data?
        latest_year = max(years.values())
        for movie_id, year in years.items():
            if year == latest_year:
                new_movies.append(movie_id)
        return new_movies

    def get_user_ratings(self, user):
        user_ratings = []
        hit_user = False
        with open(self._ratings_path, newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)  # skip header row
            for row in rating_reader:
                user_id = int(row[0])
                if user == user_id:
                    movie_id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_id, rating))
                    hit_user = True
                if hit_user and (user != user_id):
                    break

        return user_ratings

    def get_popularity_ranks(self):
        self._logger.info("Generate movie popularity ranks to measure novelty later.")
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self._ratings_path, newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)  # skip header row
            for row in rating_reader:
                movie_id = int(row[1])
                ratings[movie_id] += 1
        rank = 1
        for movie_id, rating_count in sorted(
            ratings.items(), key=lambda x: x[1], reverse=True
        ):
            rankings[movie_id] = rank
            rank += 1
        return rankings

    def get_genres(self):
        genres = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0
        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movie_id = int(row[0])
                genre_list = row[2].split("|")
                genre_id_list = []
                for genre in genre_list:
                    if genre in genre_ids:
                        genre_id = genre_ids[genre]
                    else:
                        genre_id = max_genre_id
                        genre_ids[genre] = genre_id
                        max_genre_id += 1
                    genre_id_list.append(genre_id)
                genres[movie_id] = genre_id_list
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for movie_id, genre_id_list in genres.items():
            bitfield = [0] * max_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield

        return genres

    def get_years(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movie_id = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movie_id] = int(year)
        return years

    def get_mis_en_scene(self):
        mes = defaultdict(list)
        with open(self._visual_features_path, newline="") as csvfile:
            mes_reader = csv.reader(csvfile)
            next(mes_reader)  # skip header row
            for row in mes_reader:
                self._logger.info(f"mes row: {type(row)}")
                movie_id = int(row[0])
                avg_shot_length = float(row[1])
                mean_color_variance = float(row[2])
                std_dev_color_variance = float(row[3])
                mean_motion = float(row[4])
                std_dev_motion = float(row[5])
                mean_lighting_key = float(row[6])
                num_shots = float(row[7])
                mes[movie_id] = [
                    avg_shot_length,
                    mean_color_variance,
                    std_dev_color_variance,
                    mean_motion,
                    std_dev_motion,
                    mean_lighting_key,
                    num_shots,
                ]
        return mes

    def get_movie_name(self, movie_id):
        return self._names_by_movie_id[movie_id]

    def get_movie_id(self, movie_name):
        return self._movies_by_name[movie_name]
