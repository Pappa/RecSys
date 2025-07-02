import os
import csv
import re
from pathlib import Path
from surprise import Dataset
from surprise import Reader
from collections import defaultdict


class MovieLens:
    movie_id_to_name_map: dict[int, str] = {}
    name_to_movie_id_map: dict[str, int] = {}

    @classmethod
    def load(cls, *args, **kwargs):
        lens = MovieLens()
        print("Loading movie ratings...")
        data = lens.loadMovieLensLatestSmall()
        print("\nComputing movie popularity ranks so we can measure novelty later...")
        rankings = lens.getPopularityRanks()
        return (lens, data, rankings)

    def __init__(self) -> None:
        self._path_base = os.path.dirname(os.path.realpath(__file__))
        self._ratings_path = Path(self._path_base + "/data/ratings.csv").resolve()
        self._movies_path = Path(self._path_base + "/data/movies.csv").resolve()
        self._visual_features_path = Path(
            self._path_base + "/data/LLVisualFeatures13K_Log.csv"
        ).resolve()

    def loadMovieLensLatestSmall(self):
        self.movie_id_to_name_map = {}
        self.name_to_movie_id_map = {}

        reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

        ratings_dataset = Dataset.load_from_file(self._ratings_path, reader=reader)

        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movieID = int(row[0])
                movieName = row[1]
                self.movie_id_to_name_map[movieID] = movieName
                self.name_to_movie_id_map[movieName] = movieID

        return ratings_dataset

    def getNewMovies(self):
        newMovies = []
        years = self.getYears()
        # What's the newest year in our data?
        latestYear = max(years.values())
        print("Newest year is ", latestYear)
        for movieID, year in years.items():
            if year == latestYear:
                newMovies.append(movieID)
                # print (self.getMovieName(movieID))
        return newMovies

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self._ratings_path, newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)  # skip header row
            for row in rating_reader:
                userID = int(row[0])
                if user == userID:
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if hitUser and (user != userID):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self._ratings_path, newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)  # skip header row
            for row in rating_reader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(
            ratings.items(), key=lambda x: x[1], reverse=True
        ):
            rankings[movieID] = rank
            rank += 1
        return rankings

    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movieID = int(row[0])
                genreList = row[2].split("|")
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for movieID, genreIDList in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield

        return genres

    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self._movies_path, newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)  # skip header row
            for row in movie_reader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years

    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open(self._visual_features_path, newline="") as csvfile:
            mes_reader = csv.reader(csvfile)
            next(mes_reader)  # skip header row
            for row in mes_reader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [
                    avgShotLength,
                    meanColorVariance,
                    stddevColorVariance,
                    meanMotion,
                    stddevMotion,
                    meanLightingKey,
                    numShots,
                ]
        return mes

    def getMovieName(self, movieID):
        if movieID in self.movie_id_to_name_map:
            return self.movie_id_to_name_map[movieID]
        else:
            return ""

    def getMovieID(self, movieName):
        if movieName in self.name_to_movie_id_map:
            return self.name_to_movie_id_map[movieName]
        else:
            return 0
