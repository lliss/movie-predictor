#! /usr/bin/env python

'''
Simple program that uses SVM to predict movies a user will like vs ones they
will not.
'''

import csv
import json
import re

from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

def create_base_dict():
    '''
    Create a dictionary representing the base structure for a movie.
    '''
    return {
        'budget': 0,
        'production_company': '',
        'release_year': 0,
        'revenue': 0,
        'runtime': 0,
        'Science Fiction': 0,
        'Music': 0,
        'Adventure': 0,
        'Thriller': 0,
        'Animation': 0,
        'Romance': 0,
        'Fantasy': 0,
        'Drama': 0,
        'Mystery': 0,
        'Documentary': 0,
        'War': 0,
        'Family': 0,
        'History': 0,
        'Action': 0,
        'Crime': 0,
        'Comedy': 0,
        'Western': 0,
        'TV Movie': 0,
        'Foreign': 0,
        'Horror': 0,
        'Director': '',
        'Actor': ''
    }


def extract_year(date):
    '''
    Given a string date, extract the 4 digit year and return it.
    '''
    return int(date[:4])


def prepare_movie_from_row(row):
    '''
    Given a dictionary row read from a CSV file, create a dictionary
    representing a movie.
    '''
    movie = create_base_dict()
    row['production_companies'] = json.loads(row['production_companies'])
    if len(row['production_companies']) > 0:
        movie['production_company'] = row['production_companies'][0]['name']
    movie['budget'] = int(row['budget'])
    movie['release_year'] = extract_year(row['release_date'])
    movie['revenue'] = int(row['revenue'])
    movie['runtime'] = int(row['runtime'])

    row['genres'] = json.loads(row['genres'])
    for genre in row['genres']:
        movie[genre['name']] = 1

    return movie


def append_to_movie(movie, credit, movie_id):
    '''
    Given a movie dictionary cast and crew information, append the lead actor
    and director information to the dictionary and return it.
    '''
    try:
        data = credit[movie_id]
        movie['Actor'] = data['actor']
        movie['Director'] = data['director']
        return movie
    except:
        return movie


def main():
    '''
    Main method to read in the data prepare it and run the SVM prediction.
    '''
    movies = []
    likes = []
    predictions = []
    movie_casts = {}

    with open('data/credits.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        counter = 1
        for row in reader:
            actor = re.search(r'(\'name\':?\s\')(.+?)(?=\',)', row['cast'])
            director = re.search(r'(\'name\':?\s\')(.+?)(?=\',)', row['crew'])
            try:
                movie_casts[row['id']] = {'actor': actor.group(2), 'director': director.group(2)}
            except:
                pass

    with open('data/movies_metadata.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        counter = 0
        for row in reader:
            counter += 2
            try:
                if row['prediction'] == '1' or row['prediction'] == '0':
                    movie = prepare_movie_from_row(row)
                    movie = append_to_movie(movie, movie_casts, row['id'])
                    movies.append(movie)
                    if row['prediction'] == '1':
                        likes.append(1)
                    else:
                        likes.append(0)
                elif row['prediction'] == 'p':
                    movie = prepare_movie_from_row(row)
                    movie = append_to_movie(movie, movie_casts, row['id'])
                    predictions.append({
                        'title': row['title'],
                        'data': movie
                    })
            except Exception as ex:
                print('Error on line: ', counter)
                print(ex)

    full_data_set = [] + movies
    for movie in predictions:
        full_data_set += [movie['data']]

    vectorizer = DictVectorizer()
    model = vectorizer.fit_transform(full_data_set).toarray()
    scaler = MinMaxScaler()
    model = scaler.fit_transform(model)
    training_set = model[:len(movies)]
    prediction_set = model[len(movies):]
    clf = svm.SVC(kernel='linear', cache_size=1000, max_iter=1000000000)
    clf.fit(training_set, likes)

    for index, movie in enumerate(prediction_set):
        print(clf.predict([movie]), predictions[index]['title'])


if __name__ == '__main__':
    main()
