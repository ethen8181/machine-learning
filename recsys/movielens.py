# code snippets that goes along with the 1_implicit.ipynb notebook
import os
import subprocess
import numpy as np
import pandas as pd


def create_rating_mat(file_dir):
    """create movielens rating matrix"""

    # download the dataset if it isn't in the same folder
    file_path = os.path.join(file_dir, 'u.data')
    if not os.path.isdir(file_dir):
        subprocess.call(['curl', '-O', 'http://files.grouplens.org/datasets/movielens/' + file_dir + '.zip'])
        subprocess.call(['unzip', file_dir + '.zip'])

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep = '\t', names = names)

    # create the rating matrix r_{ui}, remember to
    # subract the user and item id by 1 since
    # the indices starts from 0
    n_users = df['user_id'].unique().shape[0]
    n_items = df['item_id'].unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples(index = False):
        ratings[row.user_id - 1, row.item_id - 1] = row.rating

    return ratings


def create_train_test(file_dir):
    """
    split into training and test sets,
    remove 10 ratings from each user
    and assign them to the test set
    """
    ratings = create_rating_mat(file_dir)
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_index = np.random.choice( np.flatnonzero(ratings[user]), 
                                       size = 10, replace = False )
        train[user, test_index] = 0.0
        test[user, test_index] = ratings[user, test_index]
        
    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return train, test


if __name__ == '__main__':
    file_dir = 'ml-100k'
    train, test = create_train_test(file_dir)
    train.head()

