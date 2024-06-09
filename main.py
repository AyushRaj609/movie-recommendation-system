import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def load_movielens_1m():
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    dataset_path = tf.keras.utils.get_file('ml-1m.zip', url, extract=True)
    dataset_dir = os.path.join(os.path.dirname(dataset_path), 'ml-1m')
    ratings = pd.read_csv(os.path.join(dataset_dir, 'ratings.dat'),
                          sep='::',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          engine='python',
                          encoding='ISO-8859-1')
    movies = pd.read_csv(os.path.join(dataset_dir, 'movies.dat'),
                         sep='::',
                         names=['movie_id', 'title', 'genres'],
                         engine='python',
                         encoding='ISO-8859-1')
    users = pd.read_csv(os.path.join(dataset_dir, 'users.dat'),
                        sep='::',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                        engine='python',
                        encoding='ISO-8859-1')
    return ratings, movies, users


ratings, movies, users = load_movielens_1m()

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['user_id'])
ratings['movie'] = movie_encoder.fit_transform(ratings['movie_id'])

n_users = ratings['user'].nunique()
n_movies = ratings['movie'].nunique()

X = ratings[['user', 'movie']].values
y = ratings['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


embedding_size = 50

user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

user_embedding = Embedding(input_dim=n_users, output_dim=embedding_size, name='user_embedding')(user_input)
movie_embedding = Embedding(input_dim=n_movies, output_dim=embedding_size, name='movie_embedding')(movie_input)

user_vecs = Flatten()(user_embedding)
movie_vecs = Flatten()(movie_embedding)

input_vecs = Concatenate()([user_vecs, movie_vecs])

x = Dense(128, activation='relu')(input_vecs)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1)(x)

model = Model([user_input, movie_input], output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping]
)


def recommend_movies(user_id, num_recommendations=10):
    user_idx = user_encoder.transform([user_id])[0]
    movie_ids = np.arange(n_movies)
    user_ids = np.full(n_movies, user_idx)

    predictions = model.predict([user_ids, movie_ids])
    predictions = predictions.flatten()

    top_movie_indices = predictions.argsort()[-num_recommendations:][::-1]
    recommended_movie_ids = movie_encoder.inverse_transform(top_movie_indices)

    return movies[movies['movie_id'].isin(recommended_movie_ids)]


user_id = 1
recommendations = recommend_movies(user_id)
print(f"Recommendations for User {user_id}:")
print(recommendations[['movie_id', 'title']])
