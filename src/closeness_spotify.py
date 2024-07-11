import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

SONGS_FILE = "src/spotify.csv"

def process_chunk(chunk, model, user_index):
    chunk.index = chunk["track_name"]
    chunk_data = chunk[['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    
    model.fit(chunk_data)
    
    # Ensure the user_index is within the range of the current chunk
    if user_index < len(chunk_data):
        top_songs = model.evaluate(user_index)[1:6]  # Skip the first one as it is the same song
        print("Top 5 Songs closest to '{0}' are: \n{1}".format(index_to_instance(chunk_data, user_index), pd.Series(top_songs)))
    else:
        print(f"User index {user_index} is out of range for the current chunk.")

def index_to_instance(df, index=None):
    if index is not None:
        return df.index[index]
    else:
        return list(df.index)

class RecSysContentBased():
    def __init__(self):
        pass
    
    def fit(self, train):
        self.train_set = train
        self.similarity = cosine_similarity(train)
        self.distances = pairwise_distances(train, metric='euclidean')
    
    def evaluate(self, user_index):
        distances = sorted(list(enumerate(self.distances[user_index])), key=lambda x: x[1])
        return [index_to_instance(self.train_set, d[0]) for d in distances]
    
    def predict(self):
        pass
    
    def test(self, testset):
        pass

# Initialize the model
model = RecSysContentBased()
user_index = 10  # The index of the user/song to find recommendations for

# Load and process the dataset in chunks
chunk_size = 10000
for chunk in pd.read_csv(SONGS_FILE, chunksize=chunk_size):
    process_chunk(chunk, model, user_index)
