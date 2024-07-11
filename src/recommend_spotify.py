import pandas as pd
import numpy as np
import random
import math


NFEATURE = 11  # Number of Features specific to the new dataset
S = 50  # Hyper Parameter
totReco = 0  # Number of total recommendations till now
startConstant = 5  # for low penalty in starting phase

### Read data
DATA_DIR = "../data/"
SONGS_FILE = "spotify.csv"
Songs = pd.read_csv(SONGS_FILE)

ratedSongs = set()

def compute_utility(user_features, song_features, epoch, s=S):
    """ Compute utility U based on user preferences and song preferences """
    user_features = user_features.copy()
    song_features = song_features.copy()
    dot = np.dot(user_features, song_features)
    ee = (1.0 - 1.0*math.exp(-1.0*epoch/s))
    res = dot * ee
    return res
def get_song_features(song):
    if isinstance(song, pd.Series):
        return song[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    elif isinstance(song, pd.DataFrame):
        return get_song_features(song.iloc[0])  # Get features of the first row
    else:
        raise TypeError("{} should be a Series or DataFrame".format(song))
def best_recommendation(user_features, epoch, s):
    global Songs
    Songs = Songs.copy()
    """ Song with highest utility """
    utilities = np.zeros(len(Songs))  # Initialize utilities array
    print(len(Songs))
    print(Songs)
    print(Songs.iterrows())
    print(utilities)
    for i, song in Songs.iterrows():
        song_features = get_song_features(song)
        utility = compute_utility(user_features, song_features, epoch - song['last_t'], s)
        print("i")
        print(i)
        if(i=="last_t"):
            pass
        else:
            utilities[i] = utility  # Assign utility value to utilities array
    
    # Find the index of the song with the highest utility
    best_index = np.argmax(utilities)
    best_song = Songs.iloc[best_index]
    
    return best_song

def all_recommendation(user_features):
    """ Top 10 songs using exploration and exploitation """
    global Songs
    Songs = Songs.copy()
    i = 0
    recoSongs = []
    while i < 10:
        song = greedy_choice_no_t(user_features, totReco, S)
        recoSongs.append(song)
        # Ensure to access the index properly since song is a DataFrame
        song_index = song.index[0]  # Get the index of the first row
        Songs.at[song_index, 'last_t'] = totReco  # Update 'last_t' for the song
        i += 1
    return recoSongs


def random_choice():
    """ Random songs which haven't been rated yet """
    global Songs
    Songs = Songs.copy()
    song = Songs.sample()
    while song.index[0] in ratedSongs:
        song = Songs.sample()
    return song


def greedy_choice(user_features, epoch, s):
    """ Greedy approach to song recommendation """
    global totReco
    epsilon = 1 / math.sqrt(epoch + 1)
    totReco += 1
    if random.random() > epsilon:
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def greedy_choice_no_t(user_features, epoch, s, epsilon=0.3):
    """ Greedy approach with fixed epsilon """
    global totReco
    totReco += 1
    if random.random() > epsilon:
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def iterative_mean(old, new, t):
    """ Compute the new mean """
    t += startConstant
    return ((t - 1) / t) * old + (1 / t) * new

def update_features(user_features, song_features, rating, t):
    """ Update user features based on song rating """
    return iterative_mean(user_features, song_features * rating, float(t) + 1.0)

def reinforcement_learning(s=200, N=5):
    global Songs
    Songs = Songs.copy()
    
    user_features = np.zeros(NFEATURE)
    print("Select song features that you like")
    Features = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    
    for i, feature in enumerate(Features):
        print(f"{i + 1}. {feature}")
    
    choice = "y"
    likedFeat = set()
    
    while choice.lower().strip() == "y":
        num = input("Enter number associated with feature: ")
        likedFeat.add(Features[int(num) - 1])
        choice = input("Do you want to add another feature? (y/n): ").strip()
    
    for i, feature in enumerate(Features):
        if feature in likedFeat:
            user_features[i] = 1.0 / len(likedFeat)
    
    print(f"\nRate the following {N} songs so we can learn your taste.\n")
    
    for t in range(N):
        if t >= 10:
            recommendation = greedy_choice_no_t(user_features, t + 1, s, 0.3)
        else:
            recommendation = greedy_choice(user_features, t + 1, s)
        
        recommendation_features = get_song_features(recommendation)
        track_name = recommendation["track_name"]
        artists = recommendation["artists"]
        
        # Extract the actual track name and artist from the string
        if isinstance(track_name, str) and isinstance(artists, str):
         track_name = track_name.split('    ')[-1]
         artists = artists.split('    ')[-1]
        
        else:
         track_name = recommendation["track_name"].values[0]  # Extract the actual string value
         artists = recommendation["artists"].values[0]


         user_rating = input(f'How much do you like "{  track_name}" by "{  artists}" (1-10): ')
         user_rating = int(user_rating)
         user_rating = 1.0 * user_rating / 10.0
         user_features = update_features(user_features, recommendation_features, user_rating, t)
         utility = compute_utility(user_features, recommendation_features, t, s)
         Songs.at[recommendation.index[0], 'last_t'] = t + 1
         ratedSongs.add(recommendation.index[0])
    

        
    
         return user_features

# Removing votes and rating column from dataframe
arr = np.full(Songs.shape[0], -S)
Songs.insert(0, 'last_t', arr)

user_features = reinforcement_learning()

# UI for song rating
choice = "y"
while choice.lower() == "y":

    print("\nWait...\n")
    recommendations = all_recommendation(user_features)
    print("\nRate songs one by one or leave it blank:")
    for i, music in enumerate(recommendations):
      
    
    
        track_name = music["track_name"]  # Extract the actual string value
        artists = music["artists"]#Extract the actual string value
        
        
        # Extract the actual track name and artist from the string
        if isinstance(track_name, str) and isinstance(artists, str):
         track_name = track_name.split('    ')[-1]
         artists = artists.split('    ')[-1]
        else:
         track_name = music["track_name"].values[0]  # Extract the actual string value
         artists = music["artists"].values[0]
        print(f"{i + 1}. {track_name} by {artists}")
    
        
        if music.index[0] in ratedSongs:
            print(f"Song '{track_name}' already rated.")
            continue
        inn = input(f"Rate '{track_name}' (1-10): ").strip()
        if inn != "":
            ratedSongs.add(music.index[0])
            song_features = get_song_features(music)
            user_features = update_features(user_features, song_features, int(inn), totReco)
    
    choice = input("\nDo you want more recommendations? (y/n): ").strip()
