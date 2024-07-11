
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import defaultdict
import random

# Load the new dataset
DATA_DIR = "../data/"
SONGS_FILE = "songs.csv"
NFEATURE = 21  # Number of Features
songs = pd.read_csv(DATA_DIR + SONGS_FILE, index_col=0)

# Extract relevant features
FEATURES = ["(1980s)","(1990s)","(2000s)","(2010s)","(2020s)","Pop","Rock","Counrty","Folk","Dance","Grunge","Love","Metal","Classic","Funk","Electric","Acoustic","Indie","Jazz","SoundTrack","Rap"]


# Normalize the features (if necessary)
# songs[FEATURES] = songs[FEATURES].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

class SongRecommendationEnv(gym.Env):
    def __init__(self, songs, user_features):
        super(SongRecommendationEnv, self).__init__()
        self.songs = songs
        self.user_features = user_features
        self.action_space = spaces.Discrete(len(songs))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(user_features),), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 10  # Number of songs to recommend in each episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.user_features.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1
        song_features = self.songs.iloc[action][FEATURES].values.astype(np.float32)
        reward = self._get_reward(song_features)
        done = self.current_step >= self.max_steps
        terminated = done  # This indicates if the episode is over
        truncated = False  # This is not used in this example, set it to False
        info = {}
        return self.user_features.astype(np.float32), reward, terminated, truncated, info

    def _get_reward(self, song_features):
        # Simple reward based on the dot product of user features and song features
        return np.dot(self.user_features, song_features)

# Define initial user features (can be adjusted as needed)
user_features = np.random.rand(len(FEATURES))

# Create the environment
env = SongRecommendationEnv(songs, user_features)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon-greedy exploration factor
num_episodes = 1000

# Initialize Q1 and Q2 tables
Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
Q2 = defaultdict(lambda: np.zeros(env.action_space.n))

# Double Q-Learning training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection based on Q1 or Q2
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            if random.random() < 0.5:
                action = np.argmax(Q1[tuple(state)])
            else:
                action = np.argmax(Q2[tuple(state)])
        
        next_state, reward, done, _, _ = env.step(action)
        
        # Update Q1 and Q2
        if random.random() < 0.5:
            best_next_action = np.argmax(Q1[tuple(next_state)])
            Q1[tuple(state)][action] += alpha * (reward + gamma * Q2[tuple(next_state)][best_next_action] - Q1[tuple(state)][action])
        else:
            best_next_action = np.argmax(Q2[tuple(next_state)])
            Q2[tuple(state)][action] += alpha * (reward + gamma * Q1[tuple(next_state)][best_next_action] - Q2[tuple(state)][action])
        
        state = next_state

# After training, use Q1 or Q2 to recommend songs
def recommend_song(user_features, Q):
    action = np.argmax(Q[tuple(user_features)])
    return songs.iloc[action].name

# Test the recommendation after training
recommended_song = recommend_song(user_features, Q1)  # or Q2

print(f"Recommended song: {recommended_song} ")
