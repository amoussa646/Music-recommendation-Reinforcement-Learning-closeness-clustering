import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

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
        song_features = self.songs.iloc[action][-NFEATURE:].values.astype(np.float32)
        reward = self._get_reward(song_features)
        done = self.current_step >= self.max_steps
        terminated = done  # This indicates if the episode is over
        truncated = False  # This is not used in this example, set it to False
        info = {}
        return self.user_features.astype(np.float32), reward, terminated, truncated, info

    def _get_reward(self, song_features):
        # Simple reward based on the dot product of user features and song features
        return np.dot(self.user_features, song_features)

# Load songs data
DATA_DIR = "../data/"
SONGS_FILE = "songs.csv"
NFEATURE = 21  # Number of Features
songs = pd.read_csv(DATA_DIR + SONGS_FILE, index_col=0)

# Define initial user features (can be adjusted as needed)
user_features = np.random.rand(NFEATURE)

# Create the environment
env = SongRecommendationEnv(songs, user_features)



from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Check if the environment is valid
check_env(env)

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_song_recommender")


# Load the trained model
model = PPO.load("ppo_song_recommender")

# Test the trained agent
obs, _ = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    print(f"Recommended song: {songs.iloc[action].name}, Reward: {rewards}")
    if dones:
        break


