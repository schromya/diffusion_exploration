import numpy as np                                                                                                                                                                                                                                                                                
import pygame                                                                                                                                                                                                                                                                                     
from env.PushTEnv import PushTEnv
import zarr


def save_to_zarr(path:str, observations, actions, episode_ends):
    """
    Saves the data in a zarr file in this format:
    data/
        state     # (N, 5)  (agent_x, agent_y, block_x, block_y, block_angle)
        action    # (N, 2)   (mouse_x, mouse_y)
    meta/
        episode_ends  # (num_episodes,) - index of last step+1 per episode
    """
    root = zarr.open(path, mode='w')
    root.require_group('data')
    root.require_group('meta')

    root['data']['state'] = np.array(observations)
    root['data']['action'] = np.array(actions)
    root['meta']['episode_ends'] = np.array(episode_ends)


NUM_EPISODES = 5
PATH = "pusht_trained_state_data.zarr"

env = PushTEnv()
agent = env.teleop_agent()


all_observations = []
all_actions = []
episode_ends = []

for episode in range(NUM_EPISODES):
    env.seed(episode)
    obs, _ = env.reset()
    env.render(mode='human')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.act(obs)
        if action is not None:
            obs, reward, done, _, info = env.step(action)
            all_observations.append(obs)
            all_actions.append(np.array(action))
            env.render(mode='human')

            if done:
                break

    episode_ends.append(len(all_observations))
    save_to_zarr(PATH, all_observations, all_actions, episode_ends)
    print(f"Saved episode {episode + 1}/{NUM_EPISODES}")

env.close()

