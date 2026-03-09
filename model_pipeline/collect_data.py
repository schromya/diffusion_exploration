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

    root['data']['state'] = np.array(observations, dtype=np.float32)                                                                                                                                                                                                                              
    root['data']['action'] = np.array(actions, dtype=np.float32)
    root['meta']['episode_ends'] = np.array(episode_ends)


def collect_data(num_episodes, path, control_hz):
    env = PushTEnv()
    agent = env.teleop_agent()


    all_observations = []
    all_actions = []
    episode_ends = []
    clock = pygame.time.Clock()

    for episode in range(num_episodes):
        env.seed(episode)
        obs, _ = env.reset()
        env.render(mode='human')

        running = True
        while running:
            clock.tick(control_hz)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = agent.act(obs)
            if action is not None:
                all_observations.append(obs)
                obs, reward, done, _, info = env.step(action)
                all_actions.append(np.array(action))
                env.render(mode='human')

                if done:
                    break

        episode_ends.append(len(all_observations))
        save_to_zarr(path, all_observations, all_actions, episode_ends)
        print(f"Saved episode {episode + 1}/{NUM_EPISODES}")

    env.close()


if __name__ == "__main__":
    CONTROL_HZ = 15
    NUM_EPISODES = 50
    PATH = "pusht_trained_state_data.zarr"
    collect_data(NUM_EPISODES, PATH, CONTROL_HZ)