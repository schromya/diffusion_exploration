import numpy as np
import zarr


def combine_zarrs(paths: list[str], output_path: str):
    """
    Combines multiple zarr stores into one.

    Each zarr store must have the structure:
    data/
        state     # (N, 5)
        action    # (N, 2)
    meta/
        episode_ends  # (num_episodes,)
    """
    all_states = []
    all_actions = []
    all_episode_ends = []
    offset = 0

    for path in paths:
        z = zarr.open(path, mode='r')
        all_states.append(z['data']['state'][:])
        all_actions.append(z['data']['action'][:])
        all_episode_ends.append(z['meta']['episode_ends'][:] + offset)
        offset += len(z['data']['state'])

    out = zarr.open(output_path, mode='w')
    out.require_group('data')
    out.require_group('meta')

    out['data']['state'] = np.concatenate(all_states)
    out['data']['action'] = np.concatenate(all_actions)
    out['meta']['episode_ends'] = np.concatenate(all_episode_ends)


if __name__ == "__main__":
    ZARR_PATHS = [
        "pusht_trained_state_data.zarr",
        "pusht_trained_state_data_2.zarr",
    ]
    OUTPUT_PATH = "pusht_trained_state_data_100e.zarr"
    combine_zarrs(ZARR_PATHS, OUTPUT_PATH)