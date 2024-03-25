import numpy as np

if __name__ == "__main__":
    actions_seq = np.load("../datasets/kitchen/actions_seq.npy")
    observations_seq = np.load("../datasets/kitchen/observations_seq.npy")
    all_observations = np.load("../datasets/kitchen/all_observations.npy")
    all_init_pos = np.load("../datasets/kitchen/all_init_qpos.npy")
    all_actions = np.load("../datasets/kitchen/all_actions.npy")
