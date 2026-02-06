import pickle as pkl
import glob
import sys
sys.path.insert(0, '../../../')
import os
import time
import numpy as np
from absl import app, flags
from tqdm import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")
try:
    from experiments.mappings import CONFIG_MAPPING
except ImportError:
    print("Warning: Could not import CONFIG_MAPPING. Ensure your project structure is correct.")
    CONFIG_MAPPING = {} 

def load_demos_from_dir(pattern):
    transitions_all = []
    if not ("*" in pattern) and os.path.isfile(pattern):
        pkl_files = [pattern]
    else:
        pkl_files = glob.glob(pattern)
        
    print(f"Found {len(pkl_files)} pkl files")

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
            transitions_all.append(data)

    return transitions_all

def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    pkl_path = "/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/demo_data/flexiv_assembly_24_demos_2026-01-30_15-11-53.pkl"  # 改成你的实际路径
    transitions = load_demos_from_dir(pkl_path)

    print("Total transitions:", len(transitions))

    for i in range(len(transitions)):
        env.reset()
        time.sleep(1)
        for j in range(len(transitions[i])):
            input("enter!")
            t = transitions[i][j]
            # print(transition[0].keys())
            # print(t["actions"])
            env.step(np.array(t["actions"]))
            time.sleep(0.1)
        

if __name__ == "__main__":
    app.run(main)