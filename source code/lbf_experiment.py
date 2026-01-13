import io
import os
import pickle
import random
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import lbforaging
import imageio.v2 as imageio
import pyglet
from iql import IQL
from cql import CQL


CONFIG = {
    "seed": 0,
    "gamma": 0.95,
    "total_eps": 50000,
    "max_ep_len": 50,
    "eval_freq": 500,
    "lr": 0.05,
    "init_epsilon": 1.0,
    "eval_epsilon": 0.05,
}


ENV_IDS = {
    "non_coop": "Foraging-5x5-2p-1f-v3",
    "coop": "Foraging-5x5-2p-1f-coop-v3",
}

CHECKPOINT_DIR = "checkpoints"


def make_env(env_key: str, render_mode: str = None):
    # Make the environment
    env_id = ENV_IDS[env_key]
    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **kwargs)
    return env


def preprocess_obs(obs):
    # Convert numpy arrays to tuples of integers, so can be used as keys in dictionaries (for the Q-table)
    return tuple(tuple(map(int, o)) for o in obs)



# ===============================
# Util functions for checkpoints (to avoid retraining from scratch and for faster iteration)
# (functions created with the help of Gemini)
# ===============================

def get_checkpoint_path(algorithm: str, env_key: str) -> Tuple[str, str]:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    returns_path = os.path.join(CHECKPOINT_DIR, f"{algorithm}_{env_key}_returns.pkl")
    agent_path = os.path.join(CHECKPOINT_DIR, f"{algorithm}_{env_key}_agent.pkl")
    return returns_path, agent_path


def save_checkpoint(returns: List[float], agent, algorithm: str, env_key: str):
    returns_path, agent_path = get_checkpoint_path(algorithm, env_key)
    with open(returns_path, "wb") as f:
        pickle.dump(returns, f)
    
    # Convert defaultdicts to regular dicts for pickle compatibility
    # Don't save action_spaces - we'll recreate env when loading
    if algorithm == "IQL":
        checkpoint_data = {
            "q_tables": [dict(q_table) for q_table in agent.q_tables],
            "num_agents": agent.num_agents,
            "gamma": agent.gamma,
            "learning_rate": agent.learning_rate,
            "epsilon": agent.epsilon,
            "env_key": env_key,
        }
    else:  # CQL
        checkpoint_data = {
            "q_table": dict(agent.q_table),
            "num_agents": agent.num_agents,
            "gamma": agent.gamma,
            "learning_rate": agent.learning_rate,
            "epsilon": agent.epsilon,
            "joint_actions": agent.joint_actions,
            "num_joint_actions": agent.num_joint_actions,
            "env_key": env_key,
        }
    
    with open(agent_path, "wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(algorithm: str, env_key: str) -> Optional[Tuple[List[float], object]]:
    returns_path, agent_path = get_checkpoint_path(algorithm, env_key)
    if os.path.exists(returns_path) and os.path.exists(agent_path):
        with open(returns_path, "rb") as f:
            returns = pickle.load(f)
        with open(agent_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        # Recreate environment to get action_spaces
        env = make_env(checkpoint_data["env_key"])
        
        # Reconstruct agent from checkpoint data
        if algorithm == "IQL":
            agent = IQL(
                num_agents=checkpoint_data["num_agents"],
                action_spaces=env.action_space,
                gamma=checkpoint_data["gamma"],
                learning_rate=checkpoint_data["learning_rate"],
                epsilon=checkpoint_data["epsilon"],
            )
            # Restore Q-tables (convert back to defaultdicts)
            from collections import defaultdict
            agent.q_tables = [
                defaultdict(lambda: 0, q_table_dict)
                for q_table_dict in checkpoint_data["q_tables"]
            ]
        else:  # CQL
            agent = CQL(
                num_agents=checkpoint_data["num_agents"],
                action_spaces=env.action_space,
                gamma=checkpoint_data["gamma"],
                learning_rate=checkpoint_data["learning_rate"],
                epsilon=checkpoint_data["epsilon"],
            )
            # Restore Q-table (convert back to defaultdict)
            from collections import defaultdict
            agent.q_table = defaultdict(lambda: 0, checkpoint_data["q_table"])
            agent.joint_actions = checkpoint_data["joint_actions"]
            agent.num_joint_actions = checkpoint_data["num_joint_actions"]
        
        env.close()
        return returns, agent
    return None

def smooth(vals: List[float], window: int) -> List[float]:
    return [np.mean(vals[i-window:i]) for i in range(window, len(vals))]

# =======================================


def train_iql_lbf(env_key: str, config: Dict):
    # Load checkpoint if it exists
    checkpoint = load_checkpoint("IQL", env_key)
    if checkpoint:
        print(f"Loading checkpoint for IQL-{env_key}")
        return checkpoint

    # Otherwise, train from scratch
    print(f"Training IQL on {env_key}")
    env = make_env(env_key)
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    # Create the agents
    agents = IQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    # Train the agents
    episode_returns = []
    step_counter = 0
    max_steps = config["total_eps"] * config["max_ep_len"]

    for _ in tqdm(range(config["total_eps"])):
        obss, _ = env.reset()
        obss = preprocess_obs(obss) # For correct formatting for the Q-table
        done = False
        ep_ret = np.zeros(env.n_agents)
        t = 0

        while not done and t < config["max_ep_len"]:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, terminated, truncated, _ = env.step(acts)
            n_obss = preprocess_obs(n_obss)
            done = terminated or truncated
            agents.learn(obss, acts, rewards, n_obss, done)

            ep_ret += rewards
            obss = n_obss
            step_counter += 1
            t += 1

        episode_returns.append(float(ep_ret.mean()))

    env.close()
    save_checkpoint(episode_returns, agents, "IQL", env_key)
    return episode_returns, agents


def train_cql_lbf(env_key: str, config: Dict):
    # Load checkpoint if it exists
    checkpoint = load_checkpoint("CQL", env_key)
    if checkpoint:
        print(f"Loading checkpoint for CQL-{env_key}")
        return checkpoint

    # Otherwise, train from scratch
    print(f"Training CQL on {env_key}")
    env = make_env(env_key)
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    # Create the agent
    agent = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    # Train the agent
    episode_returns = []
    step_counter = 0
    max_steps = config["total_eps"] * config["max_ep_len"]

    for _ in tqdm(range(config["total_eps"])):
        obss, _ = env.reset()
        obss = preprocess_obs(obss) # For correct formatting for the Q-table
        done = False
        ep_ret = np.zeros(env.n_agents)
        t = 0

        while not done and t < config["max_ep_len"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            acts = agent.act(obss)
            n_obss, rewards, terminated, truncated, _ = env.step(acts)
            n_obss = preprocess_obs(n_obss)
            done = terminated or truncated
            agent.learn(obss, acts, rewards, n_obss, done)

            ep_ret += rewards
            obss = n_obss
            step_counter += 1
            t += 1

        episode_returns.append(float(ep_ret.mean()))

    env.close()
    save_checkpoint(episode_returns, agent, "CQL", env_key)
    return episode_returns, agent


def plot_returns(
    returns: Dict[str, List[float]],
    title: str,
    filename: str,
    window: int = 50
):
    # Plot the returns for each agent
    plt.figure(figsize=(5, 3))
    for label, vals in returns.items():
        sm = smooth(vals, window)
        plt.plot(sm, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Mean return per episode")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.close()


def record_gif(agent, env_id, filename):
    # Function to record a GIF of the trained agent (function created with the help of Gemini)
    env = gym.make(env_id, render_mode=None)
    
    # Force greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    frames = []
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    env.unwrapped.render_mode = "human" # Enable rendering here (doing it before gave weitd error)
    env.render()
    time.sleep(0.5) # wait so window has time to open

    done = False
    step = 0
    while not done and step < 50:
        # get an action from the agent
        actions = agent.act(obs)
        obs, rewards, done, truncated, info = env.step(actions)
        obs = preprocess_obs(obs)
        env.render()
        
        # get a frame
        try:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
            frames.append(arr)
        except Exception:
            pass

        if done or truncated:
            print(f"Episode finished at step {step+1} with reward {sum(rewards)}")
            break
        
        step += 1

    try:
        env.close()
    except (AttributeError, Exception):
        # Ignore pyglet window closing errors on macOS
        pass
    agent.epsilon = original_epsilon

    # Save GIF
    if len(frames) > 0:
        imageio.mimsave(
            filename, 
            frames, 
            fps=4, 
            loop=0
        )
        print(f"Saved {filename}")


def main():
    all_returns = {}

    # IQL non-coop and coop
    iql_non_coop_returns, iql_non_coop_agent = train_iql_lbf("non_coop", CONFIG)
    all_returns[("IQL", "non_coop")] = iql_non_coop_returns

    iql_coop_returns, iql_coop_agent = train_iql_lbf("coop", CONFIG)
    all_returns[("IQL", "coop")] = iql_coop_returns

    # CQL non-coop and coop
    cql_non_coop_returns, cql_non_coop_agent = train_cql_lbf("non_coop", CONFIG)
    all_returns[("CQL", "non_coop")] = cql_non_coop_returns

    cql_coop_returns, cql_coop_agent = train_cql_lbf("coop", CONFIG)
    all_returns[("CQL", "coop")] = cql_coop_returns

    # Plot returns for each one
    plot_returns(
        {
            "IQL": all_returns[("IQL", "non_coop")],
            "CQL": all_returns[("CQL", "non_coop")],
        },
        title="LBF Foraging-5x5-2p-1f-v3",
        filename="evaluation_returns_non_coop",
    )

    plot_returns(
        {
            "IQL": all_returns[("IQL", "coop")],
            "CQL": all_returns[("CQL", "coop")],
        },
        title="LBF Foraging-5x5-2p-1f-coop-v3",
        filename="evaluation_returns_coop",
    )

    # Record videos for each trained agent
    record_gif(iql_non_coop_agent, ENV_IDS["non_coop"], "iql_non_coop.gif")
    record_gif(iql_coop_agent, ENV_IDS["coop"], "iql_coop.gif")
    record_gif(cql_non_coop_agent, ENV_IDS["non_coop"], "cql_non_coop.gif")
    record_gif(cql_coop_agent, ENV_IDS["coop"], "cql_coop.gif")


if __name__ == "__main__":
    main()
