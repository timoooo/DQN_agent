import os.path
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch

from Agent.agent import DQNAgent
from Agent.experience import Experience

PATH_TO_SAVE = "/home/tmo/Projects/DRL_masterthesis/results/"


def save_to_csv(fname, results):
    try:
        df = pd.DataFrame(results)
        df.to_csv(PATH_TO_SAVE + "csvs/" + fname + ".csv")
    except:
        # if this doesnt work aswell just rm -rf *
        print(fname)
        print("Sadge")


def beta_annealing(n):
    return 1 - np.exp(-1e-2 * n)


def inner_train_loop(agent, env, max_num_episodes, score_to_solve, agent_name):
    # METRICS
    scores = []
    scores_window = deque(maxlen=100)
    # Hyperparameters
    eps = 1.0
    eps_decay = 0.995
    eps_end = 0.01
    i = 1
    results = []
    while True:
        score = 0
        done = False
        # get inital state
        state = env.reset()
        # beta is necessary if PER is used
        beta = beta_annealing(i)
        while not done:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            experience = Experience(
                state=state, action=action, next_state=next_state, reward=reward, done=done)
            agent.step(experience)
            agent.learn(beta)

            state = next_state
            score += reward

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f} Epsilon {} Beta {}'.format(i, np.mean(scores_window), eps, beta),
              end="")
        if i > max_num_episodes:
            torch.save(agent.network.state_dict(), PATH_TO_SAVE +
                       "checkpoints/" + agent_name + '.pth')
            return save_to_csv(fname=agent_name, results=results)
        row = {
            "episode": i,
            "epsilon": eps,
            "score": score,
            "beta": beta
        }
        results.append(row)
        i += 1


if __name__ == '__main__':
    #   key is the environment
    #   value is the score to solve the environment
    env_dict = {"MountainCar-v0": -134,
                "CartPole-v1": 195,
                "LunarLander-v2": 200,
                "Acrobot-v1": -60.00
                }

    # AGENT initializes the agent with the hyperparameters
    # FIXED: batch_size, learning_rate, gamma, target_update_rate, update_rate
    per = [True, False]
    dueling = [True, False]
    dqn_types = ["DDQN", "DQN"]
    noisy_net = [True, False]
    max_num_episodes = 1500
    for env, required_score in env_dict.items():
        # to make the results reproducible seed everything (numpy, torch and the environment)
        gym_env = gym.make(env)

        # for more statistical relevance
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for seed in seeds:
            for use_per in per:
                for use_duel in dueling:
                    for dqn_type in dqn_types:
                        for noisy_expl in noisy_net:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            gym_env.seed(seed)
                            agent_name = "{0}_use_per_{1}_use_duel_{2}_env_{3}_noisy_{4}_seed_{5}_episodes_" \
                                         "{6}_req_scores_{7}".format(str(dqn_type),
                                                                     str(use_per),
                                                                     str(use_duel),
                                                                     str(env),
                                                                     str(noisy_expl),
                                                                     str(seed),
                                                                     str(max_num_episodes),
                                                                     str(required_score))
                            agent = DQNAgent(
                                dqn_type=dqn_type,
                                action_size=gym_env.action_space.n,
                                state_size=gym_env.observation_space.shape[0],
                                learning_rate=1e-3,
                                gamma=0.99,
                                replay_memory_size=1e5,
                                batch_size=64,
                                use_per=use_per,
                                target_update_rate=100,
                                update_rate=4,
                                dueling=use_duel,
                                noisy_net=noisy_expl
                            )
                            if os.path.isfile(PATH_TO_SAVE + agent_name + ".csv"):
                                print("Skipping " + agent_name)
                                print("Warning: File already exists")
                            elif (dqn_type == "DQN" and noisy_expl is True) or (dqn_type == "DQN" and use_duel is True):
                                # skips unnecessary agent types
                                print("Skipping DQN dueling and noisy")
                            else:
                                print(agent_name)
                                inner_train_loop(agent=agent, env=gym_env, score_to_solve=required_score,
                                                 max_num_episodes=max_num_episodes, agent_name=agent_name)
