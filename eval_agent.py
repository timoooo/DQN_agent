import os.path
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch

from Agent.agent import DQNAgent
from Agent.experience import Experience
from utils.torch_gpu_check import check_gpu

ENV = "LunarLander-v2"
SCORE_TO_SOLVE = 200


check_gpu()
