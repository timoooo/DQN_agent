# Import Required Packages
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .experience import Experience, send_experiences_to_device, get_device
from .nn import DQN
from .priortized_replay_mem import PrioritizedExperienceReplayBuffer

device = get_device()


class DQNAgent:
    network: DQN

    def __init__(
            self,
            dqn_type: str = "DDQN",
            action_size: int = 5,
            state_size: int = 5,
            use_per: bool = True,
            replay_memory_size: float = 1e5,
            batch_size: int = 64,
            gamma: float = 0.99,
            learning_rate: float = 1e-3,
            target_update_rate: int = 100,
            update_rate: int = 4,
            current_step: int = 0,
            dueling: bool = False,
            alpha_per: float = 0.5,
            noisy_net: bool = False,

    ):
        """
        Initialize Agent, including:
            DQN Hyperparameters
            Local and Targat State-Action Policy Networks
            Replay Memory Buffer from Replay Buffer Class (define below)

        DQN Agent Parameters
        ======
        - state_size (int): dimension of each state
        - action_size (int): dimension of each action
        - replay_memory size (int): size of the replay memory buffer (typically 5e4
                                    to 5e6)
        - batch_size (int): size of the memory batch used for model updates (typically
                           32, 64 or 128)
        - gamma (float): set the discount ted value of future rewards (typically .95
                         to .995)
        - learning_rate (float): specifies the rate of model learing (typically
                                 1e-4 to 1e-3))
        """
        self.dueling = dueling
        self.use_per = use_per
        self.dqn_type = dqn_type
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.update_rate = update_rate
        self.target_update_rate = target_update_rate
        self.noisy_net = noisy_net
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.current_step = current_step

        # Initialize networks
        self.network = DQN(self.state_size, self.action_size, dueling_architecture=self.dueling,
                           noisy_net=self.noisy_net).to(device)
        self.target_network = DQN(self.state_size, self.action_size, dueling_architecture=self.dueling,
                                  noisy_net=self.noisy_net).to(device)

        # initially sync network and then sync according to target_update_rate
        self._sync_networks()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        if self.use_per:
            self.memory = PrioritizedExperienceReplayBuffer(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                alpha=alpha_per)
        else:
            self.memory = PrioritizedExperienceReplayBuffer(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                alpha=0)

    def act(
            self, state: np.ndarray, eps: float = 0.0
    ):
        """Return action for given state as per current policy.

        AGV id is always implicit for trained agents.

        Set epsilon to zero for greedy policy.
        """
        torch_state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.network.eval()
        with torch.no_grad():
            action_values = self.network(torch_state)
        self.network.train()

        # if noisy net dont use epsilon greedy strategy
        if self.noisy_net:
            return int(np.argmax(action_values.cpu().data.numpy()))
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Select action with highest estimated q value for action
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            # Exploration of the environment
            return np.random.choice(np.arange(self.action_size))

    def has_enough_experience(self):
        return len(self.memory) > self.batch_size

    def step(self, experience: Experience) -> None:
        """Remember a new experience (state-action-reward pair)."""
        # calculate error before adding to the replay memory
        self.memory.add_to_memory(experience)
        self.current_step += 1

    def learn(self, beta: float = 0) -> None:
        """Update value parameters using a sample from the memory."""

        if self.current_step % self.update_rate != 0:
            # Learn from memory only if enough steps since last learning
            return None

        if self.use_per:
            # sample with beta
            idxes, experiences, weights = self.memory.sample(beta)
        else:
            # set beta to 0 to have uniform sampling
            _, experiences, _ = self.memory.sample(0)

        # Send to device
        states, actions, rewards, next_states, dones = send_experiences_to_device(
            experiences
        )

        if self.dqn_type == "DDQN":
            # Double DQN
            # ************************
            Q_values = self.network(states).gather(1, actions)

            _, actions = self.network(next_states).max(1, True)

            next_Q_values = self.target_network(next_states).gather(1, actions)
            target_Q_values = rewards + (self.gamma * next_Q_values * (1 - dones))

            # calculate delta
            if self.use_per:
                deltas = target_Q_values - Q_values
                priorities = deltas.abs().cpu().detach().numpy().flatten()

                self.memory.update_priorities(idxes, priorities + 1e-6)  # priorities must be positive
                # calculate new loss with weights
                # basically MSE
                _weights = (torch.Tensor(weights).view((-1, 1))).to(device)
                loss = torch.mean((deltas * _weights) ** 2)
            # if no per calculate loss normally
            else:
                loss = F.mse_loss(Q_values, target_Q_values)
        else:
            # Regular (Vanilla) DQN
            # ************************
            # Get max Q values for (s',a') from target model
            Q_values = self.network(states).gather(1, actions)
            next_Qs = self.target_network(next_states).detach()
            next_Q_values = next_Qs.max(1)[0].unsqueeze(1)

            target_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)

            if self.use_per:
                deltas = target_Q_values - Q_values
                priorities = deltas.abs().cpu().detach().numpy().flatten()

                self.memory.update_priorities(idxes, priorities + 1e-6)  # priorities must be positive
                # calculate new loss with weights
                # weights are not needed for DQN according to the paper
            loss = F.mse_loss(Q_values, target_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.noisy_net:
            self.network.reset_noise()
            self.target_network.reset_noise()
        # ------------------- update target network ------------------- #
        if self.current_step % self.target_update_rate == 0:
            self._sync_networks()

    def save_network(self, weights_path: Path) -> None:
        """Save Agent weights to file."""
        torch.save(self.network.state_dict(), weights_path)

    def load_network(self, weights_path: Path) -> None:
        """Load Agent weights."""
        self.network.load_state_dict(torch.load(weights_path))

    def _sync_networks(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def _soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        # tau 1e-3 means that 1% of the params are copied to the target network
        tau = 1e-3
        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

