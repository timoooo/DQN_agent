import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            fc1_units: int = 256,
            fc2_units: int = 128,
            dueling_architecture: bool = False,
            noisy_net: bool = False
    ):
        self.dueling = dueling_architecture
        self.noisy = noisy_net
        super(DQN, self).__init__()
        if self.noisy:
            if self.dueling:
                self.init_linear = nn.Linear(state_size, fc1_units)
                self.noisy1 = NoisyLinear(fc1_units, fc1_units)
                self.fc_value = NoisyLinear(fc1_units, fc2_units)
                self.fc_adv = NoisyLinear(fc1_units, fc2_units)
                self.value = NoisyLinear(fc2_units, 1)
                self.adv = NoisyLinear(fc2_units, action_size)
            else:
                self.init_linear = nn.Linear(state_size, fc1_units)
                self.noisy1 = NoisyLinear(fc1_units, fc1_units)
                self.noisy2 = NoisyLinear(fc1_units, fc2_units)
                self.noisy3 = NoisyLinear(fc2_units, action_size)

        elif self.dueling:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc1_units)
            self.fc_value = nn.Linear(fc1_units, fc2_units)
            self.fc_adv = nn.Linear(fc1_units, fc2_units)
            self.value = nn.Linear(fc2_units, 1)
            self.adv = nn.Linear(fc2_units, action_size)
        else:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noisy:
            if self.dueling:
                x = F.relu(self.init_linear(x))
                x = F.relu(self.noisy1(x))
                val_stream = F.relu(self.fc_value(x))
                val = self.value(val_stream)
                adv_stream = F.relu(self.fc_adv(x))
                adv = F.relu(self.adv(adv_stream))
                avg_adv = torch.mean(adv, dim=1, keepdim=True)
                return val + (adv - avg_adv)
            else:
                x = F.relu(self.init_linear(x))
                x = F.relu(self.noisy1(x))
                x = F.relu(self.noisy2(x))
                return self.noisy3(x)
        elif self.dueling:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            val_stream = F.relu(self.fc_value(x))
            val = self.value(val_stream)

            adv_stream = F.relu(self.fc_adv(x))
            adv = self.adv(adv_stream)
            avg_adv = torch.mean(adv, dim=1, keepdim=True)
            return val + (adv - avg_adv)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def reset_noise(self):
        # reset all noisy layers
        if self.dueling:
            self.noisy1.reset_noise()
            self.fc_value.reset_noise()
            self.fc_adv.reset_noise()
            self.value.reset_noise()
            self.adv.reset_noise()
        else:
            self.noisy1.reset_noise()
            self.noisy2.reset_noise()
            self.noisy3.reset_noise()


def _scale_noise(size):
    x = torch.randn(size)
    return x.sign().mul(x.abs().sqrt())


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.017):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = _scale_noise(self.in_features)
        epsilon_out = _scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(_scale_noise(self.out_features))
