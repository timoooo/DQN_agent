import typing

import numpy as np

from .experience import Experience


class PrioritizedExperienceReplayBuffer:
    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.5,
                 random_state: np.random.RandomState = None) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0
        self._buffer = np.empty(self._buffer_size, dtype=[(
            "priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState(
        ) if random_state is None else random_state

    def __len__(self) -> int:
        return self._buffer_length

    @property
    def alpha(self):
        return self._alpha

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    def add_to_memory(self, experience: Experience) -> None:
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        return self._buffer_length == 0

    def is_full(self) -> bool:
        return self._buffer_length == self._buffer_size

    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps ** self._alpha / np.sum(ps ** self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)

        experiences = self._buffer["experience"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs]) ** -beta
        normalized_weights = weights / weights.max()

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        self._buffer["priority"][idxs] = priorities
