from dataclasses import dataclass
import numpy as np
import torch
import os

CPU = torch.device("cpu")
GPU = torch.device("cuda:0")


def get_device() -> torch.device:
    env_torch_device = os.environ.get("TORCH_DEVICE")
    if env_torch_device is not None:
        if env_torch_device == "cpu":
            return CPU
        elif env_torch_device == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cannot run on GPU: cuda not available.")
            return GPU
        else:
            raise ValueError(
                "If set, environment variable TORCH_DEVICE must be either 'cpu' or 'gpu'."  # noqa: E501
            )

    return GPU if torch.cuda.is_available() else CPU


device = get_device()


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class ExperienceData:
    priority: float
    probability: float
    weight: float
    index: int


def _to_device(
        values,
        dtype,
        to_uint8: bool = False,
) -> torch.Tensor:
    stacked_values = np.vstack(values)
    if to_uint8:
        stacked_values = stacked_values.astype(np.uint8)
    tensor = torch.from_numpy(stacked_values)
    if dtype == "float":
        tensor = tensor.float()
    elif dtype == "long":
        tensor = tensor.long()
    else:
        raise ValueError("Wrong dtype")
    return tensor.to(device)


def send_experiences_to_device(experiences):
    return (
        _to_device([e.state for e in experiences], dtype="float"),
        _to_device([e.action for e in experiences], dtype="long"),
        _to_device([e.reward for e in experiences], dtype="float"),
        _to_device([e.next_state for e in experiences], dtype="float"),
        _to_device([e.done for e in experiences],
                   dtype="float", to_uint8=True),
    )
