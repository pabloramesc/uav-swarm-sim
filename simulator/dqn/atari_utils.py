"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike


def preprocess_frame(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB frame by converting it to grayscale, resizing, and cropping.

    Parameters
    ----------
    rgb_frame : np.ndarray
        The input RGB frame with shape (height, width, 3).

    Returns
    -------
    np.ndarray
        The preprocessed grayscale frame with shape (84, 84).
    """
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 110))
    cropped_frame = resized_frame[18:102, :]
    return cropped_frame


def normalize_frame(uint8_frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame by scaling pixel values to the range [0, 1].

    Parameters
    ----------
    uint8_frame : np.ndarray
        The input frame with dtype uint8 and pixel values in the range [0, 255].

    Returns
    -------
    np.ndarray
        The normalized frame with dtype float32 and pixel values in the range [0, 1].
    """
    if uint8_frame.dtype != np.uint8:
        return uint8_frame
    normalized_frame = (uint8_frame / 255.0).astype(np.float32)
    return normalized_frame


class AtariPreprocessor:
    """
    Preprocessor for Atari frames.

    Handles frame preprocessing, stacking, and maintaining a fixed-size stack of frames.
    """

    def __init__(self, stack_size=4) -> None:
        """
        Initialize an AtariPreprocessor instance.

        Parameters
        ----------
        stack_size : int, default=4
            The number of frames to stack.
        """
        self.stack_size = stack_size
        self.frames = None

    def get_state(self) -> np.ndarray:
        """
        Get the current stacked state.

        Returns
        -------
        np.ndarray
            The stacked state with shape (84, 84, stack_size).
        """
        return np.stack(self.frames, axis=-1)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """
        Reset the preprocessor with the initial frame.

        Parameters
        ----------
        frame : np.ndarray
            The initial frame to preprocess and stack.

        Returns
        -------
        np.ndarray
            The stacked state after resetting.
        """
        processed_frame = preprocess_frame(frame)
        self.frames = [processed_frame] * self.stack_size
        return self.get_state()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a new frame and update the stacked state.

        Parameters
        ----------
        frame : np.ndarray
            The new frame to preprocess.

        Returns
        -------
        np.ndarray
            The updated stacked state.
        """
        if len(self.frames) != self.stack_size:
            self.reset(frame)
            return self.get_state()
        processed_frame = preprocess_frame(frame)
        self.frames.pop(0)
        self.frames.append(processed_frame)
        return self.get_state()


class VectorizedAtariPreprocessor:
    """
    Vectorized preprocessor for multiple Atari environments.

    Manages multiple AtariPreprocessor instances for parallel environments.
    """

    def __init__(self, num_envs: int, stack_size=4) -> None:
        """
        Initialize a VectorizedAtariPreprocessor instance.

        Parameters
        ----------
        num_envs : int
            The number of parallel environments.
        stack_size : int, default=4
            The number of frames to stack for each environment.
        """
        self.num_envs = num_envs
        self.stack_size = stack_size
        self.preprocessors: list[AtariPreprocessor] = []
        for _ in range(self.num_envs):
            self.preprocessors.append(AtariPreprocessor(self.stack_size))

    def get_states(self) -> np.ndarray:
        """
        Get the current stacked states for all environments.

        Returns
        -------
        np.ndarray
            The stacked states with shape (num_envs, 84, 84, stack_size).
        """
        states = np.stack([p.get_state() for p in self.preprocessors], axis=-1)
        return states

    def reset(self, frames: ArrayLike) -> np.ndarray:
        """
        Reset all preprocessors with the initial frames.

        Parameters
        ----------
        frames : ArrayLike
            The initial frames for all environments.

        Returns
        -------
        np.ndarray
            The stacked states after resetting.
        """
        states = np.stack(
            [p.reset(frame) for p, frame in zip(self.preprocessors, frames)], axis=0
        )
        return states

    def preprocess(self, frames: ArrayLike) -> np.ndarray:
        """
        Preprocess new frames and update the stacked states for all environments.

        Parameters
        ----------
        frames : ArrayLike
            The new frames for all environments.

        Returns
        -------
        np.ndarray
            The updated stacked states.
        """
        states = np.stack(
            [p.preprocess(frame) for p, frame in zip(self.preprocessors, frames)],
            axis=0,
        )
        return states


def plot_prepocessed_frame(state: np.ndarray):
    """
    Plot the stacked frames of a preprocessed Atari state.

    Parameters
    ----------
    state : np.ndarray
        The stacked state with shape (84, 84, 4).

    Raises
    ------
    ValueError
        If the state does not have the expected shape (84, 84, 4).
    """
    if state.shape != (84, 84, 4):
        raise ValueError("Atari prepocessed frames must be (84, 84, 4) shaped")

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    for i, ax in enumerate(axes):
        ax.imshow(state[:, :, i])
        ax.set_title(f"Frame {i} (84x84x1)")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make("ALE/Breakout-v5")
    rgb_state = env.reset()[0]

    # preprocess frame function
    gray_state = cv2.cvtColor(rgb_state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 110))
    cropped_state = resized_state[18:102, :]
    normalized_state = (cropped_state / 255.0).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    images = [
        (rgb_state, "RGB State (210x160x3)"),
        (gray_state, "Gray State (210x160x1)"),
        (resized_state, "Resized State (110x84x1)"),
        (cropped_state, "Cropped State (84x84x1)"),
    ]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    frame_preprocessor = AtariPreprocessor()
    frame = env.reset()[0]
    state = frame_preprocessor.reset(frame)
    for i in range(100):
        print(f"Step {i}")
        action = env.action_space.sample()
        frame, reward, done, trunc, info = env.step(action)
        state = frame_preprocessor.preprocess(frame)
        if i >= 4:
            plot_prepocessed_frame(state)

    env.close()
