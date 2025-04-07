"""
Contains classes and functions for creating video animations from PyTorch
tensors.
"""
from os import urandom
from pathlib import Path
from sys import byteorder

from imageio import get_writer
from numpy import float32, ndarray, uint8, zeros
from torch import Tensor, from_numpy


def create_model_seed(width: int, height: int, state_channels: int) -> Tensor:
    """
    Creates a new empty tensor for model consumption with a single active
    automata.

    The active automata is always in the center (width and height divided by
    two).

    Args:
        width: The width of the seed image in pixels.
        height: The height of the seed image in pixels.
        state_channels: The number of state channels per pixel.

    Returns:
        A new model seed as a three-dimensional tensor.
    """
    if width < 1:
        raise ValueError(f"Width must be positive: {width}.")

    if height < 1:
        raise ValueError(f"Height must be positive: {height}.")

    if state_channels < 1:
        raise ValueError(f"State channels must be positive: {state_channels}.")

    x = zeros((state_channels, height, width), dtype=float32)

    x[3:, height // 2, width // 2] = 1.0

    return from_numpy(x)


def create_random_seed(bytes: int = 8) -> int:
    """
    Creates a single 64-bit random number using a system's `/dev/urandom`.

    Please note that while this function may technically raise an exception,
    this project assumes that it will not.

    Args:
        bytes: The number of random bytes to generate.  This should almost
        always be 64 bits (8 bytes).

    Returns:
        A random integer.
    """
    return int.from_bytes(urandom(bytes), byteorder)


def to_rgb_array(x: Tensor, premultiply: bool = False) -> ndarray[uint8]:
    """
    Converts a model output tensor to an array of RGB values.

    Args:
        x: The model output to convert as a four-dimensional tensor.
        premultiply: Whether to multiply the RGB components by any alpha values.

    Returns:
        An RGB image as an array of unsigned integers.
    """
    rgb = x[:, :3, :, :].clip(min=0.0, max=1.0)
    alpha = x[:, 3:4, :, :].clip(min=0.0, max=1.0)

    if premultiply:
        rgb *= alpha

    return ((1.0 - alpha + rgb).numpy() * 255.0).astype(uint8)


class Animation:
    """
    A context manager that wraps a single :class:`Writer` instance used to
    create video animations from single image frames.
    """

    _video_path: Path
    """The full path to the resulting video file."""

    _format: str
    """The video encoding format."""

    _fps: int
    """The number of frames per second during recording."""

    def __init__(self, base_dir: Path, base_name: str, format: str = None,
                 fps: int = 25):
        """

        Args:
            base_dir: The directory to save the video file.
            base_name: The name of the file to write without an extension.
            format: The video format to use (avi, mp4, etc.).
            fps: The number of frames per second to record.
        """
        if not format:
            format = "mp4"

        self._video_path = (base_path / f"{base_name}.{format}").resolve()
        self._format = format
        self._fps = fps

    @property
    def format(self) -> str:
        """The video encoding format."""
        return self._format

    @property
    def fps(self) -> int:
        """The frames per second during recording."""
        return self._fps

    @property
    def video_path(self) -> Path:
        """The full path to a video file."""
        return self._video_path

    def __enter__(self):
        """
        Opens a new video stream with a specific encoding.

        Raises:
            FileExistsError: If `video_path` already exists.
        """
        if not self._writer:
            if self._video_path.exists():
                raise FileExistsError(
                    f"Cannot create video: {self._video_path} already exists."
                )

            self._writer = get_writer(self._video_path, format=self._format,
                                      mode="I", fps=self._fps)

    def __exit__(self, **kwargs):
        """
        Closes this video stream if active.
        """
        if self._writer:
            self._writer.close()

    def add_frame(self, x: Tensor, premultiply: bool = False):
        """
        Adds a single image frame to this video stream.

        Args:
            x: The model output to encode.
            premultiply: Whether to multiply the RGB components by any alpha
            values.
        """
        frame = to_rgb_array(x, premultiply)

        self._writer.append_data(frame)
