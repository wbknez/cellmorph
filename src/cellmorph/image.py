"""
Contains utility functions for working with images.
"""
from __future__ import annotations
from dataclasses import astuple, dataclass

from numpy import array, float32, ndarray, uint8
from PIL import Image as ImageFactory
from PIL.Image import Image
from torch import Tensor, from_numpy


ImageLike = Image | ndarray[float32] | ndarray[uint8]
"""Allows multiple types of image representations."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """
    The size of a two dimensional image in pixels.
    """

    width: int
    """The width, or length along the x-axis, of an image in pixels."""

    height: int
    """The height, or length along the y-axis, of an image in pixels."""

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"Width must be positive: {self.width}.")

        if self.height <= 0:
            raise ValueError(f"Height must be positive: {self.height}.")

    @classmethod
    def from_image(cls, img: Image) -> Dimension:
        """
        Determines the dimensions of an image size.

        Args:
            img: The image to use.

        Returns:
            The size of an image as a dimension.
        """
        return Dimension(*img.size)

    @classmethod
    def from_tensor(cls, x: Tensor) -> Dimension:
        """
        Determines the dimensions of a model tensor.

        Args:
            x: The tensor to use.

        Returns:
            The size of a tensor as a dimension.
        """
        return Dimension(x.shape[-1], x.shape[-2])


def can_thumbnail(size: Dimension, max_size: Dimension) -> bool:
    """
    Determines whether an image may be shrunk to a thumbnail.

    Images can only be shrunk; they cannot be enlarged.  This operation
    preserves the aspect ratio.

    Args:
        size: The current size of the image.
        max_size: The new desired size.

    Returns:
        `True` if the image can be resized properly, otherwise `False`.
    """
    is_equal = size == max_size
    is_w_greater = size.width < max_size.width
    is_h_greater = size.height < max_size.height

    return not (is_equal or is_w_greater or is_h_greater)


def choose_mode(color_channels: int) -> str:
    """
    Determines the image mode that corresponds to the number of color
    channels in a Pillow image of unsigned integers.

    Args:
        color_channels: The number of color channels.

    Returns:
        A Pillow image mode as a string representation.
    """
    match color_channels:
        case 1:
            return "L"
        case 2:
            return "LA"
        case 3:
            return "RGB"
        case 4:
            return "RGBA"
        case _:
            raise ValueError(
                f"Unknown number of color channels: {color_channels}."
            )


def pad(img: Image, padding: int) -> Image:
    """
    Pads an image with a specific amount of empty (black with 100% alpha) pixels
    around the border.

    Args:
        img: The image to apply padding to.
        padding: The amount of space to add in all directions (top, left, right,
        and bottom).  Every direction is constant to keep the image square.

    Returns:
        A new image whose border has been padded.
    """
    if not padding:
        return img

    width, height = img.size
    new_width, new_height = width + padding * 2, height + padding * 2

    new_img = ImageFactory.new(img.mode, (new_width, new_height), (0, 0, 0, 0))
    new_img.paste(img, (padding, padding))

    return new_img


def to_floats(img: ImageLike, premultiply: bool = False) -> ndarray[float32]:
    """
    Converts an image to a floating-point array.

    Args:
        img: The image to convert.
        premultiply: Whether to premultiply the RGB components by the alpha
        channel.

    Returns:
        An array of pixels as an array of floating-point.
    """
    is_image = isinstance(img, Image)
    is_uint_array = isinstance(img, ndarray) and (img.dtype == uint8)

    if is_image or is_uint_array:
        arr = array(img, dtype=float32) / 255.0
    else:
        arr = img

    if arr.shape[2] == 4 and premultiply:
        arr[..., :3] *= arr[..., 3:4]

    return arr


def from_tensor(x: Tensor, mode: str | None = None) -> Image:
    """
    Converts a tensor (typically output from a model) to a Pillow image.

    Args:
        x: The tensor to convert.
        mode: The image mode to use; optional.

    Returns:
        An image as an array of unsigned 8-bit integer values.
    """
    if len(x.shape) < 3:
        raise ValueError(f"Image tensor has too few dimensions: {x.shape}.")

    if len(x.shape) > 4:
        raise ValueError(f"Image tensor has too many dimensions: {x.shape}.")

    # Model output tensors always have four dimensions.
    if len(x.shape) == 4:
        x = x.squeeze(0)

    # Remove excess state channels if necessary.
    # All RGBA channels are always sequential from 0-4.
    if x.shape[0] > 4:
        x = x[:4, ...]

    arr = x.clip(min=0.0, max=1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(uint8)
    mode = choose_mode(arr.shape[-1])

    return ImageFactory.fromarray(arr, mode)


def to_tensor(img: ImageLike, premultiply: bool = False) -> Tensor:
    """
    Converts an image to a tensor suitable for input to a CNA model.

    This function essentially takes the output of `to_floats` and converts it to
    a :class:`Tensor`.

    Args:
        img: The image to convert.
        premultiply: Whether to premultiply the RGB components by the alpha
        channel.

    Returns:
        A Pillow image as a PyTorch tensor.
    """
    arr = to_floats(img, premultiply=premultiply)

    return from_numpy(arr.transpose(2, 0, 1))
