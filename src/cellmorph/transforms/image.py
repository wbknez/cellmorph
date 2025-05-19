"""
Contains a collection of project-specific image operations written as PyTorch
transformations to allow arbitrary composition.
"""
from dataclasses import astuple
from typing import Any

from PIL import Image as ImageFactory
from PIL.Image import Image
from torch import Tensor, float32
from torchvision.transforms.v2 import Transform
from torchvision.transforms.functional import to_tensor

from cellmorph.data import Dimension


class Pad(Transform):
    """
    Adds a constant number of pixels to an :class:`Image` around each side.
    """

    _amount: int
    """
    The amount of space to add in all directions (top, left, right, and bottom).
    Every direction is constant to keep the image square.
    """

    _fill_color: tuple[int, int, int, int]
    """The RGBA color of any padded pixels."""

    def __init__(self, amount: int,
                 fill_color: tuple[int, int, int, int] | None = None):
        super().__init__()

        if not fill_color:
            fill_color = (0, 0, 0, 0)

        if amount < 0:
            raise ValueError(f"Padding amount must be positive: {amount}.")

        self._amount = amount
        self._fill_color = fill_color

    def transform(self, inpt: Image, _: dict[str, Any] | None = None) -> Image:
        """
        Adds a layer of pixel padding around an image.

        Args:
            inpt: The image to transform.

        Returns:
            A new image.
        """
        if not isinstance(inpt, Image):
            raise TypeError("Pad input must be an image.")

        width, height = inpt.size

        new_width = width + self._amount * 2
        new_height = height + self._amount * 2

        new_img = ImageFactory.new(
            inpt.mode, (new_width, new_height), self._fill_color
        )
        new_img.paste(inpt, (self._amount, self._amount))

        return new_img


class Resize(Transform):
    """
    Resizes an :class:`Image` without preserving its aspect ratio.
    """

    _size: Dimension
    """A new size."""

    def __init__(self, size: Dimension):
        super().__init__()

        if not size:
            raise ValueError("Resize dimension is not valid.")

        self._size = size

    def transform(self, inpt: Image, _: dict[str, Any] | None = None) -> Image:
        """
        Transforms an input image into a differently sized version of itself.

        Args:
            inpt: The image to transform.

        Returns:
            A new image.
        """
        if not isinstance(inpt, Image):
            raise TypeError("Resize input must be an image.")

        return inpt.resize(astuple(self._size), ImageFactory.LANCZOS)


class ToTensor(Transform):
    """
    Converts a Pillow :class:`Image` into a PyTorch :class:`Tensor`.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Image, _: dict[str, Any] | None = None) -> Tensor:
        """
        Transforms an input image into a tensor.

        Args:
            inpt: The image to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Image):
            raise TypeError("ToTensor input must be an image.")

        return to_tensor(inpt).float()
