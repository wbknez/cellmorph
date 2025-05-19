"""
A package that contains multiple useful transformations for use with target
images and output tensors.
"""
from typing import Any

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.v2 import Compose, Transform

from cellmorph.transforms.image import Pad, Resize, ToTensor
from cellmorph.transforms.tensor import (
    Clip,
    Premultiply,
    Squeeze,
    ToRgb,
    ToRgba,
    ToImage
)


class Pass(Transform):
    """
    A :class:`Transform` that does nothing but return its input.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Image | Tensor,
                  _: dict[str, Any] | None = None) -> Image | Tensor:
        """
        Returns any input without modification.

        Args:
            inpt: The image or tensor to transform.

        Returns:
            An image or tensor.
        """
        if not isinstance(inpt, Image | Tensor):
            raise TypeError("Pass input must be either an image or tensor.")

        return inpt
