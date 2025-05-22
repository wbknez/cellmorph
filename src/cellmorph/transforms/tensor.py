"""
Contains a collection of project-specific tensor operations written as PyTorch
transformations to allow arbitrary composition.
"""
from typing import Any

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.v2 import Transform
from torchvision.transforms.functional import to_pil_image


class Clip(Transform):
    """
    Constrains the values of a :class:`Tensor` between a minimum and maximum.
    """

    _min: float
    """The minimum value to allow."""

    _max: float
    """The maximum value to allow."""

    def __init__(self, min: float, max: float):
        if not 0.0 <= min <= 1.0:
            raise ValueError(f"Clip minimum must be in [0, 1]: {min}.")

        if min > max:
            raise ValueError(f"Clip maximum must be greater than or equal to "
                             f"the minimum: {max}.")

        self._min = min
        self._max = max

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Tensor:
        """
        Transforms an input tensor into a tensor whose values are constrained by
        a minimum and maximum.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("Clip input must be a tensor.")

        return inpt.clip(min=self._min, max=self._max)


class Premultiply(Transform):
    """
    Multiplies the three-channel RGB values of a :class:`Tensor` by its single
    channel alpha values.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Tensor:
        """
        Transforms an input tensor into a tensor whose values are defined by the
        multiplication of the RGB channels by the alpha channel.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("Premultiply input must be a tensor.")

        if not 3 <= len(inpt.shape) <= 4:
            raise ValueError("Premultiply input must be a 3D or 4D tensor.")

        if inpt.shape[1] < 4:
            raise ValueError("Premultiply requires at least 4 channels.")

        is_4d = len(inpt.shape) == 4
        premult = inpt.clone().detach()

        if is_4d:
            premult[:, :3, ...] *= premult[:, 3:4, ...]
        else:
            premult[:3, ...] *= premult[3:4, ...]

        return premult


class Squeeze(Transform):
    """
    Reduces a :class:`Tensor` by a single dimension.
    """

    _axis: int
    """The axis to remove."""

    def __init__(self, axis: int):
        if axis < 0:
            raise ValueError(f"Squeeze axis must be positive: {axis}.")

        self._axis = axis

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Tensor:
        """
        Removes a single axis from a :class:`Tensor`.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("Squeeze input must be a tensor.")

        if self._axis >= len(inpt.shape):
            raise ValueError("Squeeze axis is out of bounds.")

        return inpt.squeeze(self._axis)


class ToImage(Transform):
    """
    Converts a PyTorch :class:`Tensor` into a Pillow :class:`Image`.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Image:
        """
        Transforms an input tensor into an image.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new image.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("ToImage input must be a tensor.")

        return to_pil_image(inpt)


class ToRgb(Transform):
    """
    Reduces a :class:`Tensor` with at least four color channels into a tensor
    with only three.

    This transformation is accomplished using the following formula:
        `1.0 - alpha + rgb`
    where `rgb` is the first three components and `alpha` is the fourth.  This
    operation inverts the transparency and incorporates it into the resulting
    RGB image.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Tensor:
        """
        Transforms an input tensor into a three-channel RGB tensor.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("ToRgb input must be a tensor.")

        if len(inpt.shape) != 4:
            raise ValueError("ToRgb input must be a 4D tensor.")

        if inpt.shape[1] < 4:
            raise ValueError("ToRgb requires at least 4 channels.")

        rgb = inpt[:, :3, ...]
        alpha = inpt[:, 3:4, ...]

        return 1.0 - alpha + rgb


class ToRgba(Transform):
    """
    Extracts the RGBA channels from a :class:`Tensor` of arbitrary shape with at
    least four color channels.
    """

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Tensor,
                  _: dict[str, Any] | None = None) -> Tensor:
        """
        Transforms an input tensor into a four-channel RGBA tensor.

        Args:
            inpt: The tensor to transform.

        Returns:
            A new tensor.
        """
        if not isinstance(inpt, Tensor):
            raise TypeError("ToRgba input must be a tensor.")

        if len(inpt.shape) != 4:
            raise ValueError("ToRgba input must be a 4D tensor.")

        if inpt.shape[1] < 4:
            raise ValueError("ToRgba requires at least 4 channels.")

        return inpt[:, :4]
