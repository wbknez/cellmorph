"""
Implements the neural cellular automata model for this project.

This model is based on the work of Mordvintsev et al. (2020) and is a direct
implementation in PyTorch.  In addition, it contains implementation details from
Gleb Sterkin's repository on Github.

## References
  1. Mordvintsev, A., Randazzo, E., Niklasson, E. and Levin, M. (2020). Growing
     neural cellular automata: Differentiable models of morphogenesis. Distill.
     Retrieved from http://distill.pub/2020/growing-ca/
  2. Sterkin, Gleb. (2020). Cellular automata pytorch. Retrieved from
     https://github.com/belkakari/cellular-automata-pytorch
"""
from typing import override

from numpy import cos as npCos, pi, sin as npSin
from torch import Tensor, float32, from_numpy, logical_and, outer, stack, tensor
from torch.distributions import Bernoulli
from torch.nn import Conv2d, Module, Parameter, ReLU, Sequential
from torch.nn.functional import conv2d, max_pool2d
from torch.nn.init import constant_


def is_active(x: Tensor, threshold: float = 0.1) -> Tensor:
    """
    Determines which cellular automata are considered "active" based on the
    value of their alpha channels.

    For this project, an automata is considered active if its alpha channel has
    a value greater than `0.1`.  This allows it to be considered for updating
    during the current time step.

    Args:
        x: The collection of automata states.
        threshold: The cutoff for being deemed active.

    Returns:
        A collection of which automata are active as a tensor boolean mask.
    """
    alpha = x[:, 3, :, :].clip(min=0.0, max=1.0)
    objs = max_pool2d(input=alpha, kernel_size=3, stride=1, padding=1)

    return objs > threshold


class PerceptionRule(Module):
    """
    A :class:`Module` that implements the perception step of this model by
    applying a depthwise convolution to an automata's neighborhood using a Sobel
    filter as weights.
    """

    _weights: Parameter
    """A computed Sobel filter."""

    _state_channels: int
    """The size of the state space per cellular automata."""

    def __init__(self, state_channels: int, rotation: float,
                 normalize: bool = False):
        """
        Initializes the Sobel filter for this model.

        Args:
            state_channels: The number of state channels per automata.
            rotation: An angle of rotation in radians.
            normalize: Whether to divide the resulting Sobel filter by the
            number of state channels.
        """
        super().__init__()

        identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()

        dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
        dy = dx.T.flipud()

        c = npCos(rotation)
        s = npSin(rotation)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])
        kernel = kernel.repeat(state_channels, 1, 1).unsqueeze(1)

        if normalize:
            kernel /= state_channels

        self._weights = Parameter(kernel, requires_grad=False)
        self._state_channels = state_channels

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply perception rule to all automata.

        Evaluates each automata's neighborhood in a Moore neighborhood of radius
        one (three by three).

        Args:
            x: The current model state.

        Returns:
            A perception vector as a tensor.
        """
        return conv2d(x, weight=self._weights, padding=1,
                      groups=self._state_channels)


class UpdateRule(Module):
    """
    A :class:`Module` that implements a differentiable update step as a series
    of convolutions.
    """

    def __init__(self, state_channels: int, intermediate_channels: int = 128,
                 kernel_size: int = 1, padding: int | str = 0,
                 use_bias: bool = False):
        """
        Initializes both convolution layers.

        Args:
            state_channels: The number of state channels per automata.
            intermediate_channels: The number of intermediate state channels.
            kernel_size: The size of the convolution kernel.
            padding: The amount of padding to use.
            use_bias: Whether to compute and learn bias.
        """
        super().__init__()

        l1_in_channels = state_channels * 3

        self._layers = Sequential(
            Conv2d(in_channels=l1_in_channels,
                   out_channels=intermediate_channels, kernel_size=kernel_size,
                   padding=padding),
            ReLU(),
            Conv2d(in_channels=intermediate_channels, out_channels=state_channels,
                   kernel_size=kernel_size, padding=padding, bias=use_bias)
        )

        # Force initialization of second convolution layer weights.
        constant_(self._layers[2].weight, 0.0)

        if use_bias:
            constant_(self._layers[2].bias, 0.0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the update "rule" to all automata as a pair of convolutions.

        Args:
            x: The current model state.
        """
        return self._layers(x)


class CellMorphModel(Module):
    """
    A :class:`Module` that models a morphogenetic process by applying
    convolution-based update rules to a collection of cellular automata.
    """

    _perceiver: PerceptionRule
    """The perception rule as a Sobel filter convolution."""

    _updater: UpdateRule
    """The update rule whose current weights are learned."""

    _state_channels: int
    """The size of the state space per cellular automata."""

    _step_size: float
    """The relative amount of time that passes per step."""

    _threshold: float
    """The alpha channel value that denotes an active automata."""

    _update_rate: float
    """The probability of a single cellular automata updating per step."""

    def __init__(self, state_channels: int = 16, padding: int | str = 0,
                 update_rate: float = 0.5, step_size: int = 1.0,
                 rotation: float = 0.0, threshold: float = 0.1,
                 normalize_kernel: bool = False, use_bias: bool = False)
        """
        Initializes all relevant class fields and ensures that all inputs are
        within expected limits.

        Raises:
            ValueError: If `state_channels` is not positive, `update_rate` is
            not in (0, 1], `rotation_angle` is not in [0, 2pi], or `threshold`
            is not in (0, 1].
        """
        super().__init__()

        if state_channels <= 0:
            raise ValueError(
                f"State channels must be positive: {state_channels}."
            )

        if isinstance(padding, str):
            if not padding.lower() in ["same", "valid"]:
                raise ValueError(f"Unknown padding mode: {padding}.")

        if not 0.0 < update_rate <= 1.0:
            raise ValueError(
                f"Update rate must be between 0 and 1: {update_rate}."
            )

        if not 0.0 < step_size:
            raise ValueError(
                f"Step size must be positive: {step_size}."
            )

        if not 0.0 <= rotation_angle <= 2 * pi:
            raise ValueError(
                f"Rotation angle must be between 0 and 2pi: {rotation_angle}."
            )

        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0 and 1: {threshold}."
            )

        self._perceiver = PerceptionRule(state_channels, rotation,
                                         normalize_kernel)
        self._updater = UpdateRule(state_channels, padding, bias)

        self._state_channels = state_channels
        self._step_size = step_size
        self._threshold = threshold
        self._update_rate = update_rate

    @property
    def perception_rule(self) -> PerceptionRule:
        """The current perception rule."""
        return self._perceiver

    @property
    def update_rule(self) -> UpdateRule:
        """The current update rule."""
        return self._updater

    @property
    def state_channels(self) -> int:
        """The size of the state space per cellular automata."""
        return self._state_channels

    @property
    def step_size(self) -> float:
        """The current time modifier per step."""
        return self._step_size

    @property
    def threshold(self) -> float:
        """The cutoff for an automata being considered active."""
        return self._threshold

    @property
    def update_rate(self) -> float:
        """The probability of a single cellular automata updating per step."""
        return self._update_rate

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a single forward pass of this NCA model.

        Args:
            x: An input tensor.

        Returns:
            Computed model output as a tensor.
        """
        pre_active = is_active(x, self._threshold).to(self.device)

        y = self._perceiver(x)
        dx = self._updater(y) * self._step_size

        to_update = (uniform(x[:, :1, :, :].shape) <= self._update_rate).float()
        x += dx * to_update.to(self.device)

        post_active = is_active(x1, self._threshold).to(self.device)
        survivors = logical_and(pre_active_mask, post_active_mask).float()

        return x * survivors
