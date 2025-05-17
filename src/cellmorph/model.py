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
  3. chenmingxiang110. (2023). Growing neural cellular automata. Retrieved from 
     https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata
"""
from __future__ import annotations
from typing import override

from numpy import cos, sin
from torch import (
    Tensor,
    compile as torch_compile,
    float32,
    from_numpy,
    load,
    logical_and,
    outer,
    rand as uniform,
    save,
    stack,
    tensor
)
from torch.distributions import Bernoulli
from torch.nn import Conv2d, Module, Parameter, ReLU, Sequential
from torch.nn.functional import conv2d, max_pool2d
from torch.nn.init import constant_
from torch.nn.utils import clip_grad_norm_

from cellmorph.utils import strip_key_prefix


COMPILED_PREFIX = "_orig_mod."
"""Key prefix used by PyTorch in :meth:`Model.state_dict()` to denote weights
inherited from a non-compiled model."""


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

    def __init__(self, state_channels: int, rotation: float = 0.0,
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

        c = cos(rotation)
        s = sin(rotation)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])
        kernel = kernel.repeat(state_channels, 1, 1).unsqueeze(1)

        if normalize:
            kernel /= state_channels

        self._weights = Parameter(kernel, requires_grad=False)
        self._state_channels = state_channels

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a perception "rule" to all automata.

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

    def __init__(self, state_channels: int, hidden_channels: int = 128,
                 use_bias: bool = False):
        """
        Initializes both convolution layers.

        Args:
            state_channels: The number of state channels per automata.
            hidden_channels: The number of hidden state channels.
            use_bias: Whether to compute and learn bias.
        """
        super().__init__()

        l1_in_channels = state_channels * 3

        self._layers = Sequential(
            Conv2d(in_channels=l1_in_channels, out_channels=hidden_channels,
                   kernel_size=1, padding=0),
            ReLU(),
            Conv2d(in_channels=hidden_channels, out_channels=state_channels,
                   kernel_size=1, padding=0, bias=use_bias)
        )

        # Force initialization of second convolution layer weights.
        constant_(self._layers[2].weight, 0.0)

        if use_bias:
            constant_(self._layers[2].bias, 0.0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies an update "rule" to all automata as a pair of convolutions.

        Args:
            x: The current model state.
        """
        return self._layers(x)


class Model(Module):
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

    _is_compiled: bool
    """Whether this model has been compiled using :meth:`torch.compile`."""

    def __init__(self, state_channels: int = 16, hidden_channels: int = 128,
                 update_rate: float = 0.5, step_size: int = 1.0,
                 rotation: float = 0.0, threshold: float = 0.1,
                 normalize_kernel: bool = False, use_bias: bool = False):
        """
        Initializes all relevant class fields and ensures that all inputs are
        within expected limits.

        Raises:
            ValueError: If `state_channels` is not positive, `update_rate` is
            not in (0, 1], `rotation_angle` is not in [0, 2pi], or `threshold`
            is not in (0, 1].
        """
        super().__init__()

        self._perceiver = PerceptionRule(state_channels, rotation,
                                         normalize_kernel)
        self._updater = UpdateRule(state_channels, hidden_channels, use_bias)

        self._state_channels = state_channels
        self._step_size = step_size
        self._threshold = threshold
        self._update_rate = update_rate
        self._is_compiled = False

    @property
    def is_compiled(self) -> bool:
        """
        Whether this model has been compiled with :meth:`torch.compile`.

        Please note that this property is only reliable if the accompanying
        function :meth:`Model.compile` is used to compile this model.
        Otherwise, the result may be inaccurate because :meth:`torch.compile`
        produces a wrapper module that forwards calls to the original model.
        """
        return self._is_compiled

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

    def clip_gradient_norms(self, gradient_cutoff: float):
        """
        Restricts the value of all gradient norms in the update rule to be less
        than a specific value.

        Args:
        """
        clip_grad_norm_(self._updater.parameters(), gradient_cutoff)

    def compile(self) -> Module:
        """
        Compiles this model using :meth:`torch.compile`.

        Please note that this function does not compile in-place; similar to
        :meth:`Module.to`, it returns a reference to a new instance (copy) of
        the original model in compiled form.

        Returns:
            A compiled model.
        """
        self._is_compiled = True

        return torch_compile(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a single forward pass of this NCA model.

        Args:
            x: An input tensor.

        Returns:
            Computed model output as a tensor.
        """
        pre_active = is_active(x, self._threshold)

        y = self._perceiver(x)
        dx = self._updater(y) * self._step_size

        to_update = (uniform(x[:, :1, :, :].shape) <= self._update_rate).float()
        x1 = x + dx * to_update.to(x.device)

        post_active = is_active(x1, self._threshold)
        survivors = logical_and(pre_active, post_active).float().unsqueeze(1)

        return x1 * survivors.to(x.device)

    def load(self, weights_path: Path) -> Model:
        """
        Loads all layer weights from a specific file path.

        Please note that this should be done **before** compilation.  The weight
        files for this project assume that uncompiled models will load their
        weights first before compilation.

        Args:
            weights_path: The file location to load weights from.

        Returns:
            A reference to this model for easy chaining.
        """
        self.load_state_dict(load(weights_path, weights_only=True))

        return self

    def save(self, weights_path: Path):
        """
        Saves the weights of this model to a specific file path.

        In addition, this function removes the compilation prefix (typically
        "_orig_mod.") from any weight names that contain it.  This prefix is
        automatically appended to all weights after the model is compiled using
        `torch.compile`.  This removal is necessary because the `OptimizedModel`
        returned from PyTorch's compilation process overrides `state_dict`,
        providing a copy of all weights prefixed with an identifier to separate
        them from the original model.  However, these values are identical but
        this process makes it impossible for uncompiled models to load these
        weights without modification.

        Args:
            weights_path: The file location to save weights to.
        """
        weights = self.state_dict()

        if self.is_compiled:
            weights = strip_key_prefix(weights, COMPILED_PREFIX)

        save(weights, weights_path)
