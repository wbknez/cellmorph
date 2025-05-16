"""
Ensures that model execution works as expected.
"""
from numpy import cos, sin
from pytest import mark
from torch import (
    Tensor,
    compile,
    equal,
    logical_and,
    manual_seed,
    no_grad,
    outer,
    rand,
    stack,
    tensor
)
from torch.distributions.binomial import Binomial
from torch.nn import Conv2d
from torch.nn.functional import conv2d, relu

from  cellmorph.model import Model, is_active
from cellmorph.utils import random_bytes


def perceive(x: Tensor, state_channels: int, rotation: float) -> Tensor:
    identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()

    dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
    dy = dx.T.flipud()

    c = cos(rotation)
    s = sin(rotation)

    kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])
    kernel = kernel.repeat(state_channels, 1, 1).unsqueeze(1)
    
    kernel.requires_grad = False

    return conv2d(x, weight=kernel, padding=1, groups=state_channels)


def update(x: Tensor, state_channels: int, hidden_channels: int, bias: bool):
    conv0 = Conv2d(in_channels=state_channels * 3, out_channels=hidden_channels,
                   kernel_size=1, padding=0)
    conv1 = Conv2d(in_channels=hidden_channels, out_channels=state_channels,
                   kernel_size=1, padding=0, bias=bias)

    with no_grad():
        conv1.weight.data.zero_()

    if bias:
        conv1.weight.bias.zero_()

    y = conv0(x)
    y = relu(y)
    return conv1(y)


def manual_stochastic_update(x: Tensor, update_rate: float) -> Tensor:
    dist = Binomial(total_count=1, probs=update_rate)
    output = (x.shape[-2], x.shape[-1])
    y = x.clone().detach()

    for i in range(len(x)):
        y[i, :] *= dist.sample(output)

    return y


def forward(x: Tensor, state_channels: int, hidden_channels: int,
            rotation: float, step_size: float, update_rate: float,
            threshold: float, bias: bool) -> Tensor:
    pa = is_active(x, threshold)

    y = perceive(x, state_channels, rotation)
    dx = update(y, state_channels, hidden_channels, bias) * step_size

    x += manual_stochastic_update(dx, update_rate)

    pl = is_active(x, threshold)
    s = logical_and(pa, pl).float().unsqueeze(1)

    return x * s


class TestModel:
    """
    Test suite for :class:`Model`.
    """

    def test_is_compiled_is_false_for_a_noncompiled_model(self):
        model = Model()

        assert not model.is_compiled
        assert isinstance(model, Model)

    def test_is_compiled_is_true_for_a_compiled_model(self):
        model = Model().compile()

        assert model.is_compiled
        assert not isinstance(model, Model)

    @mark.parametrize('state_channels,hidden_channels,rotation,step_size,update_rate,threshold,bias', [
        (16, 128, 0.0, 1.0, 0.5, 0.1, False)
    ])
    def test_forward_matches_expected_execution(
        self, state_channels: int, hidden_channels: int, rotation: float,
        step_size: float, update_rate: float, threshold: float, bias: bool
    ):
        seed = random_bytes(8)
        model = Model(
            state_channels=state_channels,
            hidden_channels=hidden_channels,
            update_rate=update_rate,
            step_size=step_size,
            rotation=rotation,
            threshold=threshold,
            normalize_kernel=False,
            use_bias=bias
        )
        x = rand((10, state_channels, 72, 72))

        manual_seed(seed)
        expected = forward(x, state_channels, hidden_channels, rotation,
                           step_size, update_rate, threshold, bias)

        manual_seed(seed)
        result = model(x)

        equal(result, expected)
