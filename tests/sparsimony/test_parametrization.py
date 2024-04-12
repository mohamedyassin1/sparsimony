import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from sparsimony.parametrization.fake_sparsity import (
    FakeSparsity,
    FakeSparsityDenseGradBuffer,
)


class TestFakeSparsity:
    @pytest.fixture(
        params=[
            torch.ones((3, 3)),
            torch.tensor(
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.float32
            ),
        ]
    )
    def fake_sparsity(self, request):
        mask = request.param
        return FakeSparsity(mask)

    def test_fake_sparsity_forward(self, fake_sparsity):
        x = torch.ones((3, 3))
        output = fake_sparsity(x)
        assert torch.allclose(output, x * fake_sparsity.mask)

    def test_fake_sparsity_state_dict(self, fake_sparsity):
        state_dict = fake_sparsity.state_dict()
        assert "mask" in state_dict


class TestFakeSparsityDenseGradBuffer:
    @pytest.fixture(
        params=[
            torch.ones((3, 3)),
            torch.tensor(
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.float32
            ),
        ]
    )
    def fake_sparsity_dense_grad_buffer(self, request):
        mask = request.param
        return FakeSparsityDenseGradBuffer(mask)

    @pytest.fixture
    def linear(self, request, fake_sparsity_dense_grad_buffer):
        fake_sparsity_dense_grad_buffer.accumulate = True
        shape = fake_sparsity_dense_grad_buffer.mask.shape
        _linear = nn.Linear(
            in_features=shape[0], out_features=shape[1], bias=False
        )
        parametrize.register_parametrization(
            _linear, "weight", fake_sparsity_dense_grad_buffer
        )
        return _linear

    def test_fake_sparsity_dense_grad_buffer_forward(
        self, fake_sparsity_dense_grad_buffer, linear
    ):
        target = torch.ones((3, 3))
        x = torch.ones((3, 3))
        out = linear(x)
        loss = torch.abs(target - out).sum()
        loss.backward()
        grad_target = linear.parametrizations.weight.original.grad
        assert (grad_target == fake_sparsity_dense_grad_buffer.dense_grad).all()

    def test_fake_sparsity_dense_grad_buffer_state_dict(
        self, fake_sparsity_dense_grad_buffer
    ):
        state_dict = fake_sparsity_dense_grad_buffer.state_dict()
        assert "mask" in state_dict
        assert "dense_grad" in state_dict
