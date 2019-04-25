import torch

from sujip.data import Batch


def test_batch_access():
    x = torch.ones(3, 3)
    y = torch.zeros(5)
    z = torch.randn(2, 2)

    batch = Batch(x=x, y=y, z=z)

    assert torch.allclose(batch.x, x)
    assert torch.allclose(batch.y, y)
    assert torch.allclose(batch.z, z)
