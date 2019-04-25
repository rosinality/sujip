import numpy as np
import torch

from sujip.data import DataFrame, Field, Row, Derived


def test_collate_pad():
    dset = DataFrame(image=Field(), text=Field(pad=True))

    batch = [
        [np.ones((3, 3)), [1, 2]],
        [np.ones((3, 3)), [1, 2, 3, 4, 5]],
        [np.ones((3, 3)), [1, 2, 3]],
    ]

    collate = dset.collate_fn()(batch)

    image = torch.ones(3, 3, 3)
    text = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])

    assert torch.allclose(collate.image, image)
    assert torch.all(collate.text == text).item() == 1


def test_collate_pad2d():
    dset = DataFrame(image=Field(pad=True))

    batch = [
        Row(image=np.arange(3 * 3, dtype=np.float32).reshape((3, 3))),
        Row(image=np.arange(5 * 3).reshape((5, 3))),
        Row(image=np.arange(5 * 5).reshape((5, 5))),
    ]

    collate = dset.collate_fn()(batch)

    image = torch.tensor(
        [
            [
                [0.0, 1, 2, 0, 0],
                [3, 4, 5, 0, 0],
                [6, 7, 8, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 2, 0, 0],
                [3, 4, 5, 0, 0],
                [6, 7, 8, 0, 0],
                [9, 10, 11, 0, 0],
                [12, 13, 14, 0, 0],
            ],
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
            ],
        ]
    )

    assert torch.allclose(collate.image, image)


def test_collate_derived():
    dset = DataFrame(
        image=Field(pad=True),
        image_perturb=Derived(fn=lambda b: b.image * -1, pad=True),
    )

    batch = [
        Row(image=np.arange(3 * 3, dtype=np.float32).reshape((3, 3))),
        Row(image=np.arange(5 * 3).reshape((5, 3))),
        Row(image=np.arange(5 * 5).reshape((5, 5))),
    ]

    collate = dset.collate_fn()(batch)

    image = torch.tensor(
        [
            [
                [0.0, 1, 2, 0, 0],
                [3, 4, 5, 0, 0],
                [6, 7, 8, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 2, 0, 0],
                [3, 4, 5, 0, 0],
                [6, 7, 8, 0, 0],
                [9, 10, 11, 0, 0],
                [12, 13, 14, 0, 0],
            ],
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
            ],
        ]
    )

    assert torch.allclose(collate.image, image)
    assert torch.allclose(collate.image_perturb, image * -1)


def test_collate_listify():
    dset = DataFrame(text=Field(listify=True))

    batch = [Row(text=[1, 2, 3]), Row(text=[1, 2]), Row(text=[1, 2, 3, 4, 5])]

    collate = dset.collate_fn()(batch)

    text = [[1, 2, 3], [1, 2], [1, 2, 3, 4, 5]]

    assert collate.text == text


def test_collate_derived_need_batch():
    dset = DataFrame(
        image=Field(pad=True),
        text=Field(listify=True),
        length=Derived(fn=lambda b: len(b.text), listify=True),
        length_mult=Derived(
            fn=lambda b, batch: len(b.text) * batch.image.shape[1],
            need_batch=True,
            listify=True,
        ),
    )

    batch = [
        Row(image=np.ones((3, 3)), text=[1, 2]),
        Row(image=np.ones((4, 3)) * 2, text=[1, 2, 3, 4, 5]),
        Row(image=np.ones((5, 3)) * 3, text=[1, 2, 3]),
    ]

    collate = dset.collate_fn()(batch)

    length = [2, 5, 3]
    length_mult = [10, 25, 15]

    assert collate.image.shape[1] == 5
    assert collate.length == length
    assert collate.length_mult == length_mult


def test_collate_derived_dtype():
    dset = DataFrame(target=Field(pad=True, dtype='float32'))

    batch = [Row(target=[1, 2]), Row(target=[1, 2, 3, 4, 5]), Row(target=[1, 2, 3])]

    collate = dset.collate_fn()(batch)

    assert collate.target.dtype == torch.float32

    dset = DataFrame(target=Field(pad=True, dtype='int64'))

    batch = [Row(target=[1.0, 2]), Row(target=[1, 2, 3, 4, 5]), Row(target=[1, 2, 3])]

    collate = dset.collate_fn()(batch)

    assert collate.target.dtype == torch.int64


def test_collate_derived_infer():
    dset = DataFrame(target=Field(pad=True, infer=False))

    batch = [Row(target=[1, 2]), Row(target=[1, 2, 3, 4, 5]), Row(target=[1, 2, 3])]

    collate = dset.collate_fn()(batch)

    text = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])

    assert torch.all(collate.target == text).item() == 1

    collate = dset.collate_fn(infer=True)(batch)

    try:
        collate.target

    except KeyError:
        assert True

    else:
        assert False
