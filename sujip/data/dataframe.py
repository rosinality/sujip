from collections import abc, OrderedDict

import torch
import numpy as np


class Field:
    def __init__(self, pad=False, listify=False, dtype=None, infer=True):
        self.pad = pad
        self.listify = listify
        self.dtype = dtype
        self.infer = infer


class Derived(Field):
    def __init__(self, fn, need_batch=False, *args, **kwargs):
        self.fn = fn
        self.need_batch = need_batch

        super().__init__(*args, **kwargs)


class Batch(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return super().__getitem__(name)

    def __setattr__(self, name, value):
        super().__setitem__(name, value)

    def __iter__(self):
        return iter(super().values())

    def to(self, *args, **kwargs):
        moved = OrderedDict()

        for key, value in super().items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(*args, **kwargs)

        for key, value in moved.items():
            super().__setitem__(key, value)

        return self


def numpy2torch_dtype(dtype):
    if dtype.name.startswith('int'):
        return torch.int64

    elif dtype.name.startswith('float32') or dtype.name.startswith('float64'):
        return torch.float32


def convert_numpy_dtype(array):
    if array.dtype == np.float64:
        return array.astype(np.float32)

    return array


def convert2numpy(seq, dtype=None):
    if isinstance(seq, np.ndarray):
        return seq

    elif isinstance(seq, torch.Tensor):
        return seq.numpy()

    elif isinstance(seq, abc.Iterable):
        if dtype is not None:
            return np.array(seq, dtype=dtype)

        else:
            return np.array(seq)

    else:
        return np.array([seq])


class Row(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(list(super().keys())[0])

        else:
            return super().__getitem__(key)

    def __getattr__(self, name):
        return super().__getitem__(name)

    def __setattr__(self, name, value):
        super().__setitem__(name, value)


class DataFrame:
    def __init__(self, **kwargs):
        self.fields = OrderedDict(kwargs.items())

    def collate_fn(self, infer=False):
        return lambda batch: self._collate_fn(batch, infer)

    def _collate_fn(self, batch, infer=False):
        results = Batch()

        batch_size = len(batch)

        for i, (name, field) in enumerate(self.fields.items()):
            if not field.infer and infer:
                continue

            if isinstance(batch[0], Row):
                ind = name

            else:
                ind = i

            # print(name)

            if isinstance(field, Derived):
                for b in batch:
                    if field.need_batch:
                        b[ind] = field.fn(b, results)

                    else:
                        b[ind] = field.fn(b)

            if field.listify:
                output = []

                for b in batch:
                    output.append(b[ind])

            elif field.pad:
                batch_array = [convert2numpy(b[ind], field.dtype) for b in batch]

                max_sizes = []
                for j in range(batch_array[0].ndim):
                    max_size = max(b.shape[j] for b in batch_array)
                    max_sizes.append(max_size)

                output = np.zeros((batch_size, *max_sizes), dtype=batch_array[0].dtype)

                for batch_ind, b in enumerate(batch_array):
                    slices = [slice(s) for s in b.shape]
                    # output.__setitem__((batch_ind, *slices), b)
                    output[(batch_ind, *slices)] = b

                output = torch.from_numpy(convert_numpy_dtype(output))

            else:
                output = [convert2numpy(b[ind]) for b in batch]
                output = np.stack(output, axis=0)
                output = torch.from_numpy(convert_numpy_dtype(output))

            results[name] = output

        return Batch(**results)
