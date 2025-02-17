from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, overload, runtime_checkable

import numpy.typing as npt


@runtime_checkable
class DatasetSplit(Protocol):
    """An interface for a dataset split containing input and target pairs."""

    @property
    def inputs(self) -> npt.NDArray[Any]:
        """The inputs of the dataset split."""

        ...

    @property
    def raw_inputs(self) -> npt.NDArray[Any]:
        """
        The untransformed input data.

        ### Raises
        ----------
        `AttributeError`: if the transformed data has been cached (i.e., the raw data is not available).
        """

        ...

    @property
    def targets(self) -> npt.NDArray[Any]:
        """The transformed target data."""
        ...

    @property
    def raw_targets(self) -> npt.NDArray[Any]:
        """
        The untransformed target data.

        ### Raises
        ----------
        `AttributeError`: if the transformed data has been cached (i.e., the raw data is not available).
        """
        ...

    @property
    def primitives(self) -> list[tuple[Any, Any]]:
        """The (transformed input, transformed target) data pairs."""
        ...

    @property
    def raw_primitives(self) -> list[tuple[Any, Any]]:
        """The (input, target) data pairs."""
        ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, idx: int) -> tuple[Any, Any]: ...

    @overload
    def __getitem__(self, idx: Sequence[int]) -> DatasetSplit: ...

    def __getitems__(self, idxs: Sequence[int]) -> DatasetSplit: ...

    def __iter__(self) -> Iterator[tuple[Any, Any]]: ...

    def sample(self, n: int = 1, replace: bool = False) -> DatasetSplit:
        """
        Get a random sample of the split.

        ### Parameters
        ----------
        `n`: the number of samples to get.
        `replace`: whether to sample with replacement.

        ### Returns
        ----------
        A `Split` containing a random sample.
        """

        ...

    def to_pytorch(self) -> DatasetPyTorchSplit:
        """
        Convert the split to a PyTorch-compatible split.
        This conversion should be memory-efficient as inputs and targets are copied by reference.
        """

        ...


@runtime_checkable
class DatasetPyTorchSplit(DatasetSplit, Protocol):
    """
    A PyTorch-compatible split of the dataset containing input and target data.
    The key difference w.r.t. `Split` is that the __getitem__ methods return the raw data, instead of a `Split` object.
    """

    @overload
    def __getitem__(self, idx: int) -> tuple[Any, Any]: ...

    @overload
    def __getitem__(self, idx: Sequence[int]) -> list[tuple[Any, Any]]: ...

    def __getitems__(self, idxs: Sequence[int]) -> list[tuple[Any, Any]]: ...


@runtime_checkable
class Dataset(Protocol):
    """An interface for a dataset that can be split into training and test sets."""

    @property
    def training_set(self) -> DatasetSplit:
        """The training set of the dataset."""

        ...

    @property
    def test_set(self) -> DatasetSplit:
        """The test set of the dataset."""

        ...
