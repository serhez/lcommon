from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .dataset import Dataset
from .usage import UsageInterface


@dataclass
class ProviderUsage(UsageInterface, Protocol):
    """An interface for a class that stores usage statistics of a provider."""

    n_calls: int
    """The number of calls to the provider's generation process."""

    n_tokens_context: int
    """The number of tokens in the context."""

    n_tokens_output: int
    """The number of tokens in the output."""

    def __add__(self, other: ProviderUsage | None) -> ProviderUsage: ...

    def __radd__(self, other: ProviderUsage | None) -> ProviderUsage: ...

    def __iadd__(self, other: ProviderUsage | None) -> ProviderUsage: ...


class Provider(Protocol):
    """
    An interface for a provider used within a method.
    A provider provides generated text based on a given text context.
    The implementation of a provider can be a language model or a method composed by other methods or language models.
    """

    @dataclass
    class GenerationInfo(Protocol):
        """Extra information about the generation process of the provider."""

        usage: ProviderUsage
        """The usage statistics of the provider."""

        def __add__(
            self, other: Provider.GenerationInfo | None
        ) -> Provider.GenerationInfo: ...

        def __radd__(
            self, other: Provider.GenerationInfo | None
        ) -> Provider.GenerationInfo: ...

        def __iadd__(
            self, other: Provider.GenerationInfo | None
        ) -> Provider.GenerationInfo: ...

    def generate(
        self,
        context: str
        | dict[str, str]
        | list[str]
        | list[dict[str, str]]
        | list[list[str]]
        | list[list[dict[str, str]]]
        | npt.NDArray[np.str_]
        | Dataset,
        n_samples: int = 1,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], Provider.GenerationInfo]:
        """
        Generate text from a given set of contexts.

        ### Definitions
        ---------------
        - A single independent message is the smallest unit of input.
            - Represented by a single string or dictionary.
            - Dictionaries allow to add provider-specific fields, such as `role` for OpenAI's models.
            - Dictionaries can contain any number of fields, but the `content` field is required and contains the message's content.
        - A single conversation of dependent messages is a list of messages, from which only a single output is generated.
            - Represented by a list of strings/dictionaries.
        - Multiple messages/conversations yield multiple outputs.
            - Represented by a list of lists of strings/dictionaries.

        ### Parameters
        --------------
        `context`: the context to generate text from.
        `n_samples`: the number of samples to generate.
        `max_tokens`: the maximum number of tokens to generate.
        `unsafe`: whether to use the unsafe generation method.
        Other keyword arguments may be used by each specific provider; refer to the provider's documentation for more information.

        ### Returns
        -----------
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - Extra information about the generation process of the model.
        """

        ...
