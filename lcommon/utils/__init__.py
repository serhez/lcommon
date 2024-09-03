import numpy as np

from .context import merge_conversation, merge_system_messages, parse_context
from .decomposition import DependencySyntax
from .graphs import DAG, DirectedGraph, Edge, Node
from .logging import NullLogger
from .prompting import (
    BULLET_POINTS_CHARS,
    END_CHARS,
    BaseShotsCollection,
    construct_shots_str,
    read_prompt,
)
from .threading import GuardedValue, safe_exec_with_lock
from .usage import Usage

__all__ = [
    "Edge",
    "Node",
    "DirectedGraph",
    "DAG",
    "IDGenerator",
    "DependencySyntax",
    "GuardedValue",
    "BaseShotsCollection",
    "Usage",
    "read_prompt",
    "construct_shots_str",
    "NullLogger",
    "merge_conversation",
    "merge_system_messages",
    "parse_context",
    "classproperty",
    "safe_exec_with_lock",
    "BULLET_POINTS_CHARS",
    "END_CHARS",
]


# Damn you, Python developers...
class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)  # type: ignore[reportOptionalCall]


class IDGenerator:
    def __init__(self, max_retries: int = 10):
        """
        Initializes the ID generator.

        ### Parameters
        ----------
        `max_retries`: the maximum number of attempts to generate a unique random ID.
        """

        self._max_retries = max_retries
        self._cache: list[int] = []
        self._current = 0

    def reset(self):
        """Resets the cache of the generated IDs."""

        self._cache = []
        self._current = 0

    def next(self) -> str:
        """
        Generates the next unique ID.

        ### Returns
        ----------
        The new unique ID.

        ### Notes
        ----------
        If, by chance, the current ID is already in the cache (perhaps due to a previous call to `random`), the method will generate a random ID.
        """

        if self._current in self._cache:
            self._current += 1
            return self.random()

        self._cache.append(self._current)
        self._current += 1

        return str(self._current - 1)

    def random(self) -> str:
        """
        Generates a random unique ID.

        ### Returns
        ----------
        The new unique ID.

        ### Raises
        ----------
        `RuntimeError`: if the ID of the object could not be generated after the specified number of attempts.
        """

        id = None

        for i in range(self._max_retries):
            id = np.random.randint(0, int(1e9))

            if id not in self._cache:
                break

            id = None

        if id is None:
            raise RuntimeError(
                f"[gen_id] The ID of the problem could not be generated after {self._max_retries} attempts."
            )

        self._cache.append(id)

        return str(id)
