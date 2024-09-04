import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class GuardedValue(Generic[T]):
    """A value guarded by a lock"""

    value: T
    """The value to be guarded."""

    lock: threading.Lock = threading.Lock()
    """The lock to be acquired before accessing the value."""


def safe_exec_with_lock(lock: threading.Lock, func, *args, **kwargs):
    """
    Execute a function with a lock acquired.

    ### Parameters
    --------------
    `lock`: the lock to be acquired.
    `func`: the function to be executed.
    `*args`: the positional arguments to be passed to the function.
    `**kwargs`: the keyword arguments to be passed to the function.

    ### Returns
    --------------
    The return value of the function.
    """

    lock.acquire()
    res = None
    try:
        res = func(*args, **kwargs)
    finally:
        lock.release()
    return res
