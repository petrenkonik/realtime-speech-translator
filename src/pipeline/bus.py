"""Multi-consumer bus: one bounded queue per subscriber, publishing never blocks.

``ChunkBus`` carries ``Chunk`` objects from the segmenter to whisper, the meter and
the demo slow consumer; ``TextBus`` carries utterances to the language lanes. They
are the same class. A slow consumer must never delay the producer (the audio path)
or another consumer, so each subscriber owns a bounded queue and, when it is full,
loses an item instead of making the publisher wait.

Items are shared by reference, never copied: ``Chunk`` is immutable (its samples are
read-only), which is what makes sharing safe.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pipeline.vad import Chunk

T = TypeVar("T")

OverflowPolicy = Literal["drop_oldest", "drop_newest"]
_POLICIES = ("drop_oldest", "drop_newest")

DEFAULT_MAXSIZE = 32


class BusConfigError(ValueError):
    """A subscription was requested with invalid arguments."""


class BusClosedError(RuntimeError):
    """An operation needs an open bus, but ``close()`` was already called."""


@dataclass(frozen=True, slots=True)
class SubscriptionStats:
    """Counters of one subscriber, for the dashboard and for tests.

    Attributes:
        delivered: Items handed to the consumer by ``get()``.
        dropped: Items lost to the overflow policy (not counting a discarded
            backlog at ``close()``).
        depth: Items waiting right now.
        max_depth: Highest depth seen so far.
    """

    delivered: int
    dropped: int
    depth: int
    max_depth: int


class Subscription(Generic[T]):
    """The consumer side of one subscriber: a bounded queue with blocking ``get()``.

    Created by ``Bus.subscribe()``; do not construct directly.
    """

    def __init__(self, name: str, maxsize: int, policy: OverflowPolicy) -> None:
        """Create an empty queue; ``Bus.subscribe()`` has already validated the arguments."""
        self._name = name
        self._maxsize = maxsize
        self._policy = policy
        self._cond = threading.Condition()
        self._items: deque[T] = deque()
        self._closed = False
        self._delivered = 0
        self._dropped = 0
        self._max_depth = 0

    @property
    def name(self) -> str:
        """The subscriber name given to ``Bus.subscribe()``."""
        return self._name

    @property
    def finished(self) -> bool:
        """True once the bus is closed and nothing is left to read."""
        with self._cond:
            return self._closed and not self._items

    def get(self, timeout_s: float | None = None) -> T | None:
        """Take the oldest item, waiting for one if needed.

        Args:
            timeout_s: Longest wait in seconds; ``None`` waits until an item
                arrives or the bus closes.

        Returns:
            The item, or ``None`` when the bus is closed and drained, or when the
            timeout passed first. Use ``finished`` to tell the two ``None`` cases
            apart.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while not self._items:
                if self._closed:
                    return None
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return None
                self._cond.wait(remaining_s)
            self._delivered += 1
            return self._items.popleft()

    def __iter__(self) -> Iterator[T]:
        """Yield items until the bus is closed and drained."""
        while (item := self.get()) is not None:
            yield item

    def stats(self) -> SubscriptionStats:
        """Return a consistent snapshot of the counters."""
        with self._cond:
            return SubscriptionStats(
                delivered=self._delivered,
                dropped=self._dropped,
                depth=len(self._items),
                max_depth=self._max_depth,
            )

    def _offer(self, item: T) -> None:
        # Called by the publisher. Holds the lock only for a few instructions and
        # never waits for the consumer.
        with self._cond:
            if self._closed:
                return
            if len(self._items) >= self._maxsize:
                self._dropped += 1
                if self._policy == "drop_newest":
                    return
                self._items.popleft()
            self._items.append(item)
            self._max_depth = max(self._max_depth, len(self._items))
            self._cond.notify()

    def _close(self, drain: bool) -> None:
        with self._cond:
            self._closed = True
            if not drain:
                self._items.clear()
            self._cond.notify_all()


class Bus(Generic[T]):
    """Fan-out from any number of publishers to named subscribers.

    ``publish()`` never blocks on a consumer. Each subscriber sees every item
    published after it subscribed, in order, unless its own queue overflowed.
    """

    def __init__(self) -> None:
        """Create an open bus without subscribers."""
        self._lock = threading.Lock()
        # Replaced as a whole (copy-on-write), so ``publish()`` can iterate it
        # without taking the lock.
        self._subscriptions: tuple[Subscription[T], ...] = ()
        self._closed = False
        self._count_lock = threading.Lock()
        self._published = 0

    @property
    def closed(self) -> bool:
        """True after ``close()``."""
        return self._closed

    @property
    def published(self) -> int:
        """Number of items accepted by ``publish()`` (not counting after close)."""
        with self._count_lock:
            return self._published

    def subscribe(
        self,
        name: str,
        maxsize: int = DEFAULT_MAXSIZE,
        policy: OverflowPolicy = "drop_oldest",
    ) -> Subscription[T]:
        """Add a subscriber that sees every item published from now on.

        Args:
            name: Unique, non-empty name used in the statistics.
            maxsize: Queue capacity (at least 1).
            policy: What to lose when the queue is full: ``"drop_oldest"`` (the
                default; fresh speech matters more than old) or ``"drop_newest"``.

        Returns:
            The ``Subscription`` the consumer reads from.

        Raises:
            BusConfigError: Invalid name, size or policy, or a duplicate name.
            BusClosedError: The bus was already closed.
        """
        if not name:
            raise BusConfigError("subscriber name must not be empty")
        if maxsize < 1:
            raise BusConfigError(f"maxsize must be at least 1, got {maxsize}")
        if policy not in _POLICIES:
            raise BusConfigError(f"unknown policy {policy!r}; use one of {_POLICIES}")
        with self._lock:
            if self._closed:
                raise BusClosedError("cannot subscribe: the bus is closed")
            if any(sub.name == name for sub in self._subscriptions):
                raise BusConfigError(f"subscriber {name!r} already exists")
            subscription: Subscription[T] = Subscription(name, maxsize, policy)
            self._subscriptions = (*self._subscriptions, subscription)
            return subscription

    def unsubscribe(self, subscription: Subscription[T]) -> bool:
        """Remove one subscriber; the others and the bus itself are not affected.

        The subscription is closed with its backlog discarded, so a consumer blocked in
        ``get()`` wakes up and sees ``finished``. Its name can be used again.

        Args:
            subscription: A value returned by this bus's ``subscribe()``.

        Returns:
            True if it was removed, False if it was not subscribed here (already removed,
            or from another bus); nothing happens in that case.
        """
        with self._lock:
            if subscription not in self._subscriptions:
                return False
            self._subscriptions = tuple(
                sub for sub in self._subscriptions if sub is not subscription
            )
        subscription._close(drain=False)
        return True

    def publish(self, item: T) -> None:
        """Offer ``item`` to every subscriber. Never blocks on a consumer.

        After ``close()`` the item is silently ignored: the audio thread can race
        with shutdown and must not crash because of it.
        """
        if self._closed:
            return
        with self._count_lock:
            self._published += 1
        for subscription in self._subscriptions:
            subscription._offer(item)

    def stats(self) -> dict[str, SubscriptionStats]:
        """Return the counters of every subscriber by name."""
        return {sub.name: sub.stats() for sub in self._subscriptions}

    def close(self, drain: bool = False) -> None:
        """End the stream for every subscriber. Safe to call more than once.

        Args:
            drain: ``False`` (default) discards queued items, so every consumer
                stops as soon as it finishes the item it is working on. ``True``
                lets consumers read the backlog first.
        """
        with self._lock:
            self._closed = True
            subscriptions = self._subscriptions
        for subscription in subscriptions:
            subscription._close(drain)


ChunkBus = Bus[Chunk]
"""Bus of finished audio chunks (segmenter -> whisper, meter, demo consumer)."""
