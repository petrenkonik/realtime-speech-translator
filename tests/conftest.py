"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pipeline.guards import StrictGuard, Violation


@pytest.fixture
def no_disk_writes() -> Iterator[list[Violation]]:
    """Watch the test for disk writes and network use; the yielded list must stay empty.

    Violations are recorded, not raised, so one assertion at the end of the test
    shows everything the code under test did. (Spec G1.)
    """
    guard = StrictGuard(on_violation=lambda violation: None)
    with guard:
        yield guard.violations
