"""CPU load for the soak run (S1): keeps N busy worker processes alive.

Threads would share one GIL and leave the other cores idle, so each worker is its own
process. They multiply small matrices in a loop (compute and cache traffic, like a real
application) and exit by themselves after ``--seconds``, so a crashed parent cannot leave them
running forever.

    python -B scripts/stress.py --cores 7 --seconds 3600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_WORKER = """
import time
import numpy as np
deadline = time.monotonic() + {seconds}
rng = np.random.default_rng()
a = rng.standard_normal((192, 192)).astype(np.float32)
b = rng.standard_normal((192, 192)).astype(np.float32)
while time.monotonic() < deadline:
    for _ in range(200):
        c = a @ b
        a = c / (np.abs(c).max() + 1.0)
"""


def start_workers(
    cores: int, seconds: float, *, below_normal: bool
) -> list[subprocess.Popen[bytes]]:
    """Start ``cores`` busy processes; each ends after ``seconds`` on its own.

    Args:
        cores: Number of workers.
        seconds: How long each one runs.
        below_normal: Run at below-normal priority so the rest of the desktop stays usable.
    """
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    flags = BELOW_NORMAL_PRIORITY_CLASS if below_normal and sys.platform == "win32" else 0
    code = _WORKER.format(seconds=repr(float(seconds)))
    return [
        subprocess.Popen(
            [sys.executable, "-B", "-c", code],
            env=environment,
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(cores)
    ]


def stop_workers(workers: list[subprocess.Popen[bytes]]) -> None:
    """End the workers now."""
    for worker in workers:
        worker.terminate()
    deadline = time.monotonic() + 5.0
    for worker in workers:
        try:
            worker.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            worker.kill()


def main() -> int:
    """Run the workers until they finish or Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--normal-priority", action="store_true", help="default is below normal")
    args = parser.parse_args()
    workers = start_workers(args.cores, args.seconds, below_normal=not args.normal_priority)
    try:
        for worker in workers:
            worker.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_workers(workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
