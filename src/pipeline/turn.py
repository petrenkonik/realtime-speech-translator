"""Smart Turn: the model wrapper and the channels that connect it to the VAD chain.

``OnnxTurnClassifier`` answers "has the speaker finished?" for the end of an audio
window. ``ThreadedTurnLink`` runs it on its own thread behind a one-slot mailbox so the
audio chain never waits for it; ``SyncTurnLink`` answers on a frame count instead of a
clock and serves tests and offline runs. Both satisfy ``vad.TurnLink``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipeline.turn_features import N_FRAMES, N_MELS, WINDOW_SAMPLES, turn_features
from pipeline.vad import Float32Array, ModelNotFoundError, TurnAnswer, TurnRequest

# Relative to ``models/``. The unquantized file: it is the vendor's accuracy reference and,
# on CPUs without VNNI, as fast as the int8 one (SPEC-LOW-LATENCY.md section 4).
DEFAULT_TURN_MODEL = "smart-turn/smart-turn-v3.2-gpu.onnx"

_THREADS_RANGE = (1, 2)
_MODEL_TIME_HISTORY = 1_000
_JOIN_TIMEOUT_S = 2.0


class TurnModelError(RuntimeError):
    """The Smart Turn model could not be used."""


class TurnClassifier(Protocol):
    """Probability that the speaker has finished, given the audio so far."""

    def predict(self, window: Float32Array) -> float:
        """Return the probability in [0, 1] for the end of ``window`` (16 kHz mono)."""
        ...


class TurnSession(Protocol):
    """The part of ``onnxruntime.InferenceSession`` the classifier uses."""

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Run the model."""
        ...


def _open_session(model_path: Path, threads: int) -> TurnSession:
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Smart Turn model not found at {model_path}; run scripts/fetch_models.py first, "
            "or use --turn-mode fixed"
        )
    import onnxruntime as ort  # heavy; imported only when the model is really needed

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    # int8 matmuls on AVX2 CPUs without VNNI can saturate and shift the answers a lot
    # (measured corr 0.71 to fp32 by default, 0.88 with this); fp32 files ignore it.
    options.add_session_config_entry("session.x64quantprecision", "1")
    session: TurnSession = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session


class OnnxTurnClassifier:
    """The Smart Turn v3 ONNX model, loaded from a local file and warmed up at once."""

    def __init__(
        self, model_path: Path, threads: int = 1, session: TurnSession | None = None
    ) -> None:
        """Load the model and check that it behaves like a probability.

        Args:
            model_path: Local ``smart-turn-v3.*.onnx`` file (``DEFAULT_TURN_MODEL``).
            threads: onnxruntime intra-op threads (1-2); one keeps cores free for whisper.
            session: Injected session for tests; opened from ``model_path`` when omitted.

        Raises:
            ModelNotFoundError: The file does not exist.
            TurnModelError: Bad ``threads``, or the warm-up output is not in [0, 1].
        """
        low, high = _THREADS_RANGE
        if not low <= threads <= high:
            raise TurnModelError(f"turn threads must be {low}-{high}, got {threads}")
        self._session = session if session is not None else _open_session(model_path, threads)
        warm = self.predict(np.zeros(WINDOW_SAMPLES // 2, dtype=np.float32))
        if not 0.0 <= warm <= 1.0:
            raise TurnModelError(
                f"the model returned {warm}, which is not a probability; "
                "wrong model file? expected smart-turn-v3.x-gpu.onnx or -cpu.onnx"
            )

    def predict(self, window: Float32Array) -> float:
        """Return the probability that the turn is complete at the end of ``window``."""
        features = turn_features(window)[np.newaxis]
        assert features.shape == (1, N_MELS, N_FRAMES)
        outputs = self._session.run(None, {"input_features": features})
        return float(np.asarray(outputs[0]).reshape(-1)[0])


class SyncTurnLink:
    """Answers on the thread that asks, a fixed number of frames later.

    Time is counted in ``poll`` calls (one per VAD frame), so a run is reproducible
    regardless of how fast the classifier is. Use it in tests and offline measurements.
    """

    def __init__(self, classifier: TurnClassifier, delay_frames: int = 1) -> None:
        """Create the link; ``delay_frames`` >= 1 is the answer delay in frames."""
        if delay_frames < 1:
            raise ValueError("delay_frames must be at least 1 (an answer is never instant)")
        self._classifier = classifier
        self._delay = delay_frames
        self._frame = 0
        self._due: deque[tuple[int, TurnAnswer]] = deque()

    def submit(self, request: TurnRequest) -> None:
        """Compute the answer now; it becomes visible ``delay_frames`` polls later."""
        answer = TurnAnswer(request.id, self._classifier.predict(request.window))
        self._due.append((self._frame + self._delay, answer))

    def poll(self) -> TurnAnswer | None:
        """Advance one frame and return an answer that is due."""
        self._frame += 1
        if self._due and self._due[0][0] <= self._frame:
            return self._due.popleft()[1]
        return None

    def reset(self) -> None:
        """Forget pending answers."""
        self._due.clear()


class ThreadedTurnLink:
    """Runs the classifier on its own thread; the audio chain never waits for it.

    The mailbox holds one request: a newer one replaces an older one that the model
    has not started (the older silence run is stale anyway). ``poll`` and ``submit``
    take no lock the worker holds while predicting.
    """

    def __init__(
        self,
        classifier: TurnClassifier,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start the worker thread."""
        self._classifier = classifier
        self._clock = clock
        self._cond = threading.Condition()
        self._mailbox: TurnRequest | None = None
        self._generation = 0
        self._stopping = False
        self._answers: deque[TurnAnswer] = deque()
        self._model_ms: deque[float] = deque(maxlen=_MODEL_TIME_HISTORY)
        self.superseded = 0
        self.errors = 0
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name="turn-model", daemon=True)
        self._thread.start()

    def submit(self, request: TurnRequest) -> None:
        """Post a request; replaces one the model has not started yet."""
        with self._cond:
            if self._mailbox is not None:
                self.superseded += 1
            self._mailbox = request
            self._cond.notify()

    def poll(self) -> TurnAnswer | None:
        """Return the oldest finished answer, if any."""
        try:
            return self._answers.popleft()
        except IndexError:
            return None

    def reset(self) -> None:
        """Drop the mailbox and every answer; a prediction already running is discarded."""
        with self._cond:
            self._generation += 1
            self._mailbox = None
        self._answers.clear()

    def model_times_ms(self) -> list[float]:
        """Recent prediction times (the last 1000), oldest first."""
        return list(self._model_ms)

    def close(self) -> None:
        """Stop the worker thread."""
        with self._cond:
            self._stopping = True
            self._mailbox = None
            self._cond.notify()
        self._thread.join(_JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._mailbox is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                request, self._mailbox = self._mailbox, None
                generation = self._generation
            assert request is not None
            started = self._clock()
            try:
                probability = self._classifier.predict(request.window)
            except Exception as error:  # a failing model must never take the thread down
                self.errors += 1
                self.last_error = f"{type(error).__name__}: {error}"
                continue  # no answer: the silence rule is the fallback
            self._model_ms.append((self._clock() - started) * 1000.0)
            with self._cond:
                if generation == self._generation:
                    self._answers.append(TurnAnswer(request.id, probability))
