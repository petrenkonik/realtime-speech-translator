"""Audio device lookup by name on the WASAPI host API.

Devices are never addressed by index: Windows renumbers them when a USB device is
unplugged, replugged or another one appears. The name (a case-insensitive
substring) plus the host API identifies a device across reconnects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

WASAPI_HOST_API = "Windows WASAPI"
DeviceKind = Literal["input", "output"]


class DeviceNotFoundError(LookupError):
    """No device matched the requested name."""


class AmbiguousDeviceError(LookupError):
    """More than one device matched and none matched exactly."""


class StreamOpenError(RuntimeError):
    """A device was found but its stream could not be opened or has failed."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What the program needs to know about one audio endpoint.

    Attributes:
        name: Device name as the host API reports it.
        index: PortAudio index; valid only until the next device-list refresh.
        host_api: Host API name, for example ``"Windows WASAPI"``.
        max_input_channels: Zero for output-only devices.
        max_output_channels: Zero for input-only devices.
        default_sample_rate_hz: The device's native rate (the shared-mode mix format).
    """

    name: str
    index: int
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate_hz: int


class DeviceListing(Protocol):
    """Source of the current device list; replaced by a fake in tests."""

    def list_devices(self) -> Sequence[DeviceInfo]:
        """Return a fresh snapshot of all devices."""
        ...

    def reinitialize(self) -> None:
        """Re-create the audio library state so unplugged/replugged devices show up."""
        ...


def find_device(
    devices: Sequence[DeviceInfo],
    query: str,
    kind: DeviceKind = "input",
    host_api: str = WASAPI_HOST_API,
) -> DeviceInfo:
    """Pick the device whose name contains ``query``.

    An exact (case-insensitive) name match wins over substring matches. Without one,
    exactly one substring match is required: guessing between two microphones would
    silently transcribe the wrong one.

    Args:
        devices: Snapshot from ``DeviceListing.list_devices()``.
        query: Name or part of the name.
        kind: ``"input"`` or ``"output"``; devices without channels of that kind
            are ignored.
        host_api: Only devices of this host API are considered.

    Returns:
        The matching device.

    Raises:
        DeviceNotFoundError: Nothing matched. The message lists the candidates.
        AmbiguousDeviceError: Several substring matches and no exact one.
    """
    wanted = query.strip().casefold()
    if not wanted:
        raise DeviceNotFoundError("device name must not be empty")
    candidates = [
        device
        for device in devices
        if device.host_api == host_api
        and (device.max_input_channels if kind == "input" else device.max_output_channels) > 0
    ]
    exact = [device for device in candidates if device.name.casefold() == wanted]
    if len(exact) == 1:
        return exact[0]
    matches = exact or [device for device in candidates if wanted in device.name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(repr(device.name) for device in candidates) or "none"
        raise DeviceNotFoundError(
            f"no {kind} device matching {query!r} on {host_api}; available: {available}"
        )
    names = ", ".join(repr(device.name) for device in matches)
    raise AmbiguousDeviceError(
        f"{query!r} matches several {kind} devices ({names}); use a longer name"
    )


def format_device_table(devices: Sequence[DeviceInfo], host_api: str = WASAPI_HOST_API) -> str:
    """Render the ``list-devices`` output: one line per device of ``host_api``."""
    lines = [f"{'in':>3} {'out':>3} {'rate':>6}  name"]
    for device in devices:
        if device.host_api != host_api:
            continue
        lines.append(
            f"{device.max_input_channels:>3} {device.max_output_channels:>3} "
            f"{device.default_sample_rate_hz:>6}  {device.name}"
        )
    return "\n".join(lines)


class SoundDeviceListing:
    """``DeviceListing`` backed by PortAudio through ``sounddevice``.

    The import is lazy so that unit tests and ``--help`` never load PortAudio.
    """

    def __init__(self) -> None:
        """Create a listing; PortAudio is touched on first use."""
        self._sd: Any = None

    def _module(self) -> Any:  # noqa: ANN401 - sounddevice ships no type information
        if self._sd is None:
            import sounddevice

            self._sd = sounddevice
        return self._sd

    def list_devices(self) -> Sequence[DeviceInfo]:
        """Return all devices with the name of their host API."""
        sd = self._module()
        host_apis = [api["name"] for api in sd.query_hostapis()]
        return [
            DeviceInfo(
                name=str(raw["name"]),
                index=int(raw["index"]),
                host_api=str(host_apis[raw["hostapi"]]),
                max_input_channels=int(raw["max_input_channels"]),
                max_output_channels=int(raw["max_output_channels"]),
                default_sample_rate_hz=int(raw["default_samplerate"]),
            )
            for raw in sd.query_devices()
        ]

    def reinitialize(self) -> None:
        """Terminate and initialize PortAudio again to rescan the devices.

        PortAudio caches the device list at initialization; without this a device
        that was unplugged and replugged is not found again.
        """
        sd = self._module()
        sd._terminate()
        sd._initialize()


BlockCallback = Callable[[Any, bool], None]
"""Called with one block of shape ``(frames, channels)`` and the input-overflow flag."""


class AudioStream(Protocol):
    """The part of an input or output stream the program uses."""

    @property
    def active(self) -> bool:
        """False once the stream stopped or failed."""
        ...

    def start(self) -> None:
        """Begin delivering blocks to the callback."""
        ...

    def stop(self) -> None:
        """Stop delivering blocks."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


StreamFactory = Callable[[DeviceInfo, BlockCallback, int], AudioStream]
"""``(device, callback, blocksize_frames) -> stream``: opens a device at its native format."""


def open_sounddevice_input(
    device: DeviceInfo, callback: BlockCallback, blocksize_frames: int
) -> AudioStream:
    """Open a WASAPI shared-mode float32 input stream at the device's native format.

    We resample ourselves instead of letting Windows do it, hence the native rate
    and channel count. float32 also holds 24-bit samples exactly.

    Raises:
        StreamOpenError: PortAudio refused the device.
    """
    import sounddevice as sd

    def on_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:  # noqa: ANN401
        callback(indata, bool(status.input_overflow))

    try:
        stream = sd.InputStream(
            device=device.index,
            samplerate=device.default_sample_rate_hz,
            channels=device.max_input_channels,
            dtype="float32",
            blocksize=blocksize_frames,
            latency="low",
            callback=on_audio,
        )
    except sd.PortAudioError as error:
        raise StreamOpenError(f"cannot open {device.name!r}: {error}") from error
    return _SoundDeviceStream(stream, sd.PortAudioError)


OutputCallback = Callable[[Any, bool], None]
"""Called with an output buffer of shape ``(frames, channels)`` to fill in place, and the
driver's underflow flag."""

OutputStreamFactory = Callable[[DeviceInfo, OutputCallback, int], AudioStream]
"""``(device, callback, blocksize_frames) -> stream`` for an output device."""

OUTPUT_CHANNELS_MAX = 2


def open_sounddevice_output(
    device: DeviceInfo, callback: OutputCallback, blocksize_frames: int
) -> AudioStream:
    """Open a WASAPI shared-mode float32 output stream at the device's native rate.

    At most two channels are opened: speech is mono, and filling a 7.1 device with eight
    copies would only waste bandwidth. PortAudio converts float32 to the device format.

    Raises:
        StreamOpenError: PortAudio refused the device.
    """
    import sounddevice as sd

    def on_audio(outdata: Any, frames: int, time_info: Any, status: Any) -> None:  # noqa: ANN401
        callback(outdata, bool(status.output_underflow))

    try:
        stream = sd.OutputStream(
            device=device.index,
            samplerate=device.default_sample_rate_hz,
            channels=min(OUTPUT_CHANNELS_MAX, device.max_output_channels),
            dtype="float32",
            blocksize=blocksize_frames,
            latency="low",
            callback=on_audio,
        )
    except sd.PortAudioError as error:
        raise StreamOpenError(f"cannot open {device.name!r}: {error}") from error
    return _SoundDeviceStream(stream, sd.PortAudioError)


class _SoundDeviceStream:
    """Maps PortAudio errors to ``StreamOpenError`` so callers need not know the library."""

    def __init__(self, stream: Any, error_type: type[Exception]) -> None:  # noqa: ANN401
        self._stream = stream
        self._error_type = error_type

    @property
    def active(self) -> bool:
        return bool(self._stream.active)

    def start(self) -> None:
        try:
            self._stream.start()
        except self._error_type as error:
            raise StreamOpenError(f"cannot start the stream: {error}") from error

    def stop(self) -> None:
        try:
            self._stream.stop()
        except self._error_type as error:
            raise StreamOpenError(f"cannot stop the stream: {error}") from error

    def close(self) -> None:
        try:
            self._stream.close()
        except self._error_type as error:
            raise StreamOpenError(f"cannot close the stream: {error}") from error
