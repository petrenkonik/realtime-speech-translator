"""Device lookup by name (spec F4 and the list-devices table)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from fakes import FakeDeviceList, make_device

from pipeline.devices import (
    AmbiguousDeviceError,
    DeviceNotFoundError,
    SoundDeviceListing,
    StreamOpenError,
    find_device,
    format_device_table,
    open_sounddevice_input,
    open_sounddevice_output,
)

pytestmark = pytest.mark.unit

DEVICES = [
    make_device("Microphone (V380 USB)", index=1, inputs=1, rate_hz=48_000),
    make_device("Speakers (Realtek Audio)", index=2, inputs=0, outputs=2),
    make_device("Headset Mic (HS100W)", index=3, inputs=1, rate_hz=44_100),
    make_device("Microphone (V380 USB)", index=4, inputs=1, host_api="MME"),
]


def test_substring_match_is_case_insensitive() -> None:
    assert find_device(DEVICES, "v380").index == 1


def test_only_the_requested_host_api_is_considered() -> None:
    # Index 4 has the same name on MME; WASAPI must win, never the duplicate.
    assert find_device(DEVICES, "V380 USB").host_api == "Windows WASAPI"


def test_output_only_devices_are_not_inputs() -> None:
    with pytest.raises(DeviceNotFoundError, match="Realtek"):
        find_device(DEVICES, "Realtek", kind="input")


def test_output_lookup_finds_output_devices() -> None:
    assert find_device(DEVICES, "Realtek", kind="output").index == 2


def test_missing_device_lists_the_candidates() -> None:
    with pytest.raises(DeviceNotFoundError) as info:
        find_device(DEVICES, "Nonexistent")
    assert "V380" in str(info.value)
    assert "HS100W" in str(info.value)


def test_ambiguous_substring_is_an_error_not_a_guess() -> None:
    devices = [make_device("Mic A (USB)", index=1), make_device("Mic B (USB)", index=2)]
    with pytest.raises(AmbiguousDeviceError, match="longer name"):
        find_device(devices, "USB")


def test_an_exact_name_wins_over_longer_names() -> None:
    devices = [make_device("Mic", index=1), make_device("Mic Array", index=2)]
    assert find_device(devices, "mic").index == 1


def test_empty_query_is_rejected() -> None:
    with pytest.raises(DeviceNotFoundError, match="empty"):
        find_device(DEVICES, "  ")


def test_f4_same_device_is_found_after_the_list_is_reordered() -> None:
    listing = FakeDeviceList(DEVICES[:3])
    before = find_device(listing.list_devices(), "V380")
    listing.renumber(["Headset Mic (HS100W)", "Speakers (Realtek Audio)", "Microphone (V380 USB)"])
    after = find_device(listing.list_devices(), "V380")
    assert after.name == before.name
    assert after.index != before.index


def test_device_table_shows_only_the_host_api_with_channels_and_rate() -> None:
    table = format_device_table(DEVICES)
    lines = table.splitlines()
    assert lines[0].split() == ["in", "out", "rate", "name"]
    assert len(lines) == 4  # header + three WASAPI devices; the MME duplicate is hidden
    assert "48000" in table
    assert "Headset Mic (HS100W)" in table


def test_sounddevice_listing_reports_names_and_reinitializes() -> None:
    class FakeSd:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def query_hostapis(self) -> list[dict[str, object]]:
            return [{"name": "MME"}, {"name": "Windows WASAPI"}]

        def query_devices(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "Mic",
                    "index": 7,
                    "hostapi": 1,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 48000.0,
                }
            ]

        def _terminate(self) -> None:
            self.calls.append("terminate")

        def _initialize(self) -> None:
            self.calls.append("initialize")

    fake = FakeSd()
    listing = SoundDeviceListing()
    listing._sd = fake  # inject without importing PortAudio
    (device,) = listing.list_devices()
    assert (device.name, device.index, device.host_api) == ("Mic", 7, "Windows WASAPI")
    assert device.default_sample_rate_hz == 48000
    listing.reinitialize()
    assert fake.calls == ["terminate", "initialize"]


# ---------------------------------------------------------------- PortAudio adapter


class _FakePortAudioError(Exception):
    pass


class _FakeStatus:
    def __init__(self, overflow: bool) -> None:
        self.input_overflow = overflow
        self.output_underflow = overflow


class _FakeSdStream:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.active = False
        self.raise_on: str | None = None

    def _maybe_fail(self, what: str) -> None:
        if self.raise_on == what:
            raise _FakePortAudioError(f"{what} failed")

    def start(self) -> None:
        self._maybe_fail("start")
        self.active = True

    def stop(self) -> None:
        self._maybe_fail("stop")
        self.active = False

    def close(self) -> None:
        self._maybe_fail("close")


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = SimpleNamespace(PortAudioError=_FakePortAudioError, created=[], refuse=False)

    def input_stream(**kwargs: object) -> _FakeSdStream:
        if module.refuse:
            raise _FakePortAudioError("device busy")
        stream = _FakeSdStream(**kwargs)
        module.created.append(stream)
        return stream

    module.InputStream = input_stream
    module.OutputStream = input_stream
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def test_the_adapter_opens_the_device_at_its_native_format(fake_sd: SimpleNamespace) -> None:
    device = make_device("Mic", index=9, rate_hz=44_100, inputs=2)
    open_sounddevice_input(device, lambda block, overflow: None, 441)
    kwargs = fake_sd.created[0].kwargs
    assert kwargs["device"] == 9
    assert kwargs["samplerate"] == 44_100
    assert kwargs["channels"] == 2
    assert kwargs["dtype"] == "float32"
    assert kwargs["blocksize"] == 441
    assert kwargs["latency"] == "low"


def test_the_adapter_passes_blocks_and_the_overflow_flag_to_our_callback(
    fake_sd: SimpleNamespace,
) -> None:
    seen: list[tuple[object, bool]] = []
    open_sounddevice_input(make_device("Mic"), lambda block, flag: seen.append((block, flag)), 480)
    sd_callback = fake_sd.created[0].kwargs["callback"]
    sd_callback("audio-1", 480, None, _FakeStatus(False))
    sd_callback("audio-2", 480, None, _FakeStatus(True))
    assert seen == [("audio-1", False), ("audio-2", True)]


def test_the_adapter_maps_portaudio_errors_to_stream_open_error(fake_sd: SimpleNamespace) -> None:
    fake_sd.refuse = True
    with pytest.raises(StreamOpenError, match="device busy"):
        open_sounddevice_input(make_device("Mic"), lambda block, flag: None, 480)


@pytest.mark.parametrize("operation", ["start", "stop", "close"])
def test_stream_operations_raise_stream_open_error(
    fake_sd: SimpleNamespace, operation: str
) -> None:
    stream = open_sounddevice_input(make_device("Mic"), lambda block, flag: None, 480)
    fake_sd.created[0].raise_on = operation
    with pytest.raises(StreamOpenError, match=operation):
        getattr(stream, operation)()


def test_the_adapter_stream_reports_active_state(fake_sd: SimpleNamespace) -> None:
    stream = open_sounddevice_input(make_device("Mic"), lambda block, flag: None, 480)
    assert not stream.active
    stream.start()
    assert stream.active
    stream.stop()
    stream.close()
    assert not stream.active


def test_the_output_adapter_opens_at_most_two_channels_at_the_native_rate(
    fake_sd: SimpleNamespace,
) -> None:
    device = make_device("Speakers", index=4, rate_hz=48_000, inputs=0, outputs=8)
    open_sounddevice_output(device, lambda out, flag: None, 480)
    kwargs = fake_sd.created[0].kwargs
    assert kwargs["device"] == 4
    assert kwargs["samplerate"] == 48_000
    assert kwargs["channels"] == 2
    assert kwargs["dtype"] == "float32"
    assert kwargs["latency"] == "low"


def test_the_output_adapter_opens_a_mono_device_with_one_channel(fake_sd: SimpleNamespace) -> None:
    device = make_device("Mono speaker", inputs=0, outputs=1)
    open_sounddevice_output(device, lambda out, flag: None, 480)
    assert fake_sd.created[0].kwargs["channels"] == 1


def test_the_output_adapter_passes_the_buffer_and_the_underflow_flag(
    fake_sd: SimpleNamespace,
) -> None:
    seen: list[tuple[object, bool]] = []
    open_sounddevice_output(
        make_device("Speakers", inputs=0, outputs=2),
        lambda out, flag: seen.append((out, flag)),
        480,
    )
    sd_callback = fake_sd.created[0].kwargs["callback"]
    sd_callback("buffer-1", 480, None, _FakeStatus(False))
    sd_callback("buffer-2", 480, None, _FakeStatus(True))
    assert seen == [("buffer-1", False), ("buffer-2", True)]


def test_the_output_adapter_maps_portaudio_errors(fake_sd: SimpleNamespace) -> None:
    fake_sd.refuse = True
    with pytest.raises(StreamOpenError, match="device busy"):
        open_sounddevice_output(
            make_device("Speakers", inputs=0, outputs=2), lambda o, f: None, 480
        )
