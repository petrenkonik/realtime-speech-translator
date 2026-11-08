---
name: audio-devices
description: Inspect and control Windows audio devices for this project - list WASAPI devices and formats, find devices by name, set up VB-Cable for virtual tests, restart a device to test reconnection, and measure output latency. Use before virtual/hw tests or when a device is not found.
---

# Audio devices on this machine

Known devices (resolve by name substring + WASAPI, never by index):

| Role | Name substring | Notes |
| --- | --- | --- |
| Input | `V380` | V380 FHD Camera, USB mic |
| Output A (EN) | `Realtek` | Realtek Speakers |
| Output B (DE) | `HS100W` | Wireless headset, USB dongle, own clock, radio latency |
| Output C (fallback) | `Q2713S` | HDMI monitor |
| Virtual | `CABLE-A`, `CABLE-B` | VB-Cable A+B, test bench only |

## List devices and formats

```powershell
python -B -c "import sounddevice as sd; w=[i for i,h in enumerate(sd.query_hostapis()) if 'WASAPI' in h['name']][0]; [print(i, d['name'], d['max_input_channels'], d['max_output_channels'], d['default_samplerate']) for i,d in enumerate(sd.query_devices()) if d['hostapi']==w]"
```

Once the CLI exists: `python -B -m pipeline list-devices`.

WASAPI shared mode only opens at the rate set in Windows
(Sound -> device -> Properties -> Advanced). Requesting another rate gives
`Invalid sample rate`; the pipeline must open at `default_samplerate` and
resample itself.

## VB-Cable for virtual tests

- Install VB-Cable A+B (test machine only, never a project dependency).
- For V1 set `CABLE-A Output` to 24 bit, 48000 Hz in its Advanced tab.
- The test writes reference audio to `CABLE-A Input`; the pipeline captures
  `CABLE-A Output`. Outputs go to `CABLE-B Input`; tests record
  `CABLE-B Output`.

## Restart a device (reconnect tests V3, H2)

Needs an elevated shell. Ask the user before running; it interrupts audio.

```powershell
Get-PnpDevice -Class MEDIA | Where-Object FriendlyName -like "*HS100W*" | Format-Table InstanceId, FriendlyName, Status
pnputil /restart-device "<InstanceId>"
```

`Disable-PnpDevice` / `Enable-PnpDevice` work as well. Physical unplugging
(H2) is done by the user.

After a device comes back, PortAudio does not see it until it is
re-initialized (`sd._terminate(); sd._initialize()`).

## Known edge case

HS100W powered off with the dongle plugged in: WASAPI reports no error and
audio goes nowhere. Not detectable in software; documented limitation (T8).

## Radio latency (T6)

Play a click on output A and B at the same moment, hold the V380 mic between
the speaker and the headset earcup, record, cross-correlate the two arrivals,
take the median of 10 runs, pass it as `--radio-offset de=<ms>`.
