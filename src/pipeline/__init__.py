"""Offline real-time interpreter pipeline.

One microphone and one source language are transcribed once per utterance,
then translated and spoken in parallel per-language lanes, each on its own
output device. Everything runs offline on the CPU and keeps audio and text
in RAM only.
"""
