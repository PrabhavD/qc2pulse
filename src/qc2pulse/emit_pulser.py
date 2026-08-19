"""Emit Pulser ``Sequence`` objects from analog pulse IR.

The IR is SI (seconds, rad/s, meters); Pulser works in nanoseconds, rad/us, and micrometers,
so this module converts. Durations are additionally rounded up to the channel clock period and
the channel minimum duration, which is why the returned duration comes from the sequence itself
rather than from the IR.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

from ._errors import EmitterNotSupportedError, MissingExtraError
from ._physics import DEFAULT_MAX_INTERACTION_RATIO, PULSER_C6_TO_SI, max_interaction_ratio
from .compile import read_pulse

__all__ = ["to_pulser"]

_RYDBERG_BASIS = "ground-rydberg"


def _import_pulser():
    try:
        from pulser import Pulse, Register, Sequence
        from pulser.waveforms import CompositeWaveform, ConstantWaveform, RampWaveform
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtraError("to_pulser needs Pulser: pip install 'qc2pulse[pulser]'") from exc
    return Pulse, Register, Sequence, CompositeWaveform, ConstantWaveform, RampWaveform


def _find_channel(device: Any, addressing: str) -> str | None:
    channels = getattr(device, "channels", None)
    if not isinstance(channels, Mapping):
        raise EmitterNotSupportedError(
            f"device {device!r} has no channels mapping; pass a Pulser Device such as "
            "pulser.devices.MockDevice"
        )
    for channel_id, channel in channels.items():
        basis = getattr(channel, "basis", None)
        if basis == _RYDBERG_BASIS and getattr(channel, "addressing", None) == addressing:
            return channel_id
    return None


def _round_up(nanoseconds: int, clock_period: int) -> int:
    if clock_period <= 1:
        return nanoseconds
    return int(math.ceil(nanoseconds / clock_period)) * clock_period


def _to_ns(seconds: float, clock_period: int) -> int:
    """Convert seconds to an integer nanosecond count on the channel clock grid."""
    nanoseconds = int(math.ceil(seconds * 1e9 - 1e-6))
    return _round_up(max(nanoseconds, 0), clock_period)


def _check_amplitude(channel: Any, amplitude_rad_per_us: float, channel_id: str) -> None:
    max_amp = getattr(channel, "max_amp", None)
    if max_amp is not None and amplitude_rad_per_us > float(max_amp) + 1e-12:
        raise EmitterNotSupportedError(
            f"Rabi amplitude {amplitude_rad_per_us:.4f} rad/us exceeds the "
            f"{channel_id} channel limit of {float(max_amp):.4f} rad/us"
        )


def _check_interactions(
    register_coords: list[list[float]],
    segments: list[dict[str, Any]],
    device: Any,
    *,
    allow_interacting: bool,
    ratio_limit: float,
) -> None:
    """Fail closed when a global X pulse is outside the independent-atom regime."""
    global_peak = max(
        (float(segment["amplitude"]) for segment in segments if segment.get("site") is None),
        default=0.0,
    )
    if global_peak <= 0.0:
        return

    coefficient = getattr(device, "interaction_coeff", None)
    if coefficient is None:
        message = (
            f"device {getattr(device, 'name', device)!r} does not expose interaction_coeff, "
            "so qc2pulse cannot verify that the global pi pulse prepares independent atoms"
        )
        if not allow_interacting:
            raise EmitterNotSupportedError(f"{message}; pass allow_interacting=True to opt in")
        warnings.warn(message, UserWarning, stacklevel=3)
        return

    ratio = max_interaction_ratio(
        register_coords, global_peak, float(coefficient) * PULSER_C6_TO_SI
    )
    if ratio <= ratio_limit:
        return

    message = (
        f"the strongest Rydberg interaction is {ratio:.3g} times the global Rabi amplitude, "
        f"above max_interaction_ratio={ratio_limit:.3g}; the global pi pulse will not "
        "faithfully prepare the requested product codeword"
    )
    if not allow_interacting:
        raise EmitterNotSupportedError(
            f"{message}. Increase atom spacing or pass allow_interacting=True to emit an "
            "exploratory, non-equivalent schedule"
        )
    warnings.warn(message, UserWarning, stacklevel=3)


def to_pulser(
    pulse: Mapping[str, Any],
    backend: Mapping[str, Any] | None,
    device: Any,
    *,
    allow_interacting: bool = False,
    max_interaction_ratio: float = DEFAULT_MAX_INTERACTION_RATIO,
) -> tuple[Any, float, int]:
    """Emit a Pulser ``Sequence`` from the ``pulse`` block of the IR.

    Args:
        pulse: The ``pulse`` block returned by
            :func:`~qc2pulse.compile.digital_repetition_to_analog`.
        backend: The same analog timings used to compile. Only ``max_rabi`` is consulted, as
            an optional amplitude ceiling.
        device: A Pulser ``Device`` or ``VirtualDevice``.
        allow_interacting: Emit even when the device's pair interaction is too strong for a
            global pi pulse to act as independent single-atom X gates. This is not equivalent
            to the requested repetition-code preparation and warns.
        max_interaction_ratio: Largest accepted ``max(C6 / r^6) / Omega`` ratio.

    Returns:
        ``(sequence, duration_s, n_segments)``, where ``duration_s`` is the sequence duration
        after clock-period rounding and so may exceed ``pulse["duration"]``.

    Raises:
        MissingExtraError: The ``pulser`` extra is not installed.
        EmitterNotSupportedError: The IR is malformed, the device has no global
            ground-rydberg channel, a site-selective inject is requested on a device with no
            local ground-rydberg channel, the pair interaction is too strong, or an amplitude
            exceeds the channel or backend limit.
    """
    pulse_cls, register_cls, sequence_cls, composite_cls, constant_cls, ramp_cls = _import_pulser()
    register_coords, segments, _ = read_pulse(pulse)

    if (
        isinstance(max_interaction_ratio, bool)
        or not isinstance(max_interaction_ratio, (int, float))
        or not math.isfinite(float(max_interaction_ratio))
        or float(max_interaction_ratio) <= 0.0
    ):
        raise EmitterNotSupportedError(
            f"max_interaction_ratio must be a finite positive number, got {max_interaction_ratio!r}"
        )

    needs_local = any(segment.get("site") is not None for segment in segments)

    # Resolve channels before touching the register so a global-only device fails with our
    # message instead of a downstream register-layout complaint.
    global_id = _find_channel(device, "Global")
    if global_id is None:
        raise EmitterNotSupportedError(
            f"device {getattr(device, 'name', device)!r} has no global {_RYDBERG_BASIS} "
            "channel, so the logical Rabi pi cannot be driven"
        )
    local_id = _find_channel(device, "Local") if needs_local else None
    if needs_local and local_id is None:
        sites = sorted({int(s["site"]) for s in segments if s.get("site") is not None})
        raise EmitterNotSupportedError(
            f"device {getattr(device, 'name', device)!r} has no local {_RYDBERG_BASIS} "
            f"channel ('rydberg_local'), so the site-selective inject on site {sites} cannot "
            "be emitted; use a device with local addressing or to_braket_ahs with "
            "allow_global_fallback=True"
        )

    max_rabi = None if backend is None else backend.get("max_rabi")
    if max_rabi is not None:
        peak = max(float(segment["amplitude"]) for segment in segments)
        if peak > float(max_rabi):
            raise EmitterNotSupportedError(
                f"peak amplitude {peak:.3e} rad/s exceeds backend['max_rabi'] = "
                f"{float(max_rabi):.3e} rad/s"
            )

    _check_interactions(
        register_coords,
        segments,
        device,
        allow_interacting=allow_interacting,
        ratio_limit=float(max_interaction_ratio),
    )

    register = register_cls(
        {f"q{index}": (x * 1e6, y * 1e6) for index, (x, y) in enumerate(register_coords)}
    )
    sequence = sequence_cls(register, device)
    sequence.declare_channel("global", global_id)
    if local_id is not None:
        sequence.declare_channel("local", local_id)

    channels = {"global": device.channels[global_id]}
    if local_id is not None:
        channels["local"] = device.channels[local_id]

    previous_name: str | None = None
    previous_end = 0.0

    for segment in segments:
        site = segment.get("site")
        name = "global" if site is None else "local"
        channel = channels[name]
        clock = int(getattr(channel, "clock_period", 1) or 1)
        min_duration = int(getattr(channel, "min_duration", 1) or 1)

        # Targeting has to happen before any align or delay, because Pulser refuses to
        # schedule on a local channel that has no target yet.
        if site is not None:
            sequence.target(f"q{int(site)}", "local")

        if previous_name is not None and name != previous_name and local_id is not None:
            sequence.align("global", "local")

        gap = float(segment["t_start"]) - previous_end
        if gap > 0:
            gap_ns = _to_ns(gap, clock)
            if gap_ns:
                sequence.delay(gap_ns, name)

        amplitude = float(segment["amplitude"]) * 1e-6
        if amplitude == 0.0:
            idle_ns = max(_to_ns(float(segment["duration"]), clock), _round_up(1, clock))
            sequence.delay(idle_ns, name)
            previous_name, previous_end = name, float(segment["t_end"])
            continue

        _check_amplitude(channel, amplitude, name)
        ramp_ns = _to_ns(float(segment["ramp"]), clock)
        hold_ns = _to_ns(float(segment["hold"]), clock)
        total_ns = 2 * ramp_ns + hold_ns
        if total_ns < min_duration:
            hold_ns += _round_up(min_duration - total_ns, clock)
            total_ns = 2 * ramp_ns + hold_ns

        parts = []
        if ramp_ns:
            parts.append(ramp_cls(ramp_ns, 0.0, amplitude))
        if hold_ns:
            parts.append(constant_cls(hold_ns, amplitude))
        if ramp_ns:
            parts.append(ramp_cls(ramp_ns, amplitude, 0.0))
        waveform = parts[0] if len(parts) == 1 else composite_cls(*parts)

        sequence.add(pulse_cls.ConstantDetuning(waveform, 0.0, 0.0), name)
        previous_name, previous_end = name, float(segment["t_end"])

    duration_s = float(sequence.get_duration()) * 1e-9
    return sequence, duration_s, len(segments)
