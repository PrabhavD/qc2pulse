"""Emit Braket ``AnalogHamiltonianSimulation`` programs from analog pulse IR.

Braket's AHS driving field is **global**: one amplitude, one detuning, and one phase shared by
every atom. A site-selective inject therefore has no faithful AHS representation, so by
default :func:`to_braket_ahs` refuses to emit it. Pass ``allow_global_fallback=True`` to drive
the whole register instead and accept the warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from ._errors import EmitterNotSupportedError, MissingExtraError
from .compile import read_pulse

__all__ = ["to_braket_ahs"]

_TIME_EPS = 1e-15


def _import_braket():
    try:
        from braket.ahs import AnalogHamiltonianSimulation, AtomArrangement, DrivingField
        from braket.timings.time_series import TimeSeries
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtraError(
            "to_braket_ahs needs the Braket SDK: pip install 'qc2pulse[braket]'"
        ) from exc
    return AnalogHamiltonianSimulation, AtomArrangement, DrivingField, TimeSeries


def _amplitude_breakpoints(
    segments: list[dict[str, Any]], duration: float
) -> list[tuple[float, float]]:
    """Piecewise-linear amplitude, starting and ending at zero as Aquila requires."""
    points: list[tuple[float, float]] = [(0.0, 0.0)]

    for segment in segments:
        t_start = float(segment["t_start"])
        t_end = float(segment["t_end"])
        ramp = float(segment["ramp"])
        amplitude = float(segment["amplitude"])

        if amplitude == 0.0:
            points.append((t_end, 0.0))
            continue

        points.append((t_start, 0.0))
        points.append((round(t_start + ramp, 18), amplitude))
        points.append((round(t_end - ramp, 18), amplitude))
        points.append((t_end, 0.0))

    points.append((duration, 0.0))

    deduped: list[tuple[float, float]] = []
    for time, value in points:
        if deduped and abs(time - deduped[-1][0]) <= _TIME_EPS:
            # Zero-hold trapezoids collapse the two flat-top breakpoints onto one instant;
            # keep the non-zero amplitude so the peak survives.
            if abs(value) > abs(deduped[-1][1]):
                deduped[-1] = (deduped[-1][0], value)
            continue
        deduped.append((time, value))
    return deduped


def to_braket_ahs(
    pulse: Mapping[str, Any],
    backend: Mapping[str, Any] | None = None,
    *,
    allow_global_fallback: bool = False,
) -> tuple[Any, float, int]:
    """Emit a Braket ``AnalogHamiltonianSimulation`` from the ``pulse`` block of the IR.

    Args:
        pulse: The ``pulse`` block returned by
            :func:`~qc2pulse.compile.digital_repetition_to_analog`.
        backend: The same analog timings used to compile. Only ``max_rabi`` is consulted, as
            an optional amplitude ceiling.
        allow_global_fallback: Emit a site-selective inject as a global drive instead of
            raising. The register-wide pulse is *not* equivalent to the requested single-site
            pulse, so this warns.

    Returns:
        ``(program, duration_s, n_segments)``.

    Raises:
        MissingExtraError: The ``braket`` extra is not installed.
        EmitterNotSupportedError: The IR is malformed, it contains a site-selective segment
            and ``allow_global_fallback`` is False, or an amplitude exceeds ``max_rabi``.
    """
    ahs_cls, arrangement_cls, driving_field_cls, time_series_cls = _import_braket()
    register_coords, segments, duration = read_pulse(pulse)

    local = [segment for segment in segments if segment.get("site") is not None]
    if local and not allow_global_fallback:
        sites = sorted({int(segment["site"]) for segment in local})
        raise EmitterNotSupportedError(
            f"Braket AHS drives every atom with one global amplitude, so the site-selective "
            f"segment(s) on site {sites} cannot be emitted faithfully. Pass "
            "allow_global_fallback=True to drive the whole register anyway, or use to_pulser "
            "with a device that has a local ground-rydberg channel"
        )
    if local:
        warnings.warn(
            f"emitting {len(local)} site-selective segment(s) as a global drive; every atom "
            "in the register receives the inject pulse",
            UserWarning,
            stacklevel=2,
        )

    max_rabi = None if backend is None else backend.get("max_rabi")
    if max_rabi is not None:
        peak = max(float(segment["amplitude"]) for segment in segments)
        if peak > float(max_rabi):
            raise EmitterNotSupportedError(
                f"peak amplitude {peak:.3e} rad/s exceeds backend['max_rabi'] = "
                f"{float(max_rabi):.3e} rad/s"
            )

    register = arrangement_cls()
    for x, y in register_coords:
        register.add([x, y])

    amplitude = time_series_cls()
    for time, value in _amplitude_breakpoints(segments, duration):
        amplitude.put(time, value)

    detuning = time_series_cls()
    phase = time_series_cls()
    for series in (detuning, phase):
        series.put(0.0, 0.0)
        series.put(duration, 0.0)

    drive = driving_field_cls(amplitude=amplitude, phase=phase, detuning=detuning)
    program = ahs_cls(register=register, hamiltonian=drive)
    return program, duration, len(segments)
