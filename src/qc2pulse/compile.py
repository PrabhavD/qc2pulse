"""Compile a parsed repetition protocol into JSON-safe analog pulse IR.

Units are SI throughout: seconds, rad/s, meters, and radians of pulse area. ``pi_logical``
and ``pi_error[site]`` are *areas* (nominally pi), so the flat-top hold of a trapezoid is
``area / rabi - ramp``. Every duration is quantized up to a multiple of ``min_dt``, and the
area that quantization actually delivers is recorded next to the target.

Only a digital ``X`` becomes a pulse. Encoder ``CX`` gates are Z-basis codeword preparation
and syndrome ``CX`` gates are destructive readout, so neither costs a segment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._errors import BackendSpecError
from .decode import SUPPORTED_DECODE_N, syndrome_table
from .parse import RepetitionSpec, parse_repetition

__all__ = [
    "IR_VERSION",
    "REQUIRED_BACKEND_KEYS",
    "compile_repetition",
    "digital_repetition_to_analog",
    "read_pulse",
]

IR_VERSION = 1

REQUIRED_BACKEND_KEYS = (
    "rabi",
    "ramp",
    "min_dt",
    "pulse_gap",
    "spacing",
    "pi_logical",
    "pi_error",
)

#: Keys that must be strictly positive; the rest only have to be non-negative.
_STRICTLY_POSITIVE = frozenset({"rabi", "min_dt", "spacing", "pi_logical"})

_QUANT_EPS = 1e-9
_ROUND_DIGITS = 18


def _round(value: float) -> float:
    """Trim floating-point dust so the IR serializes to readable JSON."""
    return float(round(value, _ROUND_DIGITS))


def _quantize_up(seconds: float, min_dt: float) -> float:
    """Round ``seconds`` up to the next multiple of the hardware time grid."""
    if seconds <= 0.0:
        return 0.0
    steps = math.ceil(seconds / min_dt - _QUANT_EPS)
    return _round(max(steps, 1) * min_dt)


def _positive_number(name: str, value: Any, *, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendSpecError(
            f"backend[{name!r}] must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise BackendSpecError(f"backend[{name!r}] must be finite, got {number}")
    if strict and number <= 0.0:
        raise BackendSpecError(f"backend[{name!r}] must be > 0, got {number}")
    if not strict and number < 0.0:
        raise BackendSpecError(f"backend[{name!r}] must be >= 0, got {number}")
    return number


def _pi_error_area(pi_error: Any, site: int, n: int) -> float:
    """Look up the calibrated inject area for ``site``.

    ``pi_error`` may be a mapping keyed by site (int or str) or a length-``n`` sequence.
    """
    if isinstance(pi_error, Mapping):
        for key in (site, str(site)):
            if key in pi_error:
                return _positive_number(f"pi_error[{site}]", pi_error[key], strict=True)
        raise BackendSpecError(
            f"backend['pi_error'] has no entry for site {site}; it only covers "
            f"{sorted(str(k) for k in pi_error)}"
        )

    if isinstance(pi_error, Sequence) and not isinstance(pi_error, (str, bytes)):
        if len(pi_error) != n:
            raise BackendSpecError(
                f"backend['pi_error'] has {len(pi_error)} entries but the code needs {n}, "
                "one per site"
            )
        return _positive_number(f"pi_error[{site}]", pi_error[site], strict=True)

    raise BackendSpecError(
        "backend['pi_error'] must be a mapping keyed by site or a sequence of length n, "
        f"got {type(pi_error).__name__}"
    )


def _validate_backend(backend: Any) -> dict[str, Any]:
    if not isinstance(backend, Mapping):
        raise BackendSpecError(
            f"backend must be a mapping of analog timings, got {type(backend).__name__}"
        )

    missing = [key for key in REQUIRED_BACKEND_KEYS if key not in backend]
    if missing:
        raise BackendSpecError(
            f"backend is missing required key(s): {', '.join(missing)}; expected "
            f"{', '.join(REQUIRED_BACKEND_KEYS)}"
        )

    checked: dict[str, Any] = {}
    for key in REQUIRED_BACKEND_KEYS:
        if key == "pi_error":
            continue
        checked[key] = _positive_number(key, backend[key], strict=key in _STRICTLY_POSITIVE)
    checked["pi_error"] = backend["pi_error"]
    return checked


def _trapezoid(
    kind: str,
    site: int | None,
    t_start: float,
    area_target: float,
    backend: Mapping[str, float],
) -> dict[str, Any]:
    """Build one Rabi trapezoid segment delivering ``area_target`` radians."""
    rabi = backend["rabi"]
    min_dt = backend["min_dt"]
    ramp = _quantize_up(backend["ramp"], min_dt)

    hold_ideal = area_target / rabi - ramp
    if hold_ideal < -_QUANT_EPS * min_dt:
        raise BackendSpecError(
            f"backend['ramp'] = {backend['ramp']:.3e} s is too long to deliver a "
            f"{area_target:.4f} rad {kind} pulse at rabi = {rabi:.3e} rad/s; the ramps alone "
            f"would supply {rabi * ramp:.4f} rad. Shorten 'ramp' or raise 'rabi'"
        )

    hold = _quantize_up(hold_ideal, min_dt) if hold_ideal > 0 else 0.0
    duration = _round(2.0 * ramp + hold)
    if duration <= 0.0:
        duration = min_dt
    # A symmetric trapezoid's two ramps contribute the area of one full ramp.
    area = _round(rabi * (ramp + hold))

    return {
        "kind": kind,
        "site": site,
        "t_start": _round(t_start),
        "t_end": _round(t_start + duration),
        "duration": duration,
        "ramp": ramp,
        "hold": hold,
        "amplitude": _round(rabi),
        "area_rad": area,
        "area_target_rad": _round(area_target),
    }


def _idle(t_start: float, min_dt: float) -> dict[str, Any]:
    return {
        "kind": "idle",
        "site": None,
        "t_start": _round(t_start),
        "t_end": _round(t_start + min_dt),
        "duration": _round(min_dt),
        "ramp": 0.0,
        "hold": _round(min_dt),
        "amplitude": 0.0,
        "area_rad": 0.0,
        "area_target_rad": 0.0,
    }


def _backend_echo(backend: Mapping[str, Any], spec: RepetitionSpec) -> dict[str, Any]:
    """JSON-safe copy of the backend, with ``pi_error`` flattened to a per-site list."""
    pi_error: list[float | None] = []
    for site in range(spec.n):
        try:
            pi_error.append(_pi_error_area(backend["pi_error"], site, spec.n))
        except BackendSpecError:
            pi_error.append(None)
    echo = {key: float(backend[key]) for key in REQUIRED_BACKEND_KEYS if key != "pi_error"}
    echo["pi_error"] = pi_error
    return echo


def compile_repetition(spec: RepetitionSpec, backend: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a :class:`~qc2pulse.parse.RepetitionSpec` into analog pulse IR.

    Args:
        spec: Output of :func:`~qc2pulse.parse.parse_repetition`.
        backend: Analog timings. Required keys are ``rabi``, ``ramp``, ``min_dt``,
            ``pulse_gap``, ``spacing``, ``pi_logical``, and ``pi_error``.

    Returns:
        A JSON-safe dict with keys ``pulse``, ``ir``, ``decode``, and ``notes``.

    Raises:
        BackendSpecError: A required key is missing, a value is unusable, ``pi_error`` has no
            entry for the injected site, or the ramp is too long for a requested pulse area.
    """
    if not isinstance(spec, RepetitionSpec):
        raise TypeError(f"spec must be a RepetitionSpec, got {type(spec).__name__}")

    checked = _validate_backend(backend)
    min_dt = checked["min_dt"]
    spacing = checked["spacing"]
    gap = _quantize_up(checked["pulse_gap"], min_dt)

    notes = [
        "encode CX rewritten as Z-basis codeword prep (no analog pulse)",
        "syndrome CX rewritten as destructive data readout (no analog pulse)",
    ]

    segments: list[dict[str, Any]] = []
    cursor = 0.0

    if spec.logical_one:
        segment = _trapezoid("logical", None, cursor, checked["pi_logical"], checked)
        segments.append(segment)
        cursor = segment["t_end"] + gap
        notes.append(
            f"|1_L> prep is a global Rabi pi of {segment['area_target_rad']:.4f} rad "
            f"over {segment['duration']:.3e} s"
        )

    if spec.inject is not None:
        area = _pi_error_area(checked["pi_error"], spec.inject, spec.n)
        segment = _trapezoid("inject", spec.inject, cursor, area, checked)
        segments.append(segment)
        cursor = segment["t_end"]
        notes.append(
            f"injected X on site {spec.inject} is a site-selective Rabi pi of "
            f"{segment['area_target_rad']:.4f} rad"
        )
    elif segments:
        # Trim the trailing gap that followed the logical prep.
        cursor = segments[-1]["t_end"]

    if not segments:
        segments.append(_idle(0.0, min_dt))
        cursor = segments[-1]["t_end"]
        notes.append(
            f"|0_L> with no injected error is an empty pulse program; emitting a single "
            f"{min_dt:.3e} s idle segment so AHS and Pulser receive a legal program"
        )

    for segment in segments:
        if segment["kind"] == "idle":
            continue
        drift = abs(segment["area_rad"] - segment["area_target_rad"])
        if drift > _QUANT_EPS:
            notes.append(
                f"{segment['kind']} segment area quantized to {segment['area_rad']:.4f} rad, "
                f"{drift:.4f} rad above the {segment['area_target_rad']:.4f} rad target "
                f"(min_dt = {min_dt:.3e} s)"
            )

    table = syndrome_table(SUPPORTED_DECODE_N) if spec.n == SUPPORTED_DECODE_N else None
    if table is None:
        notes.append(
            f"decode table is n=3 only, so decode['table'] is None for n = {spec.n}; pulses "
            "still compile and recover_logical majority-votes any odd n"
        )

    pulse = {
        "register": [[_round(site * spacing), 0.0] for site in range(spec.n)],
        "n_sites": spec.n,
        "spacing": _round(spacing),
        "duration": _round(cursor),
        "segments": segments,
        "units": {"time": "s", "amplitude": "rad/s", "position": "m", "area": "rad"},
    }

    ir = {
        "version": IR_VERSION,
        "code": spec.code,
        "n": spec.n,
        "data": list(spec.data),
        "ancilla": list(spec.ancilla),
        "inject": spec.inject,
        "logical_one": spec.logical_one,
        "backend": _backend_echo(checked, spec),
    }

    decode = {
        "n": spec.n,
        "table": table,
        "readout": "destructive_data_xor",
        "bit_order": "big",
    }

    return {"pulse": pulse, "ir": ir, "decode": decode, "notes": notes}


def read_pulse(pulse: Any) -> tuple[list[list[float]], list[dict[str, Any]], float]:
    """Validate a ``pulse`` block and return ``(register, segments, duration)``.

    Shared by both emitters so they fail the same way on hand-edited or truncated IR.

    Raises:
        EmitterNotSupportedError: ``pulse`` is not the mapping produced by
            :func:`compile_repetition`.
    """
    from ._errors import EmitterNotSupportedError

    if not isinstance(pulse, Mapping):
        raise EmitterNotSupportedError(
            "pulse must be the 'pulse' block returned by digital_repetition_to_analog, got "
            f"{type(pulse).__name__}"
        )
    missing = [key for key in ("register", "segments") if key not in pulse]
    if missing:
        raise EmitterNotSupportedError(f"pulse is missing required key(s): {', '.join(missing)}")

    register = [[float(x), float(y)] for x, y in pulse["register"]]
    segments = list(pulse["segments"])
    if not register:
        raise EmitterNotSupportedError("pulse['register'] is empty; there are no atoms to drive")
    if not segments:
        raise EmitterNotSupportedError(
            "pulse['segments'] is empty; compile_repetition always emits at least an idle "
            "segment, so this IR was truncated"
        )

    duration = float(pulse.get("duration") or max(float(s["t_end"]) for s in segments))
    return register, segments, duration


def digital_repetition_to_analog(circuit: Any, backend: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a repetition circuit and compile it to analog pulse IR.

    Args:
        circuit: A qiskit ``QuantumCircuit`` or an OpenQASM 2 string.
        backend: Analog timings, see :func:`compile_repetition`.

    Returns:
        A JSON-safe dict with keys ``pulse``, ``ir``, ``decode``, and ``notes``.
    """
    return compile_repetition(parse_repetition(circuit), backend)
