"""Decode fluorescence bitstrings from a destructive repetition-code readout.

Syndrome extraction is rewritten by the compiler as a destructive data readout, so a shot is
just the ``n`` data bits. For ``n = 3`` the syndrome is the pair of neighbour parities
``(d0 ^ d1, d1 ^ d2)``, which gives the four-row table in :func:`syndrome_table`.

Bit order matters and is therefore explicit: ``"big"`` means the leftmost character of a
bitstring is site 0, which is how atom-array fluorescence is usually reported. Pass
``bit_order="little"`` for qiskit's convention, where qubit 0 is the rightmost character.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._errors import DecodeError

__all__ = ["decode_counts", "recover_logical", "syndrome_table"]

SUPPORTED_DECODE_N = 3


def syndrome_table(n: int = 3) -> dict[str, int | None]:
    """Return the ``n=3`` syndrome table mapping a syndrome to the site to flip.

    The syndrome string is ``f"{d0 ^ d1}{d1 ^ d2}"``. ``None`` means no correction.

    Raises:
        NotImplementedError: ``n`` is anything other than 3. Larger codes still compile to
            pulses, but the compiler reports ``decode["table"] is None``.
    """
    if n != SUPPORTED_DECODE_N:
        raise NotImplementedError(
            f"syndrome_table is n=3 only, got n = {n}; larger codes still compile pulses but "
            "decode['table'] is None. Use recover_logical for majority-vote decoding of any "
            "odd n"
        )
    return {"00": None, "10": 0, "11": 1, "01": 2}


def _normalize_counts(counts: Any, *, bit_order: str) -> tuple[list[tuple[list[int], int]], int]:
    """Return ``([(bits, shots), ...], width)`` with bits ordered so index 0 is site 0."""
    if bit_order not in ("big", "little"):
        raise DecodeError(f"bit_order must be 'big' or 'little', got {bit_order!r}")
    if not isinstance(counts, Mapping) or not counts:
        raise DecodeError("counts must be a non-empty mapping of bitstring to shot count")

    records: list[tuple[list[int], int]] = []
    width: int | None = None
    for raw, shots in counts.items():
        if not isinstance(raw, str):
            raise DecodeError(f"count keys must be bitstrings, got {type(raw).__name__}")
        bitstring = raw.replace(" ", "")
        if not bitstring or any(char not in "01" for char in bitstring):
            raise DecodeError(f"bitstring {raw!r} is not a string of 0s and 1s")
        if width is None:
            width = len(bitstring)
        elif len(bitstring) != width:
            raise DecodeError(
                f"bitstring {raw!r} has length {len(bitstring)}, expected {width}; all shots "
                "must come from the same register"
            )
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 0:
            raise DecodeError(f"shot count for {raw!r} must be a non-negative int, got {shots!r}")

        bits = [int(char) for char in bitstring]
        if bit_order == "little":
            bits.reverse()
        records.append((bits, shots))

    assert width is not None
    return records, width


def decode_counts(
    counts: Mapping[str, int],
    table: Mapping[str, int | None] | None = None,
    *,
    bit_order: str = "big",
) -> dict[str, Any]:
    """Decode ``n=3`` destructive-readout counts against a syndrome table.

    Args:
        counts: Bitstring to shot count, as returned by a QPU or emulator.
        table: Syndrome table; defaults to :func:`syndrome_table`.
        bit_order: ``"big"`` (leftmost char is site 0) or ``"little"`` (qiskit order).

    Returns:
        A JSON-safe dict with ``shots``, the ``syndromes`` histogram, the ``corrections``
        histogram keyed by site (``"none"`` for the trivial syndrome), corrected ``logical``
        counts, and normalized ``logical_probs``.

    Raises:
        DecodeError: A bitstring is malformed, the widths disagree, the width is not 3, or
            the table is missing a syndrome that occurred.
    """
    lookup = dict(syndrome_table(SUPPORTED_DECODE_N) if table is None else table)
    records, width = _normalize_counts(counts, bit_order=bit_order)

    if width != SUPPORTED_DECODE_N:
        raise DecodeError(
            f"decode_counts is n=3 only, got {width}-bit shots; use recover_logical for "
            "majority-vote decoding of any odd n"
        )

    syndromes: dict[str, int] = {}
    corrections: dict[str, int] = {}
    logical = {"0": 0, "1": 0}
    shots_total = 0

    for bits, shots in records:
        syndrome = f"{bits[0] ^ bits[1]}{bits[1] ^ bits[2]}"
        if syndrome not in lookup:
            raise DecodeError(
                f"syndrome {syndrome!r} is missing from the decode table {sorted(lookup)}"
            )
        site = lookup[syndrome]
        corrected = list(bits)
        if site is not None:
            corrected[site] ^= 1

        syndromes[syndrome] = syndromes.get(syndrome, 0) + shots
        key = "none" if site is None else str(site)
        corrections[key] = corrections.get(key, 0) + shots
        logical[str(corrected[0])] += shots
        shots_total += shots

    probs = (
        {bit: count / shots_total for bit, count in logical.items()}
        if shots_total
        else {"0": 0.0, "1": 0.0}
    )
    return {
        "shots": shots_total,
        "syndromes": syndromes,
        "corrections": corrections,
        "logical": logical,
        "logical_probs": probs,
        "bit_order": bit_order,
    }


def recover_logical(
    counts: Mapping[str, int],
    *,
    bit_order: str = "big",
) -> dict[str, float]:
    """Majority-vote the logical bit for any odd ``n``.

    Returns:
        Normalized probabilities ``{"0": p0, "1": p1}``.

    Raises:
        DecodeError: A bitstring is malformed, the widths disagree, or the width is even.
    """
    records, width = _normalize_counts(counts, bit_order=bit_order)
    if width % 2 == 0:
        raise DecodeError(f"majority vote needs an odd number of data bits, got {width}-bit shots")

    logical = {"0": 0, "1": 0}
    shots_total = 0
    for bits, shots in records:
        logical[str(int(sum(bits) * 2 > width))] += shots
        shots_total += shots

    if not shots_total:
        return {"0": 0.0, "1": 0.0}
    return {bit: count / shots_total for bit, count in logical.items()}
