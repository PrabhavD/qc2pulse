"""Compile a digital bit-flip repetition circuit to analog Rydberg pulses.

Three functions make up the public API:

- :func:`digital_repetition_to_analog` parses a circuit and returns analog pulse IR,
- :func:`to_braket_ahs` emits a Braket ``AnalogHamiltonianSimulation``,
- :func:`to_pulser` emits a Pulser ``Sequence``.

The supporting layers stay importable for callers who want them: :mod:`qc2pulse.parse`,
:mod:`qc2pulse.compile`, and :mod:`qc2pulse.decode`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .compile import digital_repetition_to_analog
from .emit_braket import to_braket_ahs
from .emit_pulser import to_pulser

__all__ = [
    "digital_repetition_to_analog",
    "to_braket_ahs",
    "to_pulser",
]
