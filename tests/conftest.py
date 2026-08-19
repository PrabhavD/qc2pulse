"""Shared circuit and backend fixtures.

``bitflip_circuit`` lives here rather than in the package: it is example input, not API.
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit


def bitflip_circuit(
    n: int = 3,
    *,
    logical_one: bool = False,
    inject: int | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Build an ``[[n,1,n]]`` bit-flip repetition circuit with an optional injected error."""
    qc = QuantumCircuit(n, n)
    if logical_one:
        qc.x(0)
    for target in range(1, n):
        qc.cx(0, target)
    if inject is not None:
        qc.x(inject)
    qc.barrier()
    if measure:
        qc.measure(range(n), range(n))
    return qc


@pytest.fixture
def backend() -> dict:
    """Analog timings in SI units, loosely shaped like an Aquila-class device.

    ``rabi`` is chosen so that a pi pulse lands exactly on the ``min_dt`` grid
    (``pi / rabi == 250 ns == ramp + 4 * min_dt``), which keeps the fixture free of
    quantization overshoot.
    """
    return {
        "rabi": math.pi / 2.5e-7,
        "ramp": 5e-8,
        "min_dt": 5e-8,
        "pulse_gap": 1e-7,
        "spacing": 20e-6,
        # MockDevice C6 converted from rad/us * um^6 to rad/s * m^6.
        "interaction_coeff": 5.42015853e-24,
        "pi_logical": math.pi,
        "pi_error": {0: math.pi, 1: math.pi, 2: math.pi},
    }
