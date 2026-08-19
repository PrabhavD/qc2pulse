"""Acceptance and rejection of the [[n,1,n]] bit-flip repetition protocol."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from conftest import bitflip_circuit
from qc2pulse._errors import CircuitNotSupportedError
from qc2pulse.parse import parse_repetition


def test_accepts_repetition_3_in_logical_zero():
    spec = parse_repetition(bitflip_circuit(3))

    assert spec.n == 3
    assert spec.code == "[[3,1,3]]"
    assert spec.data == (0, 1, 2)
    assert spec.inject is None
    assert spec.logical_one is False
    assert spec.logical_qubit == 0


def test_accepts_repetition_3_in_logical_one():
    spec = parse_repetition(bitflip_circuit(3, logical_one=True))

    assert spec.logical_one is True
    assert spec.inject is None


def test_accepts_repetition_5():
    spec = parse_repetition(bitflip_circuit(5))

    assert spec.n == 5
    assert spec.code == "[[5,1,5]]"
    assert spec.data == (0, 1, 2, 3, 4)


def test_accepts_injected_x_and_reports_site_index():
    spec = parse_repetition(bitflip_circuit(3, inject=1))

    assert spec.inject == 1
    assert spec.logical_one is False


def test_accepts_openqasm2_string():
    from qiskit.qasm2 import dumps

    spec = parse_repetition(dumps(bitflip_circuit(3, inject=2)))

    assert spec.n == 3
    assert spec.inject == 2


def test_accepts_chain_encoder():
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.barrier()
    qc.measure(range(3), range(3))

    spec = parse_repetition(qc)

    assert spec.n == 3
    assert spec.data == (0, 1, 2)


def test_rejects_reconvergent_encoder_edge():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(1, 2)

    with pytest.raises(CircuitNotSupportedError, match="already encoded"):
        parse_repetition(qc)


def test_rejects_out_of_order_chain_encoder():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(1, 2)
    qc.cx(0, 1)

    with pytest.raises(CircuitNotSupportedError, match="before that qubit has been encoded"):
        parse_repetition(qc)


def test_accepts_ancilla_qubits_without_treating_them_as_data():
    qc = QuantumCircuit(5, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.measure(range(3), range(3))

    spec = parse_repetition(qc)

    assert spec.data == (0, 1, 2)
    assert spec.ancilla == (3, 4)


def test_rejects_hadamard():
    qc = bitflip_circuit(3)
    bad = QuantumCircuit(3, 3)
    bad.h(0)
    bad.compose(qc, inplace=True)

    with pytest.raises(CircuitNotSupportedError, match="unsupported gate 'h'"):
        parse_repetition(bad)


@pytest.mark.parametrize("gate", ["s", "t", "swap", "cz"])
def test_rejects_non_repetition_gates(gate):
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    if gate in ("swap", "cz"):
        getattr(qc, gate)(0, 1)
    else:
        getattr(qc, gate)(0)

    with pytest.raises(CircuitNotSupportedError, match=f"unsupported gate '{gate}'"):
        parse_repetition(qc)


def test_rejects_even_n():
    with pytest.raises(CircuitNotSupportedError, match="must be odd"):
        parse_repetition(bitflip_circuit(4))


def test_rejects_single_qubit_circuit():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    with pytest.raises(CircuitNotSupportedError, match="no encoder CX"):
        parse_repetition(qc)


def test_rejects_two_logical_qubits():
    qc = QuantumCircuit(6, 6)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.barrier()
    qc.measure(range(6), range(6))

    with pytest.raises(CircuitNotSupportedError, match=r"k = 1"):
        parse_repetition(qc)


def test_rejects_two_injected_x_gates():
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.x(1)
    qc.x(2)
    qc.barrier()
    qc.measure(range(3), range(3))

    with pytest.raises(CircuitNotSupportedError, match="2 injected X gates"):
        parse_repetition(qc)


def test_rejects_x_after_the_barrier():
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.x(1)
    qc.measure(range(3), range(3))

    with pytest.raises(CircuitNotSupportedError, match="after the barrier"):
        parse_repetition(qc)


def test_rejects_x_on_an_ancilla():
    qc = QuantumCircuit(4, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.x(3)
    qc.barrier()
    qc.measure(range(3), range(3))

    with pytest.raises(CircuitNotSupportedError, match="ancilla"):
        parse_repetition(qc)


def test_rejects_reset_instead_of_silently_dropping_it():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.reset(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    with pytest.raises(CircuitNotSupportedError, match="unsupported gate 'reset'"):
        parse_repetition(qc)


def test_rejects_post_boundary_cx_that_mutates_data():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.cx(0, 1)

    with pytest.raises(CircuitNotSupportedError, match="not a syndrome extraction gate"):
        parse_repetition(qc)


def test_rejects_post_boundary_ancilla_controlled_cx():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.cx(3, 0)

    with pytest.raises(CircuitNotSupportedError, match="not a syndrome extraction gate"):
        parse_repetition(qc)


def test_rejects_non_circuit_input():
    with pytest.raises(TypeError, match="QuantumCircuit"):
        parse_repetition(42)


def test_rejects_malformed_qasm():
    with pytest.raises(CircuitNotSupportedError, match="OpenQASM 2"):
        parse_repetition("this is not qasm")


def test_syndrome_cx_after_barrier_is_not_treated_as_encoder():
    qc = QuantumCircuit(5, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.measure([3, 4], [0, 1])

    spec = parse_repetition(qc)

    assert spec.n == 3
    assert spec.data == (0, 1, 2)
    assert spec.ancilla == (3, 4)
