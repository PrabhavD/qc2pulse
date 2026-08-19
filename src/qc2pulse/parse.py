"""Parse a digital bit-flip repetition circuit into a :class:`RepetitionSpec`.

The accepted protocol is an odd-``n`` ``[[n,1,n]]`` bit-flip repetition code:

1. optional ``X`` on the logical qubit (prepares ``|1_L>``),
2. a ``CX`` encoder that fans out (or chains) the logical qubit onto ``n - 1`` partners,
3. at most one injected ``X`` on a data qubit,
4. a ``barrier``, then syndrome ``CX`` gates and/or ``measure``.

Nothing else is accepted. ``H``, ``S``, and ``CZ`` raise, which is how a ``[[5,1,3]]``
circuit is rejected instead of being silently mis-compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._errors import CircuitNotSupportedError

__all__ = ["RepetitionSpec", "parse_repetition"]

#: Gates that carry meaning for the repetition protocol.
_STRUCTURAL_GATES = frozenset({"x", "cx", "barrier", "measure"})

#: Gates tolerated and dropped, since they do not change the codeword.
_IGNORED_GATES = frozenset({"id", "delay"})

ALLOWED_GATES = _STRUCTURAL_GATES | _IGNORED_GATES


@dataclass(frozen=True)
class RepetitionSpec:
    """What the parser recovered from the circuit.

    ``inject`` is a *site* index into :attr:`data` (0 .. n-1), not a raw qubit index, because
    the backend's ``pi_error`` table is keyed by site.
    """

    n: int
    data: tuple[int, ...]
    ancilla: tuple[int, ...]
    inject: int | None
    logical_one: bool
    code: str

    @property
    def logical_qubit(self) -> int:
        """Raw index of the qubit that carries the logical state before encoding."""
        return self.data[0]


def _as_circuit(circuit: Any):
    """Coerce a ``QuantumCircuit`` or an OpenQASM 2 string into a ``QuantumCircuit``."""
    if isinstance(circuit, str):
        from qiskit import QuantumCircuit

        try:
            return QuantumCircuit.from_qasm_str(circuit)
        except Exception as exc:  # noqa: BLE001 - qiskit raises several parser types
            raise CircuitNotSupportedError(f"could not parse the OpenQASM 2 string: {exc}") from exc

    if hasattr(circuit, "data") and hasattr(circuit, "num_qubits"):
        return circuit

    raise TypeError(
        f"expected a qiskit QuantumCircuit or an OpenQASM 2 string, got {type(circuit).__name__}"
    )


def _qubit_indices(circuit: Any, instruction: Any) -> list[int]:
    return [circuit.find_bit(qubit).index for qubit in instruction.qubits]


def _encoder_data_order(cx_pairs: list[tuple[int, int]]) -> tuple[int, tuple[int, ...]]:
    """Return ``(root, targets)`` for a sequentially valid encoder.

    Works for both a fanout encoder (``0->1``, ``0->2``) and a chain encoder
    (``0->1``, ``1->2``). Each CX must copy from an already encoded qubit onto a fresh
    ``|0>`` target; reconvergent edges and out-of-order chains do not prepare a repetition
    codeword and are rejected.
    """
    controls = [c for c, _ in cx_pairs]
    targets: list[int] = []
    for _, target in cx_pairs:
        if target not in targets:
            targets.append(target)

    roots = [c for c in dict.fromkeys(controls) if c not in targets]
    if len(roots) != 1:
        raise CircuitNotSupportedError(
            f"expected exactly one logical qubit (k = 1) but found {len(roots)} encoder "
            f"roots {roots}; qc2pulse only compiles [[n,1,n]] repetition codes"
        )
    root = roots[0]
    encoded = {root}
    ordered_targets: list[int] = []
    for index, (control, target) in enumerate(cx_pairs):
        if control not in encoded:
            raise CircuitNotSupportedError(
                f"encoder CX {index} uses control qubit {control} before that qubit has been "
                f"encoded; start at root {root} and order a chain from root to leaves"
            )
        if target in encoded:
            raise CircuitNotSupportedError(
                f"encoder CX {index} targets qubit {target}, which is already encoded; each "
                "data qubit must be introduced exactly once"
            )
        encoded.add(target)
        ordered_targets.append(target)

    return root, tuple(ordered_targets)


def parse_repetition(circuit: Any) -> RepetitionSpec:
    """Parse ``circuit`` as an odd-``n`` ``[[n,1,n]]`` bit-flip repetition protocol.

    Args:
        circuit: A qiskit ``QuantumCircuit`` or an OpenQASM 2 program string.

    Returns:
        The recovered :class:`RepetitionSpec`.

    Raises:
        CircuitNotSupportedError: The circuit uses a gate outside
            ``{x, cx, barrier, measure, id, delay}``, encodes an even ``n`` or
            ``n < 3``, has more than one logical qubit, injects more than one ``X``, or
            injects an ``X`` on an ancilla or after the barrier.
        TypeError: ``circuit`` is neither a ``QuantumCircuit`` nor a string.
    """
    qc = _as_circuit(circuit)

    cx_pairs: list[tuple[int, int]] = []
    syndrome_cx_pairs: list[tuple[int, int]] = []
    x_before_encoder: list[int] = []
    x_after_encoder: list[int] = []
    encoder_started = False
    past_boundary = False

    for instruction in qc.data:
        name = instruction.operation.name.lower()

        if name == "barrier":
            # Barriers before the encoder are cosmetic; the one after it is the boundary
            # between codeword preparation and syndrome extraction.
            if encoder_started:
                past_boundary = True
            continue

        if name == "measure":
            past_boundary = True
            continue

        if name in _IGNORED_GATES:
            continue

        if name == "x":
            (target,) = _qubit_indices(qc, instruction)
            if past_boundary:
                raise CircuitNotSupportedError(
                    f"X on qubit {target} appears after the barrier; an injected bit-flip "
                    "must be applied before syndrome extraction"
                )
            (x_after_encoder if encoder_started else x_before_encoder).append(target)
            continue

        if name == "cx":
            control, target = _qubit_indices(qc, instruction)
            if not past_boundary:
                encoder_started = True
                cx_pairs.append((control, target))
            else:
                syndrome_cx_pairs.append((control, target))
            continue

        raise CircuitNotSupportedError(
            f"unsupported gate {name!r} on qubits {_qubit_indices(qc, instruction)}; "
            f"qc2pulse only accepts {sorted(ALLOWED_GATES)}. A circuit using H, S, or CZ is "
            "not a bit-flip repetition protocol (for example [[5,1,3]] is out of scope)"
        )

    if not cx_pairs:
        raise CircuitNotSupportedError(
            "no encoder CX found; qc2pulse needs a [[n,1,n]] repetition encoder with odd "
            f"n >= 3, but this circuit has {qc.num_qubits} qubit(s) and no CX"
        )

    root, targets = _encoder_data_order(cx_pairs)
    data = (root, *targets)
    n = len(data)

    if n < 3:
        raise CircuitNotSupportedError(f"n must be at least 3, got n = {n}")
    if n % 2 == 0:
        raise CircuitNotSupportedError(
            f"n must be odd so that majority-vote decoding is unambiguous, got n = {n}"
        )

    data_set = set(data)
    ancilla = tuple(i for i in range(qc.num_qubits) if i not in data_set)
    ancilla_set = set(ancilla)

    # A data-controlled CX onto an ancilla leaves the data register unchanged and can safely be
    # replaced by destructive data readout. Other post-boundary CX directions mutate data or do
    # not represent syndrome extraction, so silently dropping them would change the program.
    for control, target in syndrome_cx_pairs:
        if control not in data_set or target not in ancilla_set:
            raise CircuitNotSupportedError(
                f"post-boundary CX {control}->{target} is not a syndrome extraction gate; "
                f"expected a data control in {sorted(data_set)} and an ancilla target in "
                f"{sorted(ancilla_set)}"
            )

    # An X on the root before the encoder is state preparation; an X anywhere else is an
    # injected error. Two X gates on the root cancel, hence the parity.
    logical_one = x_before_encoder.count(root) % 2 == 1
    injections = [q for q in x_before_encoder if q != root] + x_after_encoder

    for qubit in injections:
        if qubit not in data_set:
            raise CircuitNotSupportedError(
                f"X on qubit {qubit}, which is an ancilla; an injected bit-flip must act on "
                f"a data qubit {sorted(data_set)}"
            )

    if len(injections) > 1:
        raise CircuitNotSupportedError(
            f"found {len(injections)} injected X gates on qubits {injections}; qc2pulse "
            "compiles at most one injected bit-flip per program"
        )

    inject = data.index(injections[0]) if injections else None

    return RepetitionSpec(
        n=n,
        data=data,
        ancilla=ancilla,
        inject=inject,
        logical_one=logical_one,
        code=f"[[{n},1,{n}]]",
    )
