# qc2pulse

[![PyPI](https://img.shields.io/pypi/v/qc2pulse.svg)](https://pypi.org/project/qc2pulse/)
[![Python](https://img.shields.io/pypi/pyversions/qc2pulse.svg)](https://pypi.org/project/qc2pulse/)
[![Tests](https://github.com/PrabhavD/qc2pulse/actions/workflows/tests.yml/badge.svg)](https://github.com/PrabhavD/qc2pulse/actions/workflows/tests.yml)
[![License](https://img.shields.io/pypi/l/qc2pulse.svg)](LICENSE)

Compile a digital bit-flip repetition circuit to analog Rydberg pulses (Braket AHS or Pulser).

`qc2pulse` is a narrow protocol compiler, not a general transpiler. It takes an odd-`n`
`[[n,1,n]]` bit-flip repetition circuit (Qiskit `QuantumCircuit` or an OpenQASM 2 string),
rewrites it as analog pulse IR for a Rydberg QPU, and emits either a Braket
`AnalogHamiltonianSimulation` or a Pulser `Sequence`.

Anything outside that protocol is rejected rather than approximated.

## Install

```bash
pip install qc2pulse                 # parse + compile + decode
pip install "qc2pulse[braket]"       # + to_braket_ahs
pip install "qc2pulse[pulser]"       # + to_pulser
```

## Usage

```python
import math
from qiskit import QuantumCircuit
from qc2pulse import digital_repetition_to_analog

backend = {
    "rabi": math.pi / 2.5e-7,  # rad/s, a pi pulse in 250 ns
    "ramp": 5e-8,          # s
    "min_dt": 5e-8,        # s, hardware time grid
    "pulse_gap": 1e-7,     # s, idle between segments
    "spacing": 20e-6,      # m, atom pitch
    # Target-device C6 in rad/s * m^6; required by the Braket interaction guard.
    "interaction_coeff": 5.42015853e-24,
    "pi_logical": math.pi,             # global pulse area for |1_L>
    "pi_error": {0: math.pi, 1: math.pi, 2: math.pi},  # per-site inject area
}

qc = QuantumCircuit(3, 3)
qc.cx(0, 1)
qc.cx(0, 2)     # encode |0_L>
qc.x(1)         # injected bit-flip on site 1
qc.barrier()
qc.measure([0, 1, 2], [0, 1, 2])

out = digital_repetition_to_analog(qc, backend)
out["ir"]["code"]                     # '[[3,1,3]]'
out["pulse"]["segments"][0]["kind"]   # 'inject'
out["decode"]["table"]                # {'00': None, '10': 0, '11': 1, '01': 2}
```

The result is four JSON-safe keys: `pulse` (atom register plus Rabi trapezoid segments),
`ir` (what was parsed, plus the backend echo), `decode` (the `n=3` syndrome table), and
`notes` (human-readable record of every rewrite and quantization).

### Emit

```python
from qc2pulse import to_braket_ahs, to_pulser

program, duration_s, n_segments = to_braket_ahs(out["pulse"], backend,
                                                allow_global_fallback=True)

from pulser.devices import MockDevice
seq, duration_s, n_segments = to_pulser(out["pulse"], backend, MockDevice)
```

Both emitters return the same `(program, duration_s, n_segments)` shape so callers can swap
them. The IR is SI (seconds, rad/s, meters); the Pulser emitter converts to ns, rad/us, and um.

### Decode

```python
from qc2pulse import decode

table = decode.syndrome_table(3)
result = decode.decode_counts({"010": 900, "000": 100}, table)
result["logical_probs"]["0"]          # majority vote after correction
```

## Gate rules

Only a digital `X` becomes an analog pulse (a Rabi pi). Encoder `CX` is rewritten as Z-basis
codeword preparation, and syndrome `CX` is rewritten as destructive data readout, so neither
costs a pulse. `H`, `S`, `CZ`, and everything else raise `CircuitNotSupportedError`, which is
how a `[[5,1,3]]` circuit gets rejected instead of silently mis-compiled.

## Limits

- Odd `n >= 3`, `k = 1` only. `[[5,1,5]]` yes, `[[5,1,3]]` no.
- At most one injected `X`, on a data qubit, before the barrier.
- The decode table is `n=3` only. Larger `n` still compiles pulses, but `decode["table"]`
  is `None` and a note says so. `decode.recover_logical` still works for any odd `n`.
- Braket AHS amplitude is global-only, so a site-selective inject cannot be emitted
  faithfully. `to_braket_ahs` raises by default; pass `allow_global_fallback=True` to emit
  the inject globally and take a warning.
- Pulser local inject needs a local ground-rydberg channel. Global-only devices raise.
- A global pi is an independent-atom X only when pair interactions are small compared with the
  Rabi drive. Both emitters enforce `max(C6 / r^6) / Omega <= 0.01` by default. Pulser reads C6
  from the device; Braket reads `backend["interaction_coeff"]` in `rad/s * m^6`. Increase the
  spacing, or pass `allow_interacting=True` only to inspect a non-equivalent exploratory schedule.
- Braket active segments need a positive ramp so the AHS amplitude starts and ends at zero.
- An empty pulse program (`|0_L>`, no injected error) still emits a single `min_dt` idle
  segment so AHS and Pulser both receive a legal program.
- `pi_logical` and `pi_error[site]` are pulse **areas in radians** (nominally pi), not
  durations. Hold time is `area / rabi - ramp`, quantized up to a multiple of `min_dt`, and
  the achieved area is reported next to the target so quantization error is visible.

## Not in scope

Shot counts, seeds, hardware submission, result caching, plots, and emulator loops stay in
your notebook. This package has one runtime dependency (`qiskit`) and no numpy, pandas,
matplotlib, boto3, or cloud SDKs outside the optional emitter extras.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest                        # emitter tests skip without the extras
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m build && .venv/bin/twine check dist/*
```

`pre-commit install` wires up the same ruff checks, and the `Tests` workflow runs them on
Python 3.10 through 3.13 plus one extras-free install.

## License

Apache-2.0. See [LICENSE](LICENSE).
