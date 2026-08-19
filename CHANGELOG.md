# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Reject malformed repetition encoders, state-changing `reset` instructions, and
  post-boundary CX gates that are not safe data-to-ancilla syndrome extraction.
- Guard global Rydberg pi pulses against strong pair interactions, with an explicit warning-only
  override for exploratory non-equivalent schedules.
- Reject zero-ramp active Braket segments instead of emitting an envelope that starts and ends
  above zero.

## [0.1.0] - 2026-08-19

### Added

- `parse_repetition`: parse an odd-`n` `[[n,1,n]]` bit-flip repetition circuit from a Qiskit
  `QuantumCircuit` or an OpenQASM 2 string, with rejection of unsupported gates, even `n`,
  `k != 1`, and misplaced or duplicated injected `X` gates.
- `digital_repetition_to_analog` / `compile_repetition`: JSON-safe analog pulse IR with an
  atom register and Rabi trapezoid segments, `min_dt` quantization, and achieved-versus-target
  pulse areas.
- `to_braket_ahs`: Braket `AnalogHamiltonianSimulation` emitter, global-only with an explicit
  opt-in fallback for site-selective injects.
- `to_pulser`: Pulser `Sequence` emitter with SI to ns/rad-per-us/um conversion and local
  channel handling.
- `syndrome_table`, `decode_counts`, `recover_logical`: `n=3` syndrome decoding and
  majority-vote logical recovery for any odd `n`.

[Unreleased]: https://github.com/PrabhavD/qc2pulse/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PrabhavD/qc2pulse/releases/tag/v0.1.0
