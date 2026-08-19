"""Pulse IR shape, timing quantization, and backend validation."""

from __future__ import annotations

import json
import math

import pytest

from conftest import bitflip_circuit
from qc2pulse import digital_repetition_to_analog
from qc2pulse._errors import BackendSpecError
from qc2pulse.compile import compile_repetition
from qc2pulse.parse import parse_repetition


def _is_multiple(value: float, step: float) -> bool:
    steps = round(value / step)
    return abs(value - steps * step) <= 1e-12 * max(step, value)


def test_logical_zero_without_error_emits_one_idle_segment(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3), backend)
    segments = out["pulse"]["segments"]

    assert len(segments) == 1
    assert segments[0]["kind"] == "idle"
    assert segments[0]["amplitude"] == 0.0
    assert out["pulse"]["duration"] == pytest.approx(backend["min_dt"])
    assert any("idle segment" in note for note in out["notes"])


def test_logical_one_emits_a_single_global_segment(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)
    segments = out["pulse"]["segments"]

    assert [segment["kind"] for segment in segments] == ["logical"]
    assert segments[0]["site"] is None
    assert segments[0]["amplitude"] == pytest.approx(backend["rabi"])
    assert segments[0]["area_target_rad"] == pytest.approx(math.pi)


def test_inject_emits_a_site_selective_segment_after_the_gap(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, logical_one=True, inject=1), backend)
    first, second = out["pulse"]["segments"]

    assert [first["kind"], second["kind"]] == ["logical", "inject"]
    assert first["site"] is None
    assert second["site"] == 1
    gap = second["t_start"] - first["t_end"]
    assert gap == pytest.approx(backend["pulse_gap"])
    assert out["pulse"]["duration"] == pytest.approx(second["t_end"])


def test_inject_only_program_starts_at_zero(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, inject=2), backend)
    (segment,) = out["pulse"]["segments"]

    assert segment["kind"] == "inject"
    assert segment["site"] == 2
    assert segment["t_start"] == 0.0


def test_register_is_a_chain_with_the_backend_spacing(backend):
    out = digital_repetition_to_analog(bitflip_circuit(5), backend)
    register = out["pulse"]["register"]

    assert len(register) == 5
    assert out["pulse"]["n_sites"] == 5
    assert register[0] == [0.0, 0.0]
    for index, (x, y) in enumerate(register):
        assert x == pytest.approx(index * backend["spacing"])
        assert y == 0.0


def test_all_durations_sit_on_the_min_dt_grid(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, logical_one=True, inject=0), backend)
    min_dt = backend["min_dt"]

    for segment in out["pulse"]["segments"]:
        assert _is_multiple(segment["duration"], min_dt)
        assert _is_multiple(segment["t_start"], min_dt)
        assert _is_multiple(segment["ramp"], min_dt) or segment["ramp"] == 0.0
    assert _is_multiple(out["pulse"]["duration"], min_dt)


def test_achieved_area_is_at_least_the_target_and_within_one_grid_step(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)
    (segment,) = out["pulse"]["segments"]
    quantum = backend["rabi"] * backend["min_dt"]

    assert segment["area_rad"] >= segment["area_target_rad"] - 1e-12
    assert segment["area_rad"] - segment["area_target_rad"] < quantum
    assert segment["area_rad"] == pytest.approx(
        segment["amplitude"] * (segment["ramp"] + segment["hold"])
    )


def test_result_is_json_safe(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, inject=1), backend)

    round_tripped = json.loads(json.dumps(out))

    assert round_tripped["ir"]["code"] == "[[3,1,3]]"
    assert round_tripped["ir"]["backend"]["pi_error"] == [math.pi, math.pi, math.pi]
    assert round_tripped["decode"]["table"]["11"] == 1


def test_ir_records_what_was_parsed(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3, inject=1), backend)
    ir = out["ir"]

    assert ir["version"] == 1
    assert ir["n"] == 3
    assert ir["data"] == [0, 1, 2]
    assert ir["inject"] == 1
    assert ir["logical_one"] is False
    assert ir["backend"]["rabi"] == pytest.approx(backend["rabi"])
    assert ir["backend"]["interaction_coeff"] == pytest.approx(backend["interaction_coeff"])


def test_n5_compiles_pulses_but_has_no_decode_table(backend):
    backend["pi_error"] = dict.fromkeys(range(5), math.pi)

    out = digital_repetition_to_analog(bitflip_circuit(5, inject=3), backend)

    assert out["decode"]["table"] is None
    assert out["decode"]["n"] == 5
    assert len(out["pulse"]["register"]) == 5
    assert out["pulse"]["segments"][0]["site"] == 3
    assert any("n=3 only" in note for note in out["notes"])


def test_area_overshoot_from_a_coarse_time_grid_is_reported(backend):
    backend["min_dt"] = 1.7e-7

    out = digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)
    (segment,) = out["pulse"]["segments"]

    assert segment["area_rad"] > segment["area_target_rad"]
    assert any("area quantized" in note for note in out["notes"])


def test_notes_record_both_gate_rewrites(backend):
    out = digital_repetition_to_analog(bitflip_circuit(3), backend)
    joined = " | ".join(out["notes"])

    assert "encode CX rewritten" in joined
    assert "syndrome CX rewritten" in joined


@pytest.mark.parametrize(
    "missing", ["rabi", "ramp", "min_dt", "pulse_gap", "spacing", "pi_logical", "pi_error"]
)
def test_missing_backend_key_names_the_key(backend, missing):
    del backend[missing]

    with pytest.raises(BackendSpecError, match=missing):
        digital_repetition_to_analog(bitflip_circuit(3), backend)


def test_non_positive_rabi_is_rejected(backend):
    backend["rabi"] = 0.0

    with pytest.raises(BackendSpecError, match="'rabi'"):
        digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)


def test_non_numeric_backend_value_is_rejected(backend):
    backend["min_dt"] = "50ns"

    with pytest.raises(BackendSpecError, match="real number"):
        digital_repetition_to_analog(bitflip_circuit(3), backend)


def test_invalid_optional_interaction_coefficient_is_rejected(backend):
    backend["interaction_coeff"] = -1.0

    with pytest.raises(BackendSpecError, match="interaction_coeff"):
        digital_repetition_to_analog(bitflip_circuit(3), backend)


def test_ramp_longer_than_the_pulse_area_is_rejected(backend):
    backend["rabi"] = 1e6
    backend["ramp"] = 1e-5

    with pytest.raises(BackendSpecError, match="too long"):
        digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)


def test_missing_pi_error_entry_for_the_injected_site_is_rejected(backend):
    backend["pi_error"] = {0: math.pi}

    with pytest.raises(BackendSpecError, match=r"no entry for site 2"):
        digital_repetition_to_analog(bitflip_circuit(3, inject=2), backend)


def test_pi_error_sequence_of_the_wrong_length_is_rejected(backend):
    backend["pi_error"] = [math.pi, math.pi]

    with pytest.raises(BackendSpecError, match="one per site"):
        digital_repetition_to_analog(bitflip_circuit(3, inject=1), backend)


def test_pi_error_accepts_a_per_site_sequence(backend):
    backend["pi_error"] = [math.pi, math.pi / 2, math.pi]

    out = digital_repetition_to_analog(bitflip_circuit(3, inject=1), backend)

    assert out["pulse"]["segments"][0]["area_target_rad"] == pytest.approx(math.pi / 2)


def test_backend_must_be_a_mapping(backend):
    with pytest.raises(BackendSpecError, match="mapping"):
        digital_repetition_to_analog(bitflip_circuit(3), [1, 2, 3])


def test_compile_repetition_rejects_a_non_spec():
    with pytest.raises(TypeError, match="RepetitionSpec"):
        compile_repetition({"n": 3}, {})


def test_compile_repetition_accepts_a_spec_directly(backend):
    spec = parse_repetition(bitflip_circuit(3, logical_one=True))

    out = compile_repetition(spec, backend)

    assert out["ir"]["logical_one"] is True
    assert out["pulse"]["segments"][0]["kind"] == "logical"
