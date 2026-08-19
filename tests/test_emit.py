"""Braket AHS and Pulser emitters, including the programs they refuse to emit."""

from __future__ import annotations

import pytest

from conftest import bitflip_circuit
from qc2pulse import digital_repetition_to_analog
from qc2pulse._errors import EmitterNotSupportedError
from qc2pulse.emit_braket import to_braket_ahs
from qc2pulse.emit_pulser import to_pulser


@pytest.fixture
def logical_pulse(backend):
    return digital_repetition_to_analog(bitflip_circuit(3, logical_one=True), backend)["pulse"]


@pytest.fixture
def inject_pulse(backend):
    return digital_repetition_to_analog(bitflip_circuit(3, inject=1), backend)["pulse"]


@pytest.fixture
def idle_pulse(backend):
    return digital_repetition_to_analog(bitflip_circuit(3), backend)["pulse"]


@pytest.fixture
def logical_and_inject_pulse(backend):
    circuit = bitflip_circuit(3, logical_one=True, inject=1)
    return digital_repetition_to_analog(circuit, backend)["pulse"]


class TestBraket:
    @pytest.fixture(autouse=True)
    def _require_braket(self):
        pytest.importorskip("braket.ahs", reason="needs the braket extra")

    def test_global_program_round_trips_register_and_duration(self, logical_pulse, backend):
        program, duration, n_segments = to_braket_ahs(logical_pulse, backend)

        assert n_segments == 1
        assert duration == pytest.approx(logical_pulse["duration"])
        assert len(program.register) == 3

    def test_amplitude_starts_and_ends_at_zero(self, logical_pulse, backend):
        program, duration, _ = to_braket_ahs(logical_pulse, backend)
        amplitude = program.hamiltonian.amplitude.time_series

        values = [float(value) for value in amplitude.values()]
        times = [float(time) for time in amplitude.times()]
        assert values[0] == 0.0
        assert values[-1] == 0.0
        assert max(values) == pytest.approx(backend["rabi"])
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(duration)
        assert times == sorted(times)

    def test_idle_program_is_still_legal(self, idle_pulse, backend):
        program, duration, n_segments = to_braket_ahs(idle_pulse, backend)
        values = [float(value) for value in program.hamiltonian.amplitude.time_series.values()]

        assert n_segments == 1
        assert duration == pytest.approx(backend["min_dt"])
        assert values == [0.0, 0.0]

    def test_site_selective_segment_is_refused_by_default(self, inject_pulse, backend):
        with pytest.raises(EmitterNotSupportedError, match="global amplitude"):
            to_braket_ahs(inject_pulse, backend)

    def test_two_segments_stay_ordered_and_separated_by_the_gap(
        self, logical_and_inject_pulse, backend
    ):
        with pytest.warns(UserWarning):
            program, duration, n_segments = to_braket_ahs(
                logical_and_inject_pulse, backend, allow_global_fallback=True
            )

        series = program.hamiltonian.amplitude.time_series
        times = [float(time) for time in series.times()]
        values = [float(value) for value in series.values()]
        assert n_segments == 2
        assert times == sorted(times)
        assert len(times) == len(set(times))
        assert duration == pytest.approx(logical_and_inject_pulse["duration"])
        # Amplitude returns to zero across the inter-segment gap.
        gap_start = logical_and_inject_pulse["segments"][0]["t_end"]
        assert values[times.index(pytest.approx(gap_start))] == 0.0

    def test_site_selective_segment_can_opt_into_a_global_drive(self, inject_pulse, backend):
        with pytest.warns(UserWarning, match="global drive"):
            program, _, n_segments = to_braket_ahs(
                inject_pulse, backend, allow_global_fallback=True
            )

        assert n_segments == 1
        assert len(program.register) == 3

    def test_amplitude_above_max_rabi_is_refused(self, logical_pulse, backend):
        backend["max_rabi"] = backend["rabi"] / 2

        with pytest.raises(EmitterNotSupportedError, match="max_rabi"):
            to_braket_ahs(logical_pulse, backend)

    def test_malformed_ir_is_refused(self, backend):
        with pytest.raises(EmitterNotSupportedError, match="missing required key"):
            to_braket_ahs({"register": [[0.0, 0.0]]}, backend)


class TestPulser:
    @pytest.fixture(autouse=True)
    def _require_pulser(self):
        pytest.importorskip("pulser", reason="needs the pulser extra")

    @pytest.fixture
    def mock_device(self):
        from pulser.devices import MockDevice

        return MockDevice

    def test_global_sequence_declares_a_rydberg_global_channel(
        self, logical_pulse, backend, mock_device
    ):
        sequence, duration, n_segments = to_pulser(logical_pulse, backend, mock_device)

        assert n_segments == 1
        assert "global" in sequence.declared_channels
        assert "local" not in sequence.declared_channels
        assert duration == pytest.approx(sequence.get_duration() * 1e-9)
        assert duration >= logical_pulse["duration"] - 1e-15

    def test_register_coordinates_are_micrometers(self, logical_pulse, backend, mock_device):
        sequence, _, _ = to_pulser(logical_pulse, backend, mock_device)
        coords = [tuple(float(c) for c in xy) for xy in sequence.register.qubits.values()]

        assert coords[0] == (0.0, 0.0)
        assert coords[1][0] == pytest.approx(backend["spacing"] * 1e6)

    def test_inject_uses_a_local_channel_targeting_the_site(
        self, inject_pulse, backend, mock_device
    ):
        sequence, duration, n_segments = to_pulser(inject_pulse, backend, mock_device)

        assert n_segments == 1
        assert "local" in sequence.declared_channels
        assert duration > 0.0
        targeted = {
            qubit
            for call in sequence._schedule["local"].slots
            for qubit in getattr(call, "targets", ())
        }
        assert "q1" in targeted

    def test_global_prep_then_local_inject_runs_in_order(
        self, logical_and_inject_pulse, backend, mock_device
    ):
        sequence, duration, n_segments = to_pulser(logical_and_inject_pulse, backend, mock_device)

        assert n_segments == 2
        assert set(sequence.declared_channels) == {"global", "local"}
        # The local inject must start no earlier than the end of the global prep.
        global_end = max(slot.tf for slot in sequence._schedule["global"].slots)
        local_pulses = [slot for slot in sequence._schedule["local"].slots if slot.type != "target"]
        inject = max(local_pulses, key=lambda slot: slot.tf)
        assert inject.ti >= global_end
        assert duration >= logical_and_inject_pulse["duration"] - 1e-15

    def test_idle_sequence_is_still_legal(self, idle_pulse, backend, mock_device):
        sequence, duration, n_segments = to_pulser(idle_pulse, backend, mock_device)

        assert n_segments == 1
        assert duration >= backend["min_dt"] - 1e-15
        assert sequence.get_duration() > 0

    def test_global_only_device_refuses_a_local_inject(self, inject_pulse, backend):
        analog_device = pytest.importorskip("pulser.devices").AnalogDevice

        with pytest.raises(EmitterNotSupportedError, match="rydberg_local"):
            to_pulser(inject_pulse, backend, analog_device)

    def test_amplitude_above_the_channel_limit_is_refused(
        self, logical_pulse, backend, mock_device
    ):
        from dataclasses import replace

        # 1 rad/us is well below the fixture's 15.7 rad/us drive.
        capped_channels = tuple(
            replace(channel, max_amp=1.0) if channel.basis == "ground-rydberg" else channel
            for channel in mock_device.channel_objects
        )
        capped = replace(mock_device, channel_objects=capped_channels)

        with pytest.raises(EmitterNotSupportedError, match="rad/us"):
            to_pulser(logical_pulse, backend, capped)

    def test_amplitude_above_the_backend_ceiling_is_refused(
        self, logical_pulse, backend, mock_device
    ):
        backend["max_rabi"] = backend["rabi"] / 2

        with pytest.raises(EmitterNotSupportedError, match="max_rabi"):
            to_pulser(logical_pulse, backend, mock_device)

    def test_device_without_channels_is_refused(self, logical_pulse, backend):
        with pytest.raises(EmitterNotSupportedError, match="channels"):
            to_pulser(logical_pulse, backend, object())
