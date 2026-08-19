"""Syndrome table, destructive-readout decoding, and majority-vote recovery."""

from __future__ import annotations

import pytest

from qc2pulse._errors import DecodeError
from qc2pulse.decode import decode_counts, recover_logical, syndrome_table


def test_syndrome_table_has_exactly_four_rows():
    table = syndrome_table()

    assert table == {"00": None, "10": 0, "11": 1, "01": 2}


@pytest.mark.parametrize("n", [1, 2, 4, 5])
def test_syndrome_table_is_n3_only(n):
    with pytest.raises(NotImplementedError, match="n=3 only"):
        syndrome_table(n)


def test_clean_shots_decode_to_the_trivial_syndrome():
    result = decode_counts({"000": 80, "111": 20})

    assert result["shots"] == 100
    assert result["syndromes"] == {"00": 100}
    assert result["corrections"] == {"none": 100}
    assert result["logical"] == {"0": 80, "1": 20}
    assert result["logical_probs"]["1"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("bitstring", "syndrome", "site"),
    [("100", "10", 0), ("010", "11", 1), ("001", "01", 2)],
)
def test_single_flips_map_to_the_right_correction(bitstring, syndrome, site):
    result = decode_counts({bitstring: 10})

    assert result["syndromes"] == {syndrome: 10}
    assert result["corrections"] == {str(site): 10}
    assert result["logical"] == {"0": 10, "1": 0}


def test_bit_order_little_matches_qiskit_convention():
    big = decode_counts({"100": 5})
    little = decode_counts({"100": 5}, bit_order="little")

    assert big["corrections"] == {"0": 5}
    assert little["corrections"] == {"2": 5}


def test_counts_with_register_spaces_are_accepted():
    result = decode_counts({"0 1 0": 7})

    assert result["syndromes"] == {"11": 7}


def test_wrong_width_shots_are_rejected():
    with pytest.raises(DecodeError, match="n=3 only"):
        decode_counts({"00000": 4})


def test_mixed_width_shots_are_rejected():
    with pytest.raises(DecodeError, match="expected 3"):
        decode_counts({"000": 4, "0000": 1})


def test_non_binary_bitstrings_are_rejected():
    with pytest.raises(DecodeError, match="0s and 1s"):
        decode_counts({"0x2": 4})


def test_negative_shot_counts_are_rejected():
    with pytest.raises(DecodeError, match="non-negative"):
        decode_counts({"000": -1})


def test_empty_counts_are_rejected():
    with pytest.raises(DecodeError, match="non-empty"):
        decode_counts({})


def test_bad_bit_order_is_rejected():
    with pytest.raises(DecodeError, match="bit_order"):
        decode_counts({"000": 1}, bit_order="middle")


def test_incomplete_custom_table_is_rejected():
    with pytest.raises(DecodeError, match="missing from the decode table"):
        decode_counts({"010": 1}, {"00": None})


def test_recover_logical_majority_votes():
    probs = recover_logical({"110": 90, "000": 10})

    assert probs["1"] == pytest.approx(0.9)
    assert probs["0"] == pytest.approx(0.1)


def test_recover_logical_handles_n5():
    probs = recover_logical({"11100": 4, "00010": 6})

    assert probs["1"] == pytest.approx(0.4)


def test_recover_logical_rejects_even_width():
    with pytest.raises(DecodeError, match="odd number of data bits"):
        recover_logical({"1100": 1})
