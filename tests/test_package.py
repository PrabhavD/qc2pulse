"""Packaging invariants: version single-sourcing and the public surface."""

from __future__ import annotations

from importlib.metadata import version

import qc2pulse


def test_version_matches_installed_metadata():
    assert qc2pulse.__version__ == version("qc2pulse")


def test_public_api_is_exactly_three_functions():
    assert qc2pulse.__all__ == [
        "digital_repetition_to_analog",
        "to_braket_ahs",
        "to_pulser",
    ]
    for name in qc2pulse.__all__:
        assert callable(getattr(qc2pulse, name))


def test_importing_the_package_does_not_pull_in_the_emitter_sdks():
    import subprocess
    import sys

    probe = (
        "import sys, qc2pulse;"
        "leaked = [m for m in ('braket', 'pulser', 'qiskit') if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""
