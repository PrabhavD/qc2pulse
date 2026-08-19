"""Exception types shared by the parser, compiler, and emitters.

Each error also subclasses a builtin so that callers who only catch ``ValueError`` or
``ImportError`` keep working.
"""

from __future__ import annotations

__all__ = [
    "BackendSpecError",
    "CircuitNotSupportedError",
    "DecodeError",
    "EmitterNotSupportedError",
    "MissingExtraError",
    "Qc2PulseError",
]


class Qc2PulseError(Exception):
    """Base class for every error raised by qc2pulse."""


class CircuitNotSupportedError(Qc2PulseError, ValueError):
    """The circuit is not an odd-n [[n,1,n]] bit-flip repetition protocol."""


class BackendSpecError(Qc2PulseError, ValueError):
    """The backend timing dict is missing a key or holds an unusable value."""


class EmitterNotSupportedError(Qc2PulseError, ValueError):
    """The target device or SDK cannot represent this pulse program."""


class DecodeError(Qc2PulseError, ValueError):
    """The measured counts cannot be decoded against the syndrome table."""


class MissingExtraError(Qc2PulseError, ImportError):
    """An optional emitter dependency is not installed."""
