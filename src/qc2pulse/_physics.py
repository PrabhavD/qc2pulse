"""Small interaction checks shared by the Rydberg emitters."""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_MAX_INTERACTION_RATIO = 0.01

# Pulser devices report C6 in rad/us * um^6. Convert to rad/s * m^6.
PULSER_C6_TO_SI = 1e-30


def max_interaction_ratio(
    register: Sequence[Sequence[float]],
    amplitude_rad_s: float,
    interaction_coeff_si: float,
) -> float:
    """Return ``max(C6 / r^6) / Omega`` across all atom pairs."""
    if len(register) < 2 or amplitude_rad_s <= 0.0:
        return 0.0

    min_distance_sq = math.inf
    for index, first in enumerate(register):
        for second in register[index + 1 :]:
            dx = float(first[0]) - float(second[0])
            dy = float(first[1]) - float(second[1])
            min_distance_sq = min(min_distance_sq, dx * dx + dy * dy)

    if min_distance_sq <= 0.0:
        return math.inf
    return float(interaction_coeff_si) / (min_distance_sq**3 * amplitude_rad_s)
