from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np


class OutputMeasurement(Enum):
    RESISTOR = "Resistor (V_R)"
    INDUCTOR = "Inductor (V_L)"
    CAPACITOR = "Capacitor (V_C)"
    CURRENT = "Current (I)"


def _total_impedance_series(R: float, L: float, C: float, omega: np.ndarray) -> np.ndarray:
    jw = 1j * omega
    z_l = jw * L
    z_c = 1.0 / (jw * C)
    z_total = R + z_l + z_c
    return z_total


def compute_transfer(
    R: float, L: float, C: float, omega: np.ndarray, output: OutputMeasurement
) -> np.ndarray:
    z_total = _total_impedance_series(R, L, C, omega)
    jw = 1j * omega
    if output == OutputMeasurement.RESISTOR:
        z_out = R
        return z_out / z_total
    if output == OutputMeasurement.INDUCTOR:
        z_out = jw * L
        return z_out / z_total
    if output == OutputMeasurement.CAPACITOR:
        z_out = 1.0 / (jw * C)
        return z_out / z_total
    if output == OutputMeasurement.CURRENT:
        # current transfer: I/Vin = 1 / Z_total
        return 1.0 / z_total
    return np.zeros_like(omega, dtype=complex)


def compute_bode(
    R: float,
    L: float,
    C: float,
    freq_hz: np.ndarray,
    output: OutputMeasurement,
) -> Tuple[np.ndarray, np.ndarray]:
    omega = 2 * np.pi * np.asarray(freq_hz)
    H = compute_transfer(R, L, C, omega, output)
    mag = np.abs(H)
    phase = np.rad2deg(np.angle(H))
    return mag, phase


def compute_resonance_metrics(R: float, L: float, C: float) -> Tuple[float, float, float]:
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    q = (1.0 / R) * np.sqrt(L / C) if R > 0 else np.inf
    bandwidth = f0 / q if q > 0 else 0.0
    return float(f0), float(q), float(bandwidth)


