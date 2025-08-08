from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


class SupportedWaveform(Enum):
    DC_STEP = "DC Step"
    SINE = "Sine"
    SQUARE = "Square"
    CHIRP = "Chirp"
    TRIANGLE = "Triangle"
    SAWTOOTH = "Sawtooth"
    PULSE = "Pulse Train"
    CUSTOM_CSV = "Custom CSV"


def generate_input_function(kind: SupportedWaveform, config: Dict) -> Callable[[float], float]:
    if kind == SupportedWaveform.DC_STEP:
        amplitude = float(config.get("amplitude", 1.0))
        t_step = float(config.get("t_step", 0.0))

        def vin(t: float) -> float:
            return amplitude if t >= t_step else 0.0

        return vin

    if kind == SupportedWaveform.SINE:
        amplitude = float(config.get("amplitude", 1.0))
        f = float(config.get("frequency_hz", 1000.0))
        phase_deg = float(config.get("phase_deg", 0.0))
        phase = np.deg2rad(phase_deg)

        def vin(t: float) -> float:
            return amplitude * np.sin(2 * np.pi * f * t + phase)

        return vin

    if kind == SupportedWaveform.SQUARE:
        amplitude = float(config.get("amplitude", 1.0))
        f = float(config.get("frequency_hz", 1000.0))
        duty = float(config.get("duty", 50.0)) / 100.0

        def vin(t: float) -> float:
            period = 1.0 / f
            phase_t = t % period
            return amplitude if phase_t < duty * period else 0.0

        return vin

    if kind == SupportedWaveform.TRIANGLE:
        amplitude = float(config.get("amplitude", 1.0))
        f = float(config.get("frequency_hz", 1000.0))

        def vin(t: float) -> float:
            period = 1.0 / f
            phase_t = t % period
            # Triangle wave: linear rise then fall
            if phase_t < period / 2:
                return amplitude * (4 * phase_t / period - 1)
            else:
                return amplitude * (3 - 4 * phase_t / period)

        return vin

    if kind == SupportedWaveform.SAWTOOTH:
        amplitude = float(config.get("amplitude", 1.0))
        f = float(config.get("frequency_hz", 1000.0))

        def vin(t: float) -> float:
            period = 1.0 / f
            phase_t = t % period
            return amplitude * (2 * phase_t / period - 1)

        return vin

    if kind == SupportedWaveform.PULSE:
        amplitude = float(config.get("amplitude", 1.0))
        f = float(config.get("frequency_hz", 1000.0))
        pulse_width = float(config.get("pulse_width", 0.1))  # in seconds

        def vin(t: float) -> float:
            period = 1.0 / f
            phase_t = t % period
            return amplitude if phase_t < pulse_width else 0.0

        return vin

    if kind == SupportedWaveform.CHIRP:
        amplitude = float(config.get("amplitude", 1.0))
        f0 = float(config.get("f_start_hz", 10.0))
        f1 = float(config.get("f_end_hz", 100_000.0))
        duration = float(config.get("duration_s", 0.1))
        k = (f1 - f0) / max(duration, 1e-9)

        def vin(t: float) -> float:
            # linear chirp frequency evolution
            inst_freq = f0 + k * np.clip(t, 0.0, duration)
            return amplitude * np.sin(2 * np.pi * inst_freq * t)

        return vin

    if kind == SupportedWaveform.CUSTOM_CSV:
        t_arr = np.asarray(config.get("csv_time", np.array([0.0, 1e-3])))
        v_arr = np.asarray(config.get("csv_vin", np.array([0.0, 0.0])))
        if t_arr.ndim != 1 or v_arr.ndim != 1 or t_arr.size != v_arr.size:
            # Fallback to zero input
            def vin(_: float) -> float:
                return 0.0

            return vin

        def vin(t: float) -> float:
            # Piecewise linear interpolation, clamp on boundaries
            return float(np.interp(t, t_arr, v_arr, left=v_arr[0], right=v_arr[-1]))

        return vin

    # default
    return lambda t: 0.0


def simulate_series_rlc(
    resistance_ohm: float,
    inductance_h: float,
    capacitance_f: float,
    vin: Callable[[float], float],
    t_end: float,
    num_points: int,
    i0: float = 0.0,
    vc0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a series RLC excited by vin(t).

    State: x = [i, v_c]
    Dynamics:
        di/dt = (vin(t) - R i - v_c) / L
        dv_c/dt = i / C

    Returns arrays for time, i(t), V_R, V_L, V_C, Vin(t)
    """
    R, L, C = resistance_ohm, inductance_h, capacitance_f

    def f(_t: float, x: np.ndarray) -> np.ndarray:
        i, v_c = x
        di_dt = (vin(_t) - R * i - v_c) / L
        dvc_dt = i / C
        return np.array([di_dt, dvc_dt])

    t_eval = np.linspace(0.0, float(t_end), int(num_points))
    sol = solve_ivp(
        f,
        t_span=(0.0, float(t_end)),
        y0=np.array([i0, vc0], dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    i_t = sol.y[0]
    v_c = sol.y[1]
    v_r = R * i_t
    # compute di/dt numerically for v_l; use central differences
    di_dt = np.gradient(i_t, t_eval, edge_order=2)
    v_l = L * di_dt
    v_in = np.array([vin(t) for t in t_eval])
    return t_eval, i_t, v_r, v_l, v_c, v_in


def compute_power_analysis(
    t: np.ndarray,
    i_t: np.ndarray,
    v_r: np.ndarray,
    v_l: np.ndarray,
    v_c: np.ndarray,
    v_in: np.ndarray,
) -> Dict[str, float]:
    """Compute power analysis metrics"""
    # Instantaneous power
    p_in = v_in * i_t
    p_r = v_r * i_t
    p_l = v_l * i_t
    p_c = v_c * i_t
    
    # Average power (RMS)
    p_in_avg = np.mean(p_in)
    p_r_avg = np.mean(p_r)
    p_l_avg = np.mean(p_l)
    p_c_avg = np.mean(p_c)
    
    # Energy stored
    energy_l = 0.5 * np.trapz(i_t**2, t)  # L * i^2 / 2
    energy_c = 0.5 * np.trapz(v_c**2, t)  # C * v^2 / 2
    
    return {
        "p_in_avg": p_in_avg,
        "p_r_avg": p_r_avg,
        "p_l_avg": p_l_avg,
        "p_c_avg": p_c_avg,
        "energy_l": energy_l,
        "energy_c": energy_c,
        "power_factor": p_in_avg / (np.sqrt(np.mean(v_in**2)) * np.sqrt(np.mean(i_t**2))) if np.mean(i_t**2) > 0 else 0
    }


