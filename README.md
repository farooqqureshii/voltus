# Voltus

A real-time RLC circuit simulator with advanced signal analysis and visualization.

## Overview

Voltus provides comprehensive analysis of series RLC circuits through time-domain simulation, frequency response analysis, and power calculations. Built for electrical engineering students and professionals who need precise circuit modeling with modern visualization.

## Features

### Core Analysis
- **Time-domain simulation** using ODE solvers
- **Frequency response** with Bode plots
- **Power analysis** with energy storage calculations
- **Filter design** for low-pass, high-pass, band-pass, and notch filters

### Input Waveforms
- Step, sinusoidal, square, triangle, sawtooth
- Chirp signals and pulse trains
- Custom CSV input support

### Visualization
- Interactive Plotly charts
- Real-time parameter updates
- Export-ready figures and reports

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd voltus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## Circuit Parameters

- **Resistance (R)**: 1Ω - 10kΩ
- **Inductance (L)**: 1μH - 1H  
- **Capacitance (C)**: 1pF - 1mF

## Custom Input

Upload a CSV file with columns:
- `t`: Time in seconds
- `vin`: Input voltage in volts

The data is interpolated and used as the circuit input.

## Dependencies

- **Streamlit**: Web interface
- **NumPy/SciPy**: Numerical computation
- **Plotly**: Interactive visualization
- **ReportLab**: PDF generation

## License

MIT License - see LICENSE file for details.
