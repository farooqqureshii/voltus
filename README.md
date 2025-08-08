# RLC Circuit Analyzer (Streamlit)

An interactive, portfolio-ready series RLC analyzer with:

- Time-domain simulation (ODE) for step/sine/square/chirp/custom inputs
- Frequency response (Bode magnitude/phase) for V_R, V_L, V_C, or current
- Parametric sweeps for R/L/C with multi-curve visualization
- Clean, modern UI (dark theme, Manrope font) — no Inter
- PDF report export with figures and metrics

## Quickstart

1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

## Custom CSV Input
Upload a CSV with columns:

- `t` (seconds)
- `vin` (volts)

The input is piecewise-linearly interpolated and used as the source.

## Notes
- The circuit modeled is a series RLC. Output for Bode plots can be selected among R, L, C or current.
- PDF export requires `kaleido` to render Plotly figures; included in requirements.

## License
MIT
