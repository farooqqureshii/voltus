# Voltus
Visualize RLC Circuits

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
