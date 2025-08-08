import io
import json
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from rlc.simulation import (
    generate_input_function,
    simulate_series_rlc,
    compute_power_analysis,
    SupportedWaveform,
)
from rlc.frequency import (
    compute_bode,
    compute_resonance_metrics,
    OutputMeasurement,
)
from rlc.report import build_pdf_report


# -----------------------
# Page and global styling
# -----------------------
st.set_page_config(
    page_title="Voltus - RLC Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_local_css(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS not found. UI will use default styles.")


load_local_css("assets/styles.css")


# -----------------------
# Help menu
# -----------------------
def show_help_menu():
    with st.expander("💡 How to Use Voltus", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Quick Start Guide:**
            
            1. **Set Circuit Values** - Adjust R, L, C in the sidebar
            2. **Choose Input Signal** - Select waveform type and parameters
            3. **Explore Results** - Use tabs to view different analyses:
               - **Time Domain**: See voltage/current over time
               - **Frequency Domain**: View Bode plots and frequency response
               - **Filter Design**: Analyze filter characteristics
               - **Power Analysis**: Check power distribution and energy
               - **Export**: Generate PDF reports
            """)
        with col2:
            st.markdown("""
            **Pro Tips:**
            - Try different waveform types (sine, square, triangle, etc.)
            - Adjust frequency ranges for better Bode plots
            - Use the filter tab to understand circuit behavior
            - Export reports for documentation
            - Experiment with different component values
            """)


# -----------------------
# Helpers
# -----------------------
@dataclass
class RLCParams:
    resistance_ohm: float
    inductance_h: float
    capacitance_f: float


def format_scientific(value: float) -> str:
    """Format numbers with proper scientific notation"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f}×10⁶"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}×10³"
    elif abs(value) >= 1e-3:
        return f"{value:.6f}"
    elif abs(value) >= 1e-6:
        return f"{value/1e-6:.2f}×10⁻⁶"
    elif abs(value) >= 1e-9:
        return f"{value/1e-9:.2f}×10⁻⁹"
    elif abs(value) >= 1e-12:
        return f"{value/1e-12:.2f}×10⁻¹²"
    else:
        return f"{value:.2e}"


def _metric_card_columns():
    return st.columns(4)


def show_top_metrics(params: RLCParams):
    f0_hz, q_factor, bandwidth_hz = compute_resonance_metrics(
        params.resistance_ohm, params.inductance_h, params.capacitance_f
    )
    c1, c2, c3, c4 = _metric_card_columns()
    c1.metric("Resonance Frequency", f"{f0_hz:,.2f} Hz")
    c2.metric("Quality Factor", f"{q_factor:,.3f}")
    c3.metric("Bandwidth", f"{bandwidth_hz:,.2f} Hz")
    c4.metric("Damping Ratio", f"{1/(2*q_factor):,.3f}")


def _make_circuit_diagram():
    """Create a simple circuit diagram using Plotly"""
    fig = go.Figure()
    
    # Circuit elements (simplified representation)
    # Voltage source
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=1, line=dict(color="#3b82f6", width=3))
    fig.add_annotation(x=0.1, y=0.5, text="Vin", showarrow=False, font=dict(color="#3b82f6", size=12))
    
    # Resistor
    fig.add_shape(type="line", x0=0, y0=1, x1=1, y1=1, line=dict(color="#ef4444", width=2))
    fig.add_annotation(x=0.5, y=1.1, text="R", showarrow=False, font=dict(color="#ef4444", size=12))
    
    # Inductor
    fig.add_shape(type="line", x0=1, y0=1, x1=2, y1=1, line=dict(color="#06b6d4", width=2))
    fig.add_annotation(x=1.5, y=1.1, text="L", showarrow=False, font=dict(color="#06b6d4", size=12))
    
    # Capacitor
    fig.add_shape(type="line", x0=2, y0=1, x1=3, y1=1, line=dict(color="#f59e0b", width=2))
    fig.add_annotation(x=2.5, y=1.1, text="C", showarrow=False, font=dict(color="#f59e0b", size=12))
    
    # Ground
    fig.add_shape(type="line", x0=3, y0=1, x1=3, y1=0, line=dict(color="#a3a3a3", width=2))
    fig.add_annotation(x=3.1, y=0.5, text="GND", showarrow=False, font=dict(color="#a3a3a3", size=10))
    
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 3.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.2, 1.5]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _make_time_domain_fig(
    t: np.ndarray,
    i_t: np.ndarray,
    v_r: np.ndarray,
    v_l: np.ndarray,
    v_c: np.ndarray,
    v_in: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v_in, name="Vin", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=t, y=v_r, name="V_R", line=dict(color="#ef4444", width=2)))
    fig.add_trace(go.Scatter(x=t, y=v_l, name="V_L", line=dict(color="#06b6d4", width=2)))
    fig.add_trace(go.Scatter(x=t, y=v_c, name="V_C", line=dict(color="#f59e0b", width=2)))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=i_t, name="Current I", line=dict(color="#10b981", width=3)))
    fig2.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
    )
    return fig, fig2


def _make_power_fig(t: np.ndarray, power_data: Dict[str, float]) -> go.Figure:
    """Create power analysis visualization"""
    fig = go.Figure()
    
    # Power metrics as bar chart
    power_names = ["Input", "Resistor", "Inductor", "Capacitor"]
    power_values = [power_data["p_in_avg"], power_data["p_r_avg"], 
                   power_data["p_l_avg"], power_data["p_c_avg"]]
    colors = ["#3b82f6", "#ef4444", "#06b6d4", "#f59e0b"]
    
    fig.add_trace(go.Bar(
        x=power_names,
        y=power_values,
        marker_color=colors,
        text=[f"{v:.3f} W" for v in power_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        title="Average Power Distribution",
        margin=dict(l=30, r=30, t=60, b=40),
        yaxis_title="Power (W)",
        showlegend=False,
    )
    return fig


def _make_bode_fig(freq_hz: np.ndarray, mag: np.ndarray, phase_deg: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=freq_hz, y=20 * np.log10(np.maximum(mag, 1e-18)), name="Magnitude (dB)", yaxis="y1")
    )
    fig.add_trace(
        go.Scatter(x=freq_hz, y=phase_deg, name="Phase (deg)", yaxis="y2")
    )
    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=title,
        margin=dict(l=30, r=30, t=60, b=40),
        xaxis=dict(title="Frequency (Hz)", type="log"),
        yaxis=dict(title="Magnitude (dB)", side="left"),
        yaxis2=dict(title="Phase (deg)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_filter_response_fig(freq_hz: np.ndarray, mag: np.ndarray, filter_type: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_hz, y=20 * np.log10(np.maximum(mag, 1e-18)), 
                                name=f"{filter_type} Response", line=dict(color="#3b82f6", width=3)))
    
    # Add -3dB line
    fig.add_hline(y=-3, line_dash="dash", line_color="#a3a3a3", annotation_text="-3dB")
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title=f"{filter_type} Filter Response",
        margin=dict(l=30, r=30, t=60, b=40),
        xaxis=dict(title="Frequency (Hz)", type="log"),
        yaxis=dict(title="Magnitude (dB)"),
    )
    return fig


def _download_fig_png(fig: go.Figure, filename: str) -> None:
    png_bytes = fig.to_image(format="png", scale=2, engine="kaleido")
    st.download_button(
        label=f"Download {filename}.png",
        data=png_bytes,
        file_name=f"{filename}.png",
        mime="image/png",
    )


# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    # Clean logo at very top of sidebar
    st.markdown("""
    <div style="padding: 5px 0 10px 0; display: flex; align-items: center; gap: 8px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L20 8L20 16L12 22L4 16L4 8L12 2Z" fill="#3b82f6"/>
            <path d="M12 6L16 9L16 15L12 18L8 15L8 9L12 6Z" fill="white" fill-opacity="0.2"/>
        </svg>
        <span style="font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 600; color: white; letter-spacing: -0.5px;">
            Voltus
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Help menu
    show_help_menu()
    
    st.markdown("---")
    st.markdown("## Circuit Parameters")
    
    # Circuit values with better formatting
    st.markdown("### Component Values")
    col_r, col_l, col_c = st.columns(3)
    with col_r:
        resistance = st.number_input("Resistance (Ω)", min_value=0.01, max_value=1e5, value=100.0, step=10.0)
        st.caption(f"R = {format_scientific(resistance)} Ω")
    with col_l:
        inductance = st.number_input("Inductance (H)", min_value=1e-6, max_value=10.0, value=10e-3, step=1e-3)
        st.caption(f"L = {format_scientific(inductance)} H")
    with col_c:
        capacitance = st.number_input("Capacitance (F)", min_value=1e-12, max_value=1.0, value=1e-6, step=1e-6)
        st.caption(f"C = {format_scientific(capacitance)} F")

    params = RLCParams(resistance_ohm=resistance, inductance_h=inductance, capacitance_f=capacitance)

    st.markdown("---")
    st.markdown("### Input Signal")
    source_kind = st.selectbox(
        "Waveform Type",
        options=[w.value for w in SupportedWaveform],
        index=0,
    )

    source_config: Dict[str, float] = {}
    if source_kind == SupportedWaveform.DC_STEP.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["t_step"] = st.number_input("Step time (s)", value=0.0, min_value=0.0)
    elif source_kind == SupportedWaveform.SINE.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["frequency_hz"] = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.01)
        source_config["phase_deg"] = st.number_input("Phase (deg)", value=0.0)
    elif source_kind == SupportedWaveform.SQUARE.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["frequency_hz"] = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.01)
        source_config["duty"] = st.slider("Duty Cycle (%)", min_value=1, max_value=99, value=50)
    elif source_kind == SupportedWaveform.TRIANGLE.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["frequency_hz"] = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.01)
    elif source_kind == SupportedWaveform.SAWTOOTH.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["frequency_hz"] = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.01)
    elif source_kind == SupportedWaveform.PULSE.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["frequency_hz"] = st.number_input("Frequency (Hz)", value=1000.0, min_value=0.01)
        source_config["pulse_width"] = st.number_input("Pulse Width (s)", value=0.1, min_value=0.001)
    elif source_kind == SupportedWaveform.CHIRP.value:
        source_config["amplitude"] = st.number_input("Amplitude (V)", value=5.0)
        source_config["f_start_hz"] = st.number_input("Start Frequency (Hz)", value=10.0, min_value=0.001)
        source_config["f_end_hz"] = st.number_input("End Frequency (Hz)", value=100000.0, min_value=0.01)
        source_config["duration_s"] = st.number_input("Duration (s)", value=0.1, min_value=0.001)
    elif source_kind == SupportedWaveform.CUSTOM_CSV.value:
        st.caption("Upload CSV with columns: t (seconds), vin (volts)")
        uploaded = st.file_uploader("Custom Input CSV", type=["csv"]) 
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if not {"t", "vin"}.issubset(set(df.columns)):
                    st.error("CSV must have columns: t, vin")
                else:
                    source_config["csv_time"] = df["t"].to_numpy()
                    source_config["csv_vin"] = df["vin"].to_numpy()
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    st.markdown("---")
    st.markdown("### Simulation Settings")
    t_end = st.number_input("Duration (s)", value=0.05, min_value=1e-4)
    sample_rate = st.number_input("Sample Rate (Hz)", value=200_000.0, min_value=100.0)
    t_points = max(int(t_end * sample_rate), 100)


# -----------------------
# Main layout
# -----------------------
# Circuit diagram
st.plotly_chart(_make_circuit_diagram(), use_container_width=True)

show_top_metrics(params)

# Simplified tab structure
tab_time, tab_freq, tab_filter, tab_power, tab_export = st.tabs([
    "Time Domain", "Frequency Domain", "Filter Design", "Power Analysis", "Export"
])


with tab_time:
    st.markdown("### Time Domain Analysis")
    
    # Run simulation
    input_fun = generate_input_function(SupportedWaveform(source_kind), source_config)
    t, i_t, v_r, v_l, v_c, v_in = simulate_series_rlc(
        params.resistance_ohm,
        params.inductance_h,
        params.capacitance_f,
        input_fun,
        t_end,
        t_points,
    )
    
    # Display plots
    fig_v, fig_i = _make_time_domain_fig(t, i_t, v_r, v_l, v_c, v_in)
    st.plotly_chart(fig_v, use_container_width=True)
    st.plotly_chart(fig_i, use_container_width=True)
    
    # Data table
    with st.expander("Signal Data"):
        st.write(
            pd.DataFrame(
                {
                    "Time (s)": t,
                    "Vin (V)": v_in,
                    "Current (A)": i_t,
                    "V_R (V)": v_r,
                    "V_L (V)": v_l,
                    "V_C (V)": v_c,
                }
            )
        )
        st.download_button(
            "Download CSV",
            data=pd.DataFrame({
                "t": t,
                "vin": v_in,
                "i": i_t,
                "v_r": v_r,
                "v_l": v_l,
                "v_c": v_c,
            }).to_csv(index=False).encode(),
            file_name="time_domain.csv",
            mime="text/csv",
        )


with tab_freq:
    st.markdown("### Frequency Response Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        out_choice = st.selectbox(
            "Output Measurement",
            options=[m.value for m in OutputMeasurement],
            index=0,
        )
    with col2:
        n_points = st.number_input("Frequency Points", value=600, min_value=50, step=50)
    
    f_min = st.number_input("Min Frequency (Hz)", value=1.0, min_value=1e-6)
    f_max = st.number_input("Max Frequency (Hz)", value=1_000_000.0, min_value=1e-6)

    freq_hz = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))
    mag, phase = compute_bode(
        params.resistance_ohm,
        params.inductance_h,
        params.capacitance_f,
        freq_hz,
        OutputMeasurement(out_choice),
    )
    bode_fig = _make_bode_fig(freq_hz, mag, phase, title=f"Bode Plot — Output: {out_choice}")
    st.plotly_chart(bode_fig, use_container_width=True)


with tab_filter:
    st.markdown("### Filter Design & Analysis")
    
    filter_type = st.selectbox("Filter Type", ["Low-Pass", "High-Pass", "Band-Pass", "Notch"], index=0)
    
    if filter_type == "Low-Pass":
        st.info("Low-pass filter: Passes frequencies below cutoff, attenuates high frequencies")
        freq_hz_filter = np.logspace(np.log10(1.0), np.log10(1_000_000.0), 400)
        mag_filter, _ = compute_bode(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            freq_hz_filter,
            OutputMeasurement.CAPACITOR,
        )
        filter_fig = _make_filter_response_fig(freq_hz_filter, mag_filter, "Low-Pass")
        
    elif filter_type == "High-Pass":
        st.info("High-pass filter: Passes frequencies above cutoff, attenuates low frequencies")
        freq_hz_filter = np.logspace(np.log10(1.0), np.log10(1_000_000.0), 400)
        mag_filter, _ = compute_bode(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            freq_hz_filter,
            OutputMeasurement.INDUCTOR,
        )
        filter_fig = _make_filter_response_fig(freq_hz_filter, mag_filter, "High-Pass")
        
    elif filter_type == "Band-Pass":
        st.info("Band-pass filter: Passes frequencies within a specific range")
        freq_hz_filter = np.logspace(np.log10(1.0), np.log10(1_000_000.0), 400)
        mag_filter, _ = compute_bode(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            freq_hz_filter,
            OutputMeasurement.CURRENT,
        )
        filter_fig = _make_filter_response_fig(freq_hz_filter, mag_filter, "Band-Pass")
        
    else:  # Notch
        st.info("Notch filter: Attenuates frequencies within a specific range")
        freq_hz_filter = np.logspace(np.log10(1.0), np.log10(1_000_000.0), 400)
        mag_filter, _ = compute_bode(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            freq_hz_filter,
            OutputMeasurement.CURRENT,
        )
        mag_filter = 1 - mag_filter
        filter_fig = _make_filter_response_fig(freq_hz_filter, mag_filter, "Notch")
    
    st.plotly_chart(filter_fig, use_container_width=True)


with tab_power:
    st.markdown("### Power Analysis")
    
    # Run simulation for power analysis
    input_fun = generate_input_function(SupportedWaveform(source_kind), source_config)
    t, i_t, v_r, v_l, v_c, v_in = simulate_series_rlc(
        params.resistance_ohm,
        params.inductance_h,
        params.capacitance_f,
        input_fun,
        t_end,
        t_points,
    )
    
    power_data = compute_power_analysis(t, i_t, v_r, v_l, v_c, v_in)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Power Distribution")
        power_fig = _make_power_fig(t, power_data)
        st.plotly_chart(power_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Power Metrics")
        st.metric("Input Power", f"{power_data['p_in_avg']:.3f} W")
        st.metric("Resistor Power", f"{power_data['p_r_avg']:.3f} W")
        st.metric("Inductor Power", f"{power_data['p_l_avg']:.3f} W")
        st.metric("Capacitor Power", f"{power_data['p_c_avg']:.3f} W")
        st.metric("Power Factor", f"{power_data['power_factor']:.3f}")
        
        st.markdown("### Energy Storage")
        st.metric("Inductor Energy", f"{power_data['energy_l']:.6f} J")
        st.metric("Capacitor Energy", f"{power_data['energy_c']:.6f} J")


with tab_export:
    st.markdown("### Generate PDF Report")
    
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Report Title", value="Voltus Analysis Report")
        author = st.text_input("Author", value="Electrical Engineering Student")
    with col2:
        include_time = st.checkbox("Include time-domain plots", value=True)
        include_bode = st.checkbox("Include bode plot", value=True)
        include_filter = st.checkbox("Include filter analysis", value=True)
        include_power = st.checkbox("Include power analysis", value=True)

    if st.button("Generate PDF Report", type="primary"):
        # Re-generate figures to ensure they are current
        input_fun = generate_input_function(SupportedWaveform(source_kind), source_config)
        t, i_t, v_r, v_l, v_c, v_in = simulate_series_rlc(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            input_fun,
            t_end,
            t_points,
        )
        fig_v, fig_i = _make_time_domain_fig(t, i_t, v_r, v_l, v_c, v_in)

        freq_hz = np.logspace(np.log10(1.0), np.log10(1_000_000.0), 600)
        mag, phase = compute_bode(
            params.resistance_ohm,
            params.inductance_h,
            params.capacitance_f,
            freq_hz,
            OutputMeasurement.CAPACITOR,
        )
        bode_fig = _make_bode_fig(freq_hz, mag, phase, title="Bode Plot — Output: Capacitor")

        figs_to_include: Dict[str, Optional[go.Figure]] = {
            "voltages": fig_v if include_time else None,
            "current": fig_i if include_time else None,
            "bode": bode_fig if include_bode else None,
        }

        pdf_bytes = build_pdf_report(
            title=title,
            author=author,
            params=dict(R=resistance, L=inductance, C=capacitance),
            resonance_metrics=dict(
                zip(["f0_hz", "q_factor", "bandwidth_hz"], compute_resonance_metrics(resistance, inductance, capacitance))
            ),
            figures=figs_to_include,
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="voltus_analysis_report.pdf",
            mime="application/pdf",
        )


