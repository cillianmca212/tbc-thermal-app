"""
app.py -- Streamlit front-end for the transient TBC thermal model.

Run locally with:   streamlit run app.py
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tbc_thermal import Layer, TBCModel

# --------------------------------------------------------------------------- #
# Optional speed-up: reuse a sparse LU factorisation instead of the pure-Python
# Thomas sweep. Same maths, same answer, but the per-step solve runs in C, which
# makes long runs (100k+ steps) practical inside a web app.
# --------------------------------------------------------------------------- #
try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu

    class FastTBCModel(TBCModel):
        @staticmethod
        def _tdma_factor(a, b, c):
            A = sps.diags([a[1:], b, c[:-1]], [-1, 0, 1], format="csc")
            return splu(A), None

        @staticmethod
        def _tdma_solve(a, lu, denom, d):
            return lu.solve(d)

    MODEL_CLS = FastTBCModel
except ImportError:  # pragma: no cover - fallback if scipy is unavailable
    MODEL_CLS = TBCModel


st.set_page_config(page_title="TBC Thermal Model", page_icon="🔥", layout="wide")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_STACK = pd.DataFrame(
    {
        "Layer name": ["YSZ top coat", "Bond coat", "Substrate"],
        "Thickness (mm)": [0.30, 0.12, 2.00],
        "k (W/m·K)": [1.2, 8.0, 18.0],
        "rho (kg/m³)": [5200.0, 7300.0, 8400.0],
        "cp (J/kg·K)": [600.0, 560.0, 500.0],
        "Cells": [24, 12, 50],
    }
)

DEFAULT_PROFILE = pd.DataFrame(
    {
        "Time (s)": [0, 200, 340, 400, 500, 550, 590, 610, 720, 870, 1000],
        "Temperature": [300, 400, 600, 700, 880, 900, 940, 940, 600, 400, 300],
    }
)


# --------------------------------------------------------------------------- #
# Cached solve -- only primitives go in, so Streamlit can hash the inputs and
# skip recomputation when nothing has changed.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def run_model(layer_rows, back_bc, h, T_back, t_end, dt, hot_t, hot_T,
              T_init, store_every):
    layers = [
        Layer(name=n, thickness=th, k=k, rho=rho, cp=cp, n_nodes=int(nn))
        for (n, th, k, rho, cp, nn) in layer_rows
    ]
    model = MODEL_CLS(layers, back_bc=back_bc, h=h, T_back=T_back)
    res = model.solve(
        t_end=t_end,
        dt=dt,
        hot_side=(np.asarray(hot_t), np.asarray(hot_T)),
        T_init=T_init,
        store_field_every=int(store_every),
    )
    meta = {
        "L": model.L,
        "N": model.N,
        "steady": model.steady_state_inner(float(np.max(hot_T))),
        "alphas": [lay.alpha for lay in layers],
        "names": [lay.name for lay in layers],
        "thicknesses": [lay.thickness for lay in layers],
        "ks": [lay.k for lay in layers],
    }
    return res, meta


# --------------------------------------------------------------------------- #
# Sidebar -- boundary conditions and solver settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Inner (cold) boundary")
    back_bc = st.selectbox(
        "Back-face condition",
        ["convective", "adiabatic", "fixed"],
        help=(
            "adiabatic = perfectly insulated back face (worst case, hottest "
            "inner wall)\n\nconvective = internally cooled\n\n"
            "fixed = back face held at a set temperature"
        ),
    )

    h = 0.0
    T_back = None
    if back_bc == "convective":
        h = st.number_input("Heat transfer coefficient h (W/m²·K)",
                            min_value=0.1, value=1000.0, step=100.0)
        T_back = st.number_input("Coolant temperature", value=18.0, step=5.0)
    elif back_bc == "fixed":
        T_back = st.number_input("Back-face temperature", value=18.0, step=5.0)

    st.header("Solver settings")
    t_end = st.number_input("End time (s)", min_value=0.001,
                            value=1000.0, step=50.0)
    dt = st.number_input("Timestep dt (s)", min_value=1e-6,
                         value=0.05, step=0.01, format="%.4f",
                         help="Backward Euler is unconditionally stable, so dt "
                              "is chosen for accuracy, not stability. Smaller "
                              "dt = more accurate but slower.")
    T_init = st.number_input("Initial wall temperature", value=300.0, step=25.0)
    n_snapshots = st.slider("Through-thickness snapshots to store", 5, 200, 40)

    n_steps = int(round(t_end / dt))
    st.caption(f"≈ {n_steps:,} timesteps")
    if n_steps > 500_000:
        st.warning("That's a lot of steps — the run may take a while. "
                   "Consider a larger timestep.")

    st.header("Units")
    temp_unit = st.radio("Temperature label", ["°C", "K"], horizontal=True,
                         help="The model is unit-agnostic; this only changes "
                              "the axis labels. Just be consistent.")

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🔥 Thermal Barrier Coating — Transient Thermal Model")
st.markdown(
    "1-D transient conduction through a multilayer wall. Set the material "
    "properties and the hot-side temperature history, then run the model to "
    "see how heat soaks through to the substrate."
)

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Layer stack")
    st.caption("Ordered **hot side first**. The last row is treated as the substrate.")
    stack_df = st.data_editor(
        DEFAULT_STACK,
        num_rows="dynamic",
        width="stretch",
        key="stack_editor",
        column_config={
            "Thickness (mm)": st.column_config.NumberColumn(format="%.4f", min_value=0.0),
            "k (W/m·K)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
            "rho (kg/m³)": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
            "cp (J/kg·K)": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
            "Cells": st.column_config.NumberColumn(
                min_value=1, step=1,
                help="Number of control volumes used to discretise this layer."),
        },
    )

with col_right:
    st.subheader("Hot-side temperature history")
    st.caption("Outer-edge temperature vs time. Linearly interpolated between points.")
    profile_df = st.data_editor(
        DEFAULT_PROFILE,
        num_rows="dynamic",
        width="stretch",
        key="profile_editor",
        column_config={
            "Time (s)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
            "Temperature": st.column_config.NumberColumn(format="%.2f"),
        },
    )

run = st.button("▶️  Run simulation", type="primary", width="stretch")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate(stack_df, profile_df):
    errors = []

    s = stack_df.dropna(subset=["Thickness (mm)", "k (W/m·K)",
                                "rho (kg/m³)", "cp (J/kg·K)", "Cells"])
    if s.empty:
        errors.append("Add at least one layer to the stack.")
    else:
        for i, row in s.iterrows():
            for col in ["Thickness (mm)", "k (W/m·K)", "rho (kg/m³)", "cp (J/kg·K)"]:
                if row[col] <= 0:
                    errors.append(f"Layer row {i + 1}: **{col}** must be greater than 0.")
            if row["Cells"] < 1:
                errors.append(f"Layer row {i + 1}: **Cells** must be at least 1.")

    p = profile_df.dropna(subset=["Time (s)", "Temperature"]).sort_values("Time (s)")
    if len(p) < 2:
        errors.append("The hot-side profile needs at least two points.")
    elif np.any(np.diff(p["Time (s)"].to_numpy()) <= 0):
        errors.append("Hot-side profile times must be strictly increasing "
                      "(no duplicate time values).")

    return s, p, errors


if run:
    stack_clean, profile_clean, errors = validate(stack_df, profile_df)
    if errors:
        for e in errors:
            st.error(e)
    else:
        layer_rows = tuple(
            (
                str(r["Layer name"]) if pd.notna(r["Layer name"]) else f"Layer {i + 1}",
                float(r["Thickness (mm)"]) * 1e-3,   # mm -> m
                float(r["k (W/m·K)"]),
                float(r["rho (kg/m³)"]),
                float(r["cp (J/kg·K)"]),
                int(r["Cells"]),
            )
            for i, (_, r) in enumerate(stack_clean.iterrows())
        )
        hot_t = tuple(float(v) for v in profile_clean["Time (s)"])
        hot_T = tuple(float(v) for v in profile_clean["Temperature"])
        store_every = max(1, int(round(t_end / dt / max(1, n_snapshots))))

        try:
            with st.spinner("Solving…"):
                res, meta = run_model(
                    layer_rows, back_bc, float(h),
                    None if T_back is None else float(T_back),
                    float(t_end), float(dt), hot_t, hot_T,
                    float(T_init), store_every,
                )
            st.session_state["result"] = (res, meta)
        except Exception as exc:  # surface model errors in the UI
            st.error(f"The model could not run: {exc}")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if "result" in st.session_state:
    res, meta = st.session_state["result"]
    u = temp_unit

    st.divider()
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak substrate face temp", f"{res.peak_substrate_outer:.1f} {u}",
              help="Hottest point the substrate itself sees "
                   "(its outer / bond-coat-facing surface).")
    m2.metric("Peak inner-wall temp", f"{res.peak_inner:.1f} {u}")
    m3.metric("Final inner-wall temp", f"{res.final_inner:.1f} {u}")
    m4.metric("Wall thickness", f"{meta['L'] * 1e3:.3f} mm",
              help=f"{meta['N']} control volumes")

    peak_hot = float(np.max(res.T_hot))
    st.caption(
        f"Peak hot-side input: {peak_hot:.1f} {u} → "
        f"temperature drop to the substrate face: "
        f"{peak_hot - res.peak_substrate_outer:.1f} {u}. "
        f"Analytic steady-state inner wall at the peak hot-side temperature: "
        f"{meta['steady']:.1f} {u}."
    )

    # ---- Plot 1: time histories ------------------------------------------- #
    fig1, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(res.times, res.T_hot, lw=2, label="Outer edge (input)")
    ax.plot(res.times, res.T_substrate_outer, lw=2,
            label="Substrate outer face")
    ax.plot(res.times, res.T_inner, lw=2, ls="--", label="Inner wall")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Temperature [{u}]")
    ax.set_title("Temperature history")
    ax.legend()
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

    # ---- Plot 2: through-thickness snapshots ------------------------------ #
    n_avail = res.T.shape[0]
    max_curves = min(12, n_avail)
    n_curves = st.slider("Profile curves to draw", 2, max(2, max_curves),
                         min(6, max_curves))
    idx = np.unique(np.linspace(0, n_avail - 1, n_curves).astype(int))

    fig2, ax2 = plt.subplots(figsize=(11, 4.8))
    x_mm = res.x * 1e3
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(idx)))
    for colour, i in zip(cmap, idx):
        ax2.plot(x_mm, res.T[i], color=colour, lw=1.8,
                 label=f"t = {res.field_times[i]:.1f} s")
    shades = plt.cm.Pastel1.colors
    for j, name in enumerate(res.layer_names):
        ax2.axvspan(res.layer_edges[j] * 1e3, res.layer_edges[j + 1] * 1e3,
                    color=shades[j % len(shades)], alpha=0.30, zorder=0)
        mid = 0.5 * (res.layer_edges[j] + res.layer_edges[j + 1]) * 1e3
        ax2.text(mid, 0.97, name, transform=ax2.get_xaxis_transform(),
                 ha="center", va="top", fontsize=8)
    ax2.set_xlabel("Depth through wall [mm]   (0 = hot outer edge)")
    ax2.set_ylabel(f"Temperature [{u}]")
    ax2.set_title("Through-thickness profiles")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    # ---- Layer summary ---------------------------------------------------- #
    with st.expander("Layer summary (diffusivity & thermal resistance)"):
        st.dataframe(
            pd.DataFrame({
                "Layer": meta["names"],
                "Thickness (mm)": [t * 1e3 for t in meta["thicknesses"]],
                "Diffusivity α (mm²/s)": [a * 1e6 for a in meta["alphas"]],
                "Resistance L/k (m²·K/W)": [
                    t / k for t, k in zip(meta["thicknesses"], meta["ks"])
                ],
            }).round(6),
            width="stretch", hide_index=True,
        )

    # ---- Downloads -------------------------------------------------------- #
    st.subheader("Download results")

    hist_df = pd.DataFrame({
        "time_s": res.times,
        "T_hot": res.T_hot,
        "T_substrate_outer": res.T_substrate_outer,
        "T_inner_wall": res.T_inner,
    })
    field_df = pd.DataFrame(
        res.T.T,
        index=pd.Index(res.x * 1e3, name="depth_mm"),
        columns=[f"t={t:.3f}s" for t in res.field_times],
    ).reset_index()

    png1, png2 = io.BytesIO(), io.BytesIO()
    fig1.savefig(png1, format="png", dpi=150, bbox_inches="tight")
    fig2.savefig(png2, format="png", dpi=150, bbox_inches="tight")

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Time history (CSV)",
                       hist_df.to_csv(index=False).encode(),
                       "tbc_time_history.csv", "text/csv",
                       width="stretch")
    d2.download_button("Depth profiles (CSV)",
                       field_df.to_csv(index=False).encode(),
                       "tbc_depth_profiles.csv", "text/csv",
                       width="stretch")
    d3.download_button("History plot (PNG)", png1.getvalue(),
                       "tbc_history.png", "image/png",
                       width="stretch")
    d4.download_button("Profile plot (PNG)", png2.getvalue(),
                       "tbc_profiles.png", "image/png",
                       width="stretch")

    plt.close(fig1)
    plt.close(fig2)
else:
    st.info("Set your parameters above and press **Run simulation**.")
