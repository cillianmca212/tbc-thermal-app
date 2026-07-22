"""
tbc_thermal.py
==============

Transient 1-D heat-conduction model for a thermal barrier coating (TBC) stack.

You prescribe the temperature history on the OUTER (hot) edge of the coating and
the material properties of each layer, and the model returns the temperature
everywhere through the wall over time -- in particular the temperature at the
INNER wall of the substrate.

Physics
-------
The governing equation in each layer is the 1-D heat equation

        rho * cp * dT/dt = d/dx ( k * dT/dx )

solved with a control-volume (finite-volume) discretisation and an implicit
(backward-Euler) time integration. Backward Euler is unconditionally stable, so
the timestep is chosen for accuracy rather than to satisfy a stability limit --
this matters for TBCs, whose thin, low-diffusivity ceramic top coats would
otherwise force an extremely small explicit timestep.

Material interfaces are treated with series thermal resistance between cell
centres, which keeps the heat flux continuous across a property jump.

Boundary conditions
-------------------
  Outer (hot) edge  : prescribed temperature history  T_hot(t)   (Dirichlet)
  Inner (cold) edge : one of
        'adiabatic'   - perfectly insulated back face (no heat loss).
                        Gives the *highest* inner-wall temperature, i.e. the
                        conservative / worst case.
        'convective'  - cooled back face: q = h * (T_wall - T_coolant).
                        Realistic for an internally air-cooled component.
        'fixed'       - back face held at a fixed temperature.

Units
-----
SI throughout: metres, seconds, kg, W, J, K. Temperatures may be given in either
deg C or K as long as you are consistent (the equation is linear in T); for a
convective or fixed back face, give the coolant / back temperature in the same
unit you use for the hot-side profile.

Author: built for transient TBC analysis. Properties are taken as constant per
layer (as supplied); see notes at the bottom for adding temperature dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Layer definition
# --------------------------------------------------------------------------- #
@dataclass
class Layer:
    """A single material layer in the stack.

    Parameters
    ----------
    name      : label for plotting / reporting
    thickness : layer thickness            [m]
    k         : thermal conductivity       [W/(m.K)]
    rho       : density                    [kg/m^3]
    cp        : specific heat capacity     [J/(kg.K)]
    n_nodes   : number of control volumes used to discretise this layer
    """
    name: str
    thickness: float
    k: float
    rho: float
    cp: float
    n_nodes: int = 20

    def __post_init__(self) -> None:
        for field_name in ("thickness", "k", "rho", "cp"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"Layer '{self.name}': {field_name} must be > 0")
        if self.n_nodes < 1:
            raise ValueError(f"Layer '{self.name}': n_nodes must be >= 1")

    @property
    def alpha(self) -> float:
        """Thermal diffusivity k / (rho * cp)  [m^2/s]."""
        return self.k / (self.rho * self.cp)


# --------------------------------------------------------------------------- #
# Hot-side profile helper
# --------------------------------------------------------------------------- #
HotSide = Union[Callable[[float], float], Tuple[Sequence[float], Sequence[float]]]


def _make_hotside_fn(hot_side: HotSide) -> Callable[[float], float]:
    """Return a callable T(t) from either a callable or (times, temps) arrays."""
    if callable(hot_side):
        return hot_side
    times, temps = hot_side
    times = np.asarray(times, dtype=float)
    temps = np.asarray(temps, dtype=float)
    if times.ndim != 1 or temps.ndim != 1 or times.size != temps.size:
        raise ValueError("hot_side arrays must be 1-D and the same length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("hot_side time values must be strictly increasing")
    # Linear interpolation; hold the end values outside the supplied range.
    return lambda t: float(np.interp(t, times, temps))


def load_profile_csv(path: str, t_col: int = 0, T_col: int = 1,
                     skip_header: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load a (time, temperature) profile from a CSV file.

    Returns (times, temps) arrays suitable for passing as `hot_side`.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, t_col], data[:, T_col]


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    # full-resolution time histories (one value per timestep)
    times: np.ndarray            # [s]                  shape (nt,)
    T_hot: np.ndarray            # outer-edge temp      shape (nt,)
    T_inner: np.ndarray          # inner-wall temp      shape (nt,)
    T_substrate_outer: np.ndarray  # substrate outer-face temp shape (nt,)
    # spatial field snapshots (subsampled by store_field_every)
    x: np.ndarray                # cell-centre coords   shape (N,)   [m]
    field_times: np.ndarray      # snapshot times       shape (ns,)
    T: np.ndarray                # full field           shape (ns, N)
    # geometry
    layer_edges: np.ndarray      # x of layer interfaces (including 0 and L)
    layer_names: Sequence[str]

    @property
    def peak_inner(self) -> float:
        return float(np.max(self.T_inner))

    @property
    def final_inner(self) -> float:
        return float(self.T_inner[-1])

    @property
    def peak_substrate_outer(self) -> float:
        return float(np.max(self.T_substrate_outer))

    @property
    def final_substrate_outer(self) -> float:
        return float(self.T_substrate_outer[-1])

    def to_csv(self, path: str) -> None:
        """Save time, hot-side temp, inner-wall temp and substrate
        outer-face temp to a CSV file."""
        header = "time_s,T_hot,T_inner_wall,T_substrate_outer"
        out = np.column_stack([self.times, self.T_hot, self.T_inner,
                                self.T_substrate_outer])
        np.savetxt(path, out, delimiter=",", header=header, comments="")


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class TBCModel:
    """Transient 1-D conduction model through a multilayer TBC stack.

    Example
    -------
    >>> stack = [Layer("Top coat", 300e-6, 1.2, 5200, 600, n_nodes=24),
    ...          Layer("Substrate", 2.0e-3, 18.0, 8400, 500, n_nodes=40)]
    >>> model = TBCModel(stack, back_bc="convective", h=1000, T_back=18)
    >>> res = model.solve(t_end=20.0, dt=0.01,
    ...                   hot_side=([0, 1, 20], [400, 1300, 1300]),
    ...                   T_init=400)
    >>> res.peak_inner
    """

    def __init__(self,
                 layers: Sequence[Layer],
                 back_bc: str = "adiabatic",
                 h: float = 0.0,
                 T_back: Optional[float] = None):
        if not layers:
            raise ValueError("Provide at least one Layer")
        back_bc = back_bc.lower()
        if back_bc not in ("adiabatic", "convective", "fixed"):
            raise ValueError("back_bc must be 'adiabatic', 'convective' or 'fixed'")
        if back_bc == "convective" and h <= 0:
            raise ValueError("convective back_bc requires a positive heat "
                             "transfer coefficient h [W/(m^2.K)]")
        if back_bc in ("convective", "fixed") and T_back is None:
            raise ValueError(f"back_bc '{back_bc}' requires T_back")

        self.layers = list(layers)
        self.back_bc = back_bc
        self.h = float(h)
        self.T_back = T_back

        self._build_mesh()

    # ----- mesh construction ------------------------------------------------ #
    def _build_mesh(self) -> None:
        dx, k, rhocp, layer_id = [], [], [], []
        edges = [0.0]
        x0 = 0.0
        for lid, lay in enumerate(self.layers):
            cell = lay.thickness / lay.n_nodes
            for _ in range(lay.n_nodes):
                dx.append(cell)
                k.append(lay.k)
                rhocp.append(lay.rho * lay.cp)
                layer_id.append(lid)
            x0 += lay.thickness
            edges.append(x0)

        self.dx = np.asarray(dx)                  # cell widths           [m]
        self.k = np.asarray(k)                    # cell conductivity     [W/m.K]
        self.rhocp = np.asarray(rhocp)            # volumetric heat cap.  [J/m^3.K]
        self.layer_id = np.asarray(layer_id)
        self.N = self.dx.size
        self.L = float(x0)
        self.layer_edges = np.asarray(edges)
        self.layer_names = [lay.name for lay in self.layers]

        # index of the first cell of the substrate (assumed to be the last
        # layer in the stack) -- this cell's outer face is the
        # hottest point of the substrate
        substrate_lid = len(self.layers) - 1
        self.substrate_first_cell = int(np.argmax(self.layer_id == substrate_lid))

        # cell-centre coordinates
        self.x = np.cumsum(self.dx) - self.dx / 2.0

        # heat capacity per unit area of each cell  [J/(m^2.K)]
        self.cap = self.rhocp * self.dx

        # series resistance between adjacent cell centres  [m^2.K/W]
        # R[i] couples cell i and cell i+1  (size N-1)
        half = self.dx / 2.0
        self.R = half[:-1] / self.k[:-1] + half[1:] / self.k[1:]

        # half-cell resistance from the hot surface to the first cell centre
        self.R_surf = half[0] / self.k[0]

        # back-face coupling resistance (cell centre -> external reference)
        if self.back_bc == "adiabatic":
            self.R_back = np.inf
        elif self.back_bc == "convective":
            self.R_back = half[-1] / self.k[-1] + 1.0 / self.h
        else:  # fixed
            self.R_back = half[-1] / self.k[-1]

    # ----- matrix assembly (constant in time -> assembled once) ------------- #
    def _assemble(self, dt: float):
        N = self.N
        a = np.zeros(N)   # sub-diagonal
        b = np.zeros(N)   # main diagonal
        c = np.zeros(N)   # super-diagonal
        cap_dt = self.cap / dt
        invR = 1.0 / self.R              # conductance between cells (size N-1)

        # interior coupling
        b[:] = cap_dt
        b[:-1] += invR          # east conductance on every cell except the last
        b[1:] += invR           # west conductance on every cell except the first
        c[:-1] = -invR
        a[1:] = -invR

        # hot (west) Dirichlet surface coupling on cell 0
        b[0] += 1.0 / self.R_surf

        # back (east) boundary on the last cell
        if self.back_bc in ("convective", "fixed"):
            b[-1] += 1.0 / self.R_back

        return a, b, c, cap_dt

    # ----- tridiagonal (Thomas) factorisation ------------------------------- #
    @staticmethod
    def _tdma_factor(a, b, c):
        """Pre-compute the forward sweep of the Thomas algorithm.

        Because the matrix is constant in time, this is done once and reused
        every timestep; only the right-hand side changes."""
        n = b.size
        cstar = np.empty(n)
        denom = np.empty(n)
        denom[0] = b[0]
        cstar[0] = c[0] / b[0]
        for i in range(1, n):
            denom[i] = b[i] - a[i] * cstar[i - 1]
            cstar[i] = c[i] / denom[i]
        return cstar, denom

    @staticmethod
    def _tdma_solve(a, cstar, denom, d):
        """Solve using a pre-computed factorisation for right-hand side d."""
        n = d.size
        dstar = np.empty(n)
        dstar[0] = d[0] / denom[0]
        for i in range(1, n):
            dstar[i] = (d[i] - a[i] * dstar[i - 1]) / denom[i]
        x = np.empty(n)
        x[-1] = dstar[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dstar[i] - cstar[i] * x[i + 1]
        return x

    # ----- back-face surface temperature ----------------------------------- #
    def _inner_wall_temp(self, T_last: float) -> float:
        """Temperature of the physical inner wall surface from the last cell."""
        if self.back_bc == "adiabatic":
            # no flux -> no gradient in the half cell -> surface == centre
            return T_last
        if self.back_bc == "fixed":
            return float(self.T_back)
        # convective: flux leaving = (T_last - T_back) / R_back
        q = (T_last - self.T_back) / self.R_back
        half_cell_R = (self.dx[-1] / 2.0) / self.k[-1]
        return T_last - q * half_cell_R

    # ----- substrate outer-face surface temperature ------------------------- #
    def _substrate_outer_temp(self, T: np.ndarray) -> float:
        """Temperature at the outer (hot-side-facing) surface of the
        substrate -- i.e. the substrate/bond-coat interface. This is the
        hottest point the substrate itself experiences.

        If the substrate is the only layer (its first cell is cell 0), the
        outer face is the prescribed hot-side surface, so return that cell's
        surface temperature using the same construction as for the hot
        boundary on cell 0.
        """
        i = self.substrate_first_cell
        if i == 0:
            # substrate's outer face is the prescribed hot-side surface
            T_surf = T[0] + (T[0] - T[1] if self.N > 1 else 0.0) * 0.0
            # The hot surface temperature itself is the prescribed BC value;
            # callers should use T_hot directly in this case. Fall back to
            # the cell-centre temperature.
            return float(T[0])

        # flux from cell i-1 to cell i across their shared resistance
        R_link = self.R[i - 1]
        q = (T[i - 1] - T[i]) / R_link

        # half-cell resistance from cell i's centre to its west (outer) face
        half_cell_R = (self.dx[i] / 2.0) / self.k[i]
        return float(T[i] + q * half_cell_R)

    # ----- main solve loop -------------------------------------------------- #
    def solve(self,
              t_end: float,
              dt: float,
              hot_side: HotSide,
              T_init: Optional[Union[float, np.ndarray]] = None,
              store_field_every: int = 1) -> Result:
        """Integrate the temperature field from t=0 to t=t_end.

        Parameters
        ----------
        t_end             : end time [s]
        dt                : timestep [s]
        hot_side          : outer-edge temperature history. Either a callable
                            T(t) or a tuple (times, temps) that is linearly
                            interpolated.
        T_init            : initial temperature. Scalar (uniform) or array of
                            length N. Defaults to the hot-side value at t=0.
        store_field_every : store the full spatial field every Nth step (the
                            inner-wall and hot-side histories are always stored
                            at every step).
        """
        if dt <= 0 or t_end <= 0:
            raise ValueError("t_end and dt must be positive")

        Thot = _make_hotside_fn(hot_side)
        n_steps = int(round(t_end / dt))
        times = np.arange(n_steps + 1) * dt

        # initial condition
        if T_init is None:
            T = np.full(self.N, Thot(0.0))
        elif np.isscalar(T_init):
            T = np.full(self.N, float(T_init))
        else:
            T = np.asarray(T_init, dtype=float).copy()
            if T.size != self.N:
                raise ValueError("T_init array must have length N "
                                 f"(= {self.N} cells)")

        a, b, c, cap_dt = self._assemble(dt)
        cstar, denom = self._tdma_factor(a, b, c)

        inv_Rsurf = 1.0 / self.R_surf
        inv_Rback = 0.0 if not np.isfinite(self.R_back) else 1.0 / self.R_back

        # storage
        stored_idx = [0]
        field = [T.copy()]
        T_hot_hist = np.empty(n_steps + 1)
        T_inner_hist = np.empty(n_steps + 1)
        T_sub_outer_hist = np.empty(n_steps + 1)
        T_hot_hist[0] = Thot(0.0)
        T_inner_hist[0] = self._inner_wall_temp(T[-1])
        T_sub_outer_hist[0] = (T_hot_hist[0] if self.substrate_first_cell == 0
                               else self._substrate_outer_temp(T))

        for step in range(1, n_steps + 1):
            t_new = times[step]
            T_surf = Thot(t_new)

            # right-hand side = explicit storage term + boundary forcing
            d = cap_dt * T
            d[0] += inv_Rsurf * T_surf
            if self.back_bc in ("convective", "fixed"):
                d[-1] += inv_Rback * self.T_back

            T = self._tdma_solve(a, cstar, denom, d)

            T_hot_hist[step] = T_surf
            T_inner_hist[step] = self._inner_wall_temp(T[-1])
            T_sub_outer_hist[step] = (T_surf if self.substrate_first_cell == 0
                                      else self._substrate_outer_temp(T))
            if step % store_field_every == 0:
                field.append(T.copy())
                stored_idx.append(step)

        return Result(
            times=times,                 # full-resolution history times
            T_hot=T_hot_hist,
            T_inner=T_inner_hist,
            T_substrate_outer=T_sub_outer_hist,
            x=self.x,
            field_times=times[stored_idx],
            T=np.array(field),
            layer_edges=self.layer_edges,
            layer_names=self.layer_names,
        )

    # ----- steady-state analytic check -------------------------------------- #
    def steady_state_inner(self, T_hot: float) -> float:
        """Analytic steady-state inner-wall temperature for a constant hot-side
        temperature. Useful as a sanity check against the transient result run
        to equilibrium."""
        if self.back_bc == "adiabatic":
            return T_hot                      # no heat loss -> uniform = hot side
        R_total = self.R_surf + self.R.sum() + self.R_back
        q = (T_hot - self.T_back) / R_total   # steady flux
        if self.back_bc == "fixed":
            return float(self.T_back)
        # convective: inner wall surface temp = coolant + q / h
        return self.T_back + q / self.h


# --------------------------------------------------------------------------- #
# Plotting helper (optional; needs matplotlib)
# --------------------------------------------------------------------------- #
def plot_result(res: Result, save_path: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (1) hot-side input vs substrate outer-face temperature over time
    ax1.plot(res.times, res.T_hot, label="Outer edge (input)", lw=2)
    ax1.plot(res.times, res.T_substrate_outer,
             label="Substrate outer face (max substrate temp)", lw=2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Temperature history")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) spatial profile snapshots through the wall
    n_snap = min(6, res.T.shape[0])
    idx = np.linspace(0, res.T.shape[0] - 1, n_snap).astype(int)
    x_mm = res.x * 1e3
    for i in idx:
        ax2.plot(x_mm, res.T[i], label=f"t = {res.field_times[i]:.2f} s")
    # shade layers
    colors = plt.cm.Pastel1.colors
    for j in range(len(res.layer_names)):
        ax2.axvspan(res.layer_edges[j] * 1e3, res.layer_edges[j + 1] * 1e3,
                    color=colors[j % len(colors)], alpha=0.35,
                    label=f"_{res.layer_names[j]}")
        mid = 0.5 * (res.layer_edges[j] + res.layer_edges[j + 1]) * 1e3
        ax2.text(mid, ax2.get_ylim()[1], res.layer_names[j],
                 ha="center", va="top", fontsize=8, rotation=0)
    ax2.set_xlabel("Depth through wall [mm]   (0 = hot outer edge)")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Through-thickness profiles")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
