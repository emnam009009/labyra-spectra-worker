"""
Rietveld refinement (R185-7c-1 + 7c-2).

Step 1 (7c-1, baseline): scale factors only with fixed FWHM and rough background.
Step 2 (7c-2, this file): Caglioti profile params + Pseudo-Voigt + Chebyshev
       background + zero shift. Production-grade for nanocrystalline samples.

Algorithm self-implemented per Option A license audit.
References documented in algorithm-attributions.md.

@phase R185-7c-2-rietveld-profile-bg
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfileParameters:
    """Caglioti UVW + Pseudo-Voigt mixing + zero shift."""
    U: float = 0.01
    V: float = -0.005
    W: float = 0.005
    eta: float = 0.5         # Pseudo-Voigt mixing (0=Gauss, 1=Lorentz)
    zero_shift: float = 0.0  # degrees 2theta

    U_unc: float = 0.0
    V_unc: float = 0.0
    W_unc: float = 0.0
    eta_unc: float = 0.0
    zero_shift_unc: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackgroundFit:
    """Chebyshev polynomial background coefficients."""
    coefficients: list[float] = field(default_factory=lambda: [0.0] * 5)
    uncertainties: list[float] = field(default_factory=lambda: [0.0] * 5)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RietveldPhaseResult:
    """Per-phase Rietveld scale + derived mass fraction + crystallite size + R_Bragg."""
    formula: str
    scale_factor: float
    scale_uncertainty: float
    mass_fraction: float
    mass_uncertainty: float
    cell_volume_A3: float | None = None
    formula_mass: float | None = None
    formula_units_per_cell: int | None = None
    crystallite_size_nm: float | None = None
    crystallite_size_uncertainty_nm: float | None = None
    # R185-7c-3: per-phase Bragg R-factor (peak-integrated)
    r_bragg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RietveldResult:
    """Full Rietveld refinement output (R185-7c-3 augmented)."""
    converged: bool
    n_iterations: int
    r_wp: float | None = None
    r_p: float | None = None             # R185-7c-3
    r_exp: float | None = None           # R185-7c-3
    gof: float | None = None             # R185-7c-3: chi_squared_v = R_wp / R_exp
    chi_squared: float | None = None
    phases: list[RietveldPhaseResult] = field(default_factory=list)
    profile: ProfileParameters | None = None
    background: BackgroundFit | None = None
    notes: list[str] = field(default_factory=list)
    # R185-7c-3: difference plot data (capped for UI payload size)
    difference_plot: dict[str, list[float]] | None = None  # { x, y_obs, y_calc, diff }
    # R185-7c-3: per-phase contributions (already scaled), downsampled to ~200 points
    phase_contributions: dict[str, list[float]] | None = None  # { formula: y_array }

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "r_wp": self.r_wp,
            "r_p": self.r_p,
            "r_exp": self.r_exp,
            "gof": self.gof,
            "chi_squared": self.chi_squared,
            "phases": [p.to_dict() for p in self.phases],
            "profile": self.profile.to_dict() if self.profile else None,
            "background": self.background.to_dict() if self.background else None,
            "notes": self.notes,
            "difference_plot": self.difference_plot,
            "phase_contributions": self.phase_contributions,
        }


# ── Pattern simulation with refinable profile ────────────────────────────────

def _caglioti_fwhm(two_theta_deg: float | np.ndarray, U: float, V: float, W: float) -> np.ndarray:
    """
    Caglioti FWHM dependence on 2theta:
        FWHM^2 = U*tan^2(theta) + V*tan(theta) + W
    Note: 'theta' here = 2theta/2.
    """
    theta = np.radians(np.asarray(two_theta_deg) / 2.0)
    tan_th = np.tan(theta)
    fwhm_sq = U * tan_th**2 + V * tan_th + W
    fwhm_sq = np.maximum(fwhm_sq, 1e-6)  # floor to avoid negative under-fit
    return np.sqrt(fwhm_sq)


def _pseudo_voigt(x: np.ndarray, center: float, fwhm: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt profile = eta*Lorentzian + (1-eta)*Gaussian, normalized to unit area."""
    if fwhm <= 0:
        return np.zeros_like(x)
    eta = max(0.0, min(1.0, eta))
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    gauss = np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    gamma = fwhm / 2.0
    lorentz = (gamma / math.pi) / ((x - center) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss


def simulate_phase_pattern_with_profile(
    bragg_x: np.ndarray,
    bragg_intensity: np.ndarray,
    x_grid: np.ndarray,
    U: float,
    V: float,
    W: float,
    eta: float,
    zero_shift: float,
) -> np.ndarray:
    """
    Build phase pattern using refinable profile parameters.

    Each Bragg reflection at theoretical position 2θ_k convolved with
    Pseudo-Voigt of FWHM = Caglioti(2θ_k - zero_shift).
    """
    y = np.zeros_like(x_grid, dtype=float)
    for x_k, i_k in zip(bragg_x, bragg_intensity, strict=True):
        if i_k <= 0:
            continue
        shifted_center = x_k + zero_shift
        fwhm = float(_caglioti_fwhm(shifted_center, U, V, W))
        y += i_k * _pseudo_voigt(x_grid, shifted_center, fwhm, eta)
    return y


def _extract_bragg_reflections(
    structure_dict: dict[str, Any],
    two_theta_range: tuple[float, float],
    wavelength_A: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract Bragg peak positions + intensities from pymatgen."""
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator  # type: ignore
        from pymatgen.core.structure import Structure  # type: ignore
    except ImportError:
        return None

    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        logger.exception("Failed to parse structure")
        return None

    calc = XRDCalculator(wavelength=wavelength_A)
    try:
        pattern = calc.get_pattern(structure, two_theta_range=two_theta_range, scaled=False)
    except Exception:
        logger.exception("XRDCalculator failed")
        return None

    return np.asarray(pattern.x), np.asarray(pattern.y)


# ── Chebyshev background ─────────────────────────────────────────────────────

def chebyshev_background(x_norm: np.ndarray, coeffs: list[float] | np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev T_k(x_norm) polynomial sum. x_norm should be in [-1, 1]."""
    return np.polynomial.chebyshev.chebval(x_norm, np.asarray(coeffs))


def _normalize_x(x: np.ndarray) -> np.ndarray:
    """Map x to [-1, 1] for Chebyshev evaluation."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max - x_min <= 0:
        return np.zeros_like(x)
    return 2 * (x - x_min) / (x_max - x_min) - 1


# ── Hill-Howard mass fraction ────────────────────────────────────────────────

def _phase_mass_factor(structure_dict: dict[str, Any]) -> float | None:
    """Z * M / V^2 factor from Hill-Howard 1987 formula."""
    try:
        from pymatgen.core.structure import Structure
    except ImportError:
        return None
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return None
    volume = structure.volume
    if volume <= 0:
        return None
    composition = structure.composition
    _, factor = composition.get_reduced_composition_and_factor()
    z = max(1, int(factor))
    formula_mass = composition.weight / z
    return z * formula_mass / (volume ** 2)


def _phase_metadata(structure_dict: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    """Return (cell_volume_A3, formula_mass, Z)."""
    try:
        from pymatgen.core.structure import Structure
    except ImportError:
        return None, None, None
    try:
        st = Structure.from_dict(structure_dict)
        volume = float(st.volume)
        _, factor = st.composition.get_reduced_composition_and_factor()
        z = max(1, int(factor))
        formula_mass = float(st.composition.weight / z)
        return volume, formula_mass, z
    except Exception:
        return None, None, None


# ── Scherrer crystallite size from refined Caglioti ──────────────────────────

def _crystallite_size_from_profile(
    profile: ProfileParameters,
    representative_two_theta: float = 30.0,
    wavelength_A: float = 1.5406,
    K: float = 0.9,
    instrumental_fwhm_deg: float = 0.05,
) -> tuple[float, float] | None:
    """
    Estimate crystallite size via Scherrer using refined FWHM at a representative 2θ.

    Subtracts instrumental broadening (default 0.05° for typical diffractometers).
    Returns (size_nm, uncertainty_nm) or None if calculation invalid.
    """
    fwhm_total = float(_caglioti_fwhm(representative_two_theta, profile.U, profile.V, profile.W))
    fwhm_sample_sq = fwhm_total ** 2 - instrumental_fwhm_deg ** 2
    if fwhm_sample_sq <= 0:
        return None
    fwhm_sample = math.sqrt(fwhm_sample_sq)

    beta_rad = math.radians(fwhm_sample)
    theta_rad = math.radians(representative_two_theta / 2)
    cos_theta = math.cos(theta_rad)
    if cos_theta <= 0:
        return None

    size_nm = K * wavelength_A / (beta_rad * cos_theta) / 10  # /10 to convert A to nm
    # Uncertainty estimate from U,V,W stderrs (rough propagation)
    rel_unc = math.sqrt(
        (profile.U_unc / max(abs(profile.U), 1e-6)) ** 2 +
        (profile.W_unc / max(abs(profile.W), 1e-6)) ** 2
    )
    rel_unc = min(rel_unc, 0.5)  # cap at 50% relative
    unc_nm = size_nm * rel_unc
    return round(size_nm, 2), round(unc_nm, 2)


# ── Main refinement ──────────────────────────────────────────────────────────

def refine_full(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    phase_structures: dict[str, dict[str, Any]],
    wavelength_A: float = 1.5406,
    n_bg_coeffs: int = 5,
) -> RietveldResult:
    """
    Full Rietveld refinement: scale + profile + background + zero shift.

    Args:
        observed_x: 2θ array (degrees)
        observed_y: raw intensity (background NOT subtracted upfront)
        phase_structures: { formula: structure_dict }
        wavelength_A: X-ray wavelength
        n_bg_coeffs: number of Chebyshev background coefficients

    Returns RietveldResult with refined scale + profile + background.
    """
    try:
        from lmfit import Minimizer, Parameters
    except ImportError:
        return RietveldResult(converged=False, n_iterations=0,
                              notes=["lmfit unavailable"])

    if not phase_structures:
        return RietveldResult(converged=False, n_iterations=0,
                              notes=["No phases to refine"])

    two_theta_range = (float(np.min(observed_x)), float(np.max(observed_x)))

    # Extract Bragg reflections per phase ONCE (peaks positions don't change)
    phase_bragg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for formula, structure_dict in phase_structures.items():
        bragg = _extract_bragg_reflections(structure_dict, two_theta_range, wavelength_A)
        if bragg is None:
            continue
        phase_bragg[formula] = bragg

    if not phase_bragg:
        return RietveldResult(converged=False, n_iterations=0,
                              notes=["No Bragg reflections extractable from any phase"])

    x_norm = _normalize_x(observed_x)

    # ── Initialize parameters ───────────────────────────────────────────────
    params = Parameters()

    # Background coefficients (initial guess: constant ~ min(observed_y))
    initial_bg_constant = float(np.min(observed_y))
    for k in range(n_bg_coeffs):
        params.add(f"bg_{k}", value=initial_bg_constant if k == 0 else 0.0, vary=True)

    # Profile params (typical starting values for laboratory diffractometer)
    params.add("U", value=0.01, min=-0.05, max=0.5, vary=True)
    params.add("V", value=-0.005, min=-0.5, max=0.5, vary=True)
    params.add("W", value=0.005, min=1e-5, max=1.0, vary=True)  # must stay positive
    params.add("eta", value=0.5, min=0.0, max=1.0, vary=True)
    params.add("zero_shift", value=0.0, min=-0.3, max=0.3, vary=True)

    # Scale factors — initial guess from intensity scaling
    max_obs = float(np.max(observed_y))
    formulas = list(phase_bragg.keys())
    for formula in formulas:
        _, bragg_i = phase_bragg[formula]
        max_bragg = float(np.max(bragg_i)) if len(bragg_i) > 0 else 1.0
        initial_scale = max_obs / (max_bragg * len(formulas) + 1e-6)
        var = f"scale_{formula.replace('+', 'p').replace('-', 'm')}"
        params.add(var, value=initial_scale, min=0.0)

    # Weights for Poisson statistics
    weights = 1.0 / np.sqrt(np.maximum(observed_y, 1.0))

    def residuals(p):
        U = p["U"].value
        V = p["V"].value
        W = p["W"].value
        eta = p["eta"].value
        zs = p["zero_shift"].value

        # Background curve
        bg_coeffs = [p[f"bg_{k}"].value for k in range(n_bg_coeffs)]
        bg = chebyshev_background(x_norm, bg_coeffs)

        # Sum of scaled phase patterns
        model = bg.copy()
        for formula in formulas:
            var = f"scale_{formula.replace('+', 'p').replace('-', 'm')}"
            s = p[var].value
            bragg_x, bragg_i = phase_bragg[formula]
            phase_pattern = simulate_phase_pattern_with_profile(
                bragg_x=bragg_x, bragg_intensity=bragg_i,
                x_grid=observed_x,
                U=U, V=V, W=W, eta=eta, zero_shift=zs,
            )
            model += s * phase_pattern

        return (observed_y - model) * weights

    try:
        minimizer = Minimizer(residuals, params)
        out = minimizer.minimize(method="leastsq", max_nfev=500)
    except Exception:
        logger.exception("lmfit refine_full failed")
        return RietveldResult(converged=False, n_iterations=0,
                              notes=["Refinement raised exception"])

    chi2 = float(out.chisqr) if out.chisqr is not None else None

    # Compute full R-factor suite (R185-7c-3) + diff plot + phase contributions
    r_wp = None
    r_p = None
    r_exp = None
    gof = None
    diff_plot_data: dict[str, list[float]] | None = None
    phase_contrib_data: dict[str, list[float]] | None = None
    phase_r_bragg: dict[str, float] = {}

    if out.success:
        try:
            U = out.params["U"].value
            V = out.params["V"].value
            W = out.params["W"].value
            eta = out.params["eta"].value
            zs = out.params["zero_shift"].value
            bg_coeffs = [out.params[f"bg_{k}"].value for k in range(n_bg_coeffs)]
            bg = chebyshev_background(x_norm, bg_coeffs)

            phase_models: dict[str, np.ndarray] = {}
            model = bg.copy()
            for formula in formulas:
                var = f"scale_{formula.replace('+', 'p').replace('-', 'm')}"
                s = out.params[var].value
                bragg_x, bragg_i = phase_bragg[formula]
                phase_pattern = simulate_phase_pattern_with_profile(
                    bragg_x=bragg_x, bragg_intensity=bragg_i,
                    x_grid=observed_x,
                    U=U, V=V, W=W, eta=eta, zero_shift=zs,
                )
                scaled = s * phase_pattern
                phase_models[formula] = scaled
                model += scaled

            # R_wp = weighted profile
            num_wp = np.sum(weights ** 2 * (observed_y - model) ** 2)
            den_wp = np.sum(weights ** 2 * observed_y ** 2)
            if den_wp > 0:
                r_wp = round(math.sqrt(num_wp / den_wp) * 100, 2)

            # R_p = unweighted profile
            sum_obs = np.sum(observed_y)
            if sum_obs > 0:
                r_p = round(np.sum(np.abs(observed_y - model)) / sum_obs * 100, 2)

            # R_exp = sqrt((N - P) / sum(w * y_o^2))
            n_data = len(observed_y)
            n_params = len([p for p in out.params.values() if p.vary])
            if n_data > n_params and den_wp > 0:
                r_exp = round(math.sqrt((n_data - n_params) / den_wp) * 100, 2)

            # GoF = R_wp / R_exp (target < 2 ideal, < 4 acceptable)
            if r_wp is not None and r_exp is not None and r_exp > 0:
                gof = round(r_wp / r_exp, 2)

            # Per-phase R_Bragg: peak-integrated intensity comparison
            for formula, scaled_pattern in phase_models.items():
                bragg_x, bragg_i = phase_bragg[formula]
                # Sum of model intensity in narrow windows around each Bragg peak
                window = 0.3  # 2theta degrees
                ratios_num, ratios_den = 0.0, 0.0
                for bx, _bi in zip(bragg_x, bragg_i, strict=True):
                    mask = (observed_x >= bx - window) & (observed_x <= bx + window)
                    if not mask.any():
                        continue
                    i_obs = float(np.sum(observed_y[mask] - bg[mask]))
                    i_calc = float(np.sum(scaled_pattern[mask]))
                    ratios_num += abs(i_obs - i_calc)
                    ratios_den += max(i_obs, 0.0)
                if ratios_den > 0:
                    phase_r_bragg[formula] = round(ratios_num / ratios_den * 100, 2)

            # Difference plot — downsample to ~200 points for UI payload
            target_points = 200
            step = max(1, n_data // target_points)
            diff = observed_y - model
            diff_plot_data = {
                "x": [round(float(v), 4) for v in observed_x[::step]],
                "y_obs": [round(float(v), 2) for v in observed_y[::step]],
                "y_calc": [round(float(v), 2) for v in model[::step]],
                "diff": [round(float(v), 2) for v in diff[::step]],
            }

            # Phase contributions — also downsampled
            phase_contrib_data = {
                formula: [round(float(v), 2) for v in phase_models[formula][::step]]
                for formula in formulas
            }
        except Exception:
            logger.exception("R-factor computation failed (non-blocking)")

    # Build profile dataclass
    profile = ProfileParameters(
        U=round(float(out.params["U"].value), 5),
        V=round(float(out.params["V"].value), 5),
        W=round(float(out.params["W"].value), 5),
        eta=round(float(out.params["eta"].value), 3),
        zero_shift=round(float(out.params["zero_shift"].value), 4),
        U_unc=float(out.params["U"].stderr or 0.0),
        V_unc=float(out.params["V"].stderr or 0.0),
        W_unc=float(out.params["W"].stderr or 0.0),
        eta_unc=float(out.params["eta"].stderr or 0.0),
        zero_shift_unc=float(out.params["zero_shift"].stderr or 0.0),
    )

    background = BackgroundFit(
        coefficients=[round(float(out.params[f"bg_{k}"].value), 3) for k in range(n_bg_coeffs)],
        uncertainties=[float(out.params[f"bg_{k}"].stderr or 0.0) for k in range(n_bg_coeffs)],
    )

    # Compute mass fractions via Hill-Howard
    scales_with_unc: dict[str, tuple[float, float]] = {}
    for formula in formulas:
        var = f"scale_{formula.replace('+', 'p').replace('-', 'm')}"
        param = out.params[var]
        scales_with_unc[formula] = (
            float(param.value),
            float(param.stderr or 0.0),
        )

    mass_factors = {f: _phase_mass_factor(phase_structures[f]) for f in formulas}
    valid_mass = {f: mf for f, mf in mass_factors.items() if mf is not None}

    phase_results: list[RietveldPhaseResult] = []
    rietveld_diverged = False
    if len(valid_mass) == len(formulas):
        total = sum(scales_with_unc[f][0] * mass_factors[f] for f in formulas)  # type: ignore[operator]
        # B5: total<=0 means all scale factors hit the lower bound — the
        # fit diverged; phases will all read 0% which is misleading.
        rietveld_diverged = total <= 0
        for formula in formulas:
            s, ds = scales_with_unc[formula]
            mf = mass_factors[formula]
            x_mass = (s * mf) / total if total > 0 and mf is not None else 0.0
            x_unc = (ds / s) * x_mass if s > 0 else 0.0

            cell_vol, formula_mass, z = _phase_metadata(phase_structures[formula])
            size_result = _crystallite_size_from_profile(profile, wavelength_A=wavelength_A)

            phase_results.append(RietveldPhaseResult(
                formula=formula,
                scale_factor=round(s, 6),
                scale_uncertainty=round(ds, 6),
                mass_fraction=round(x_mass, 3),
                mass_uncertainty=round(x_unc, 3),
                cell_volume_A3=round(cell_vol, 2) if cell_vol else None,
                formula_mass=round(formula_mass, 2) if formula_mass else None,
                formula_units_per_cell=z,
                crystallite_size_nm=size_result[0] if size_result else None,
                crystallite_size_uncertainty_nm=size_result[1] if size_result else None,
                r_bragg=phase_r_bragg.get(formula),
            ))
    else:
        for formula in formulas:
            s, ds = scales_with_unc[formula]
            phase_results.append(RietveldPhaseResult(
                formula=formula, scale_factor=round(s, 6), scale_uncertainty=round(ds, 6),
                mass_fraction=0.0, mass_uncertainty=0.0,
            ))

    notes: list[str] = []
    if r_wp is not None:
        if r_wp > 20:
            notes.append(f"High R_wp ({r_wp}%); fit poor, check phases or instrument params")
        elif r_wp > 10:
            notes.append(f"R_wp {r_wp}% — acceptable for non-publication use")
        else:
            notes.append(f"R_wp {r_wp}% — good fit quality")
    if profile.W < 1e-4:
        notes.append("W parameter at lower bound; FWHM may be unrealistic")
    if rietveld_diverged:
        notes.append(
            "All phase scale factors near zero — Rietveld fit diverged; "
            "mass fractions are not meaningful. Check initial structures + 2theta range."
        )

    # GoF interpretation note
    if gof is not None:
        if gof > 4:
            notes.append(f"GoF = {gof}: poor fit, check model")
        elif gof > 2:
            notes.append(f"GoF = {gof}: acceptable but room to improve")
        else:
            notes.append(f"GoF = {gof}: good statistical fit")

    return RietveldResult(
        converged=bool(out.success),
        n_iterations=int(out.nfev) if out.nfev else 0,
        r_wp=r_wp,
        r_p=r_p,
        r_exp=r_exp,
        gof=gof,
        chi_squared=round(chi2, 4) if chi2 is not None else None,
        phases=phase_results,
        profile=profile,
        background=background,
        notes=notes,
        difference_plot=diff_plot_data,
        phase_contributions=phase_contrib_data,
    )


# ── High-level dispatcher (replaces R185-7c-1 version) ───────────────────────

def attempt_rietveld_refinement(
    spectrum_curve: dict[str, list[float]] | None,
    components: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    wavelength_A: float = 1.5406,
) -> RietveldResult | None:
    """Dispatcher: collect data, run full refinement (7c-2)."""
    if not spectrum_curve:
        return None

    x_obs_raw = spectrum_curve.get("x") or spectrum_curve.get("two_theta")
    y_obs_raw = spectrum_curve.get("y") or spectrum_curve.get("intensity")
    if not x_obs_raw or not y_obs_raw:
        return None

    x_obs = np.asarray(x_obs_raw, dtype=float)
    y_obs = np.asarray(y_obs_raw, dtype=float)

    if len(x_obs) < 50 or len(y_obs) != len(x_obs):
        return None

    # Collect structures
    phase_structures: dict[str, dict[str, Any]] = {}
    for comp in components:
        formula = comp["formula"]
        profile = profiles.get(formula)
        if not profile:
            continue
        structure_dict = profile.get("structure") or profile.get("crystalStructure")
        if not structure_dict:
            continue
        phase_structures[formula] = structure_dict

    if not phase_structures:
        return None

    return refine_full(
        observed_x=x_obs,
        observed_y=y_obs,
        phase_structures=phase_structures,
        wavelength_A=wavelength_A,
    )


# ── Legacy R185-7c-1 API (kept for backward compatibility) ───────────────────

def subtract_background(y: np.ndarray, n_windows: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Moving-min background subtraction. Kept for legacy callers and tests."""
    if len(y) < n_windows:
        return y, np.zeros_like(y)
    window_size = len(y) // n_windows
    bg_x: list[int] = []
    bg_y: list[float] = []
    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(y))
        if end > start:
            bg_x.append((start + end) // 2)
            bg_y.append(float(np.min(y[start:end])))
    if len(bg_x) < 2:
        return y, np.zeros_like(y)
    background = np.interp(np.arange(len(y)), bg_x, bg_y)
    y_corr = np.maximum(y - background, 0)
    return y_corr, background


def refine_scales(observed_x, observed_y, phase_patterns, phase_structures=None) -> RietveldResult:
    """Legacy scale-only refinement (R185-7c-1). Use refine_full for production."""
    # Convert phase_patterns into structures dict; if patterns provided without
    # structures, we cannot compute mass fractions but still fit scales.
    try:
        from lmfit import Minimizer, Parameters
    except ImportError:
        return RietveldResult(converged=False, n_iterations=0, notes=["lmfit unavailable"])

    interp_patterns: dict[str, np.ndarray] = {}
    for formula, (x_sim, y_sim) in phase_patterns.items():
        interp_patterns[formula] = np.interp(observed_x, x_sim, y_sim, left=0, right=0)

    max_obs = float(np.max(observed_y))
    sum_max_sim = sum(float(np.max(y)) for y in interp_patterns.values())
    if sum_max_sim <= 0:
        return RietveldResult(converged=False, n_iterations=0,
                              notes=["All simulated patterns are zero"])
    initial_scale = max_obs / sum_max_sim

    params = Parameters()
    formulas = list(interp_patterns.keys())
    for f in formulas:
        var = f"scale_{f.replace('+', 'p').replace('-', 'm')}"
        params.add(var, value=initial_scale, min=0.0)

    weights = 1.0 / np.sqrt(np.maximum(observed_y, 1.0))

    def residuals(p):
        model = np.zeros_like(observed_y, dtype=float)
        for f in formulas:
            var = f"scale_{f.replace('+', 'p').replace('-', 'm')}"
            model += p[var].value * interp_patterns[f]
        return (observed_y - model) * weights

    out = Minimizer(residuals, params).minimize(method="leastsq", max_nfev=200)
    chi2 = float(out.chisqr) if out.chisqr is not None else None

    phase_results: list[RietveldPhaseResult] = []
    for f in formulas:
        var = f"scale_{f.replace('+', 'p').replace('-', 'm')}"
        param = out.params[var]
        phase_results.append(RietveldPhaseResult(
            formula=f,
            scale_factor=round(float(param.value), 6),
            scale_uncertainty=round(float(param.stderr or 0.0), 6),
            mass_fraction=0.0, mass_uncertainty=0.0,
        ))

    return RietveldResult(
        converged=bool(out.success),
        n_iterations=int(out.nfev) if out.nfev else 0,
        chi_squared=round(chi2, 4) if chi2 is not None else None,
        phases=phase_results,
    )


def simulate_phase_pattern(structure_dict, formula, two_theta_range, wavelength_A=1.5406,
                           fwhm_deg=0.1, n_points=1000):
    """Legacy simple Gaussian-convoluted pattern. Used by old tests."""
    bragg = _extract_bragg_reflections(structure_dict, two_theta_range, wavelength_A) if structure_dict else None
    if bragg is None:
        return None
    bragg_x, bragg_i = bragg
    x = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    sigma = fwhm_deg / (2 * math.sqrt(2 * math.log(2)))
    y = np.zeros_like(x)
    for c, i in zip(bragg_x, bragg_i, strict=True):
        if i <= 0:
            continue
        y += i * np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    return x, y
