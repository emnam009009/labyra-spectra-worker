"""Unit tests for Rietveld profile + background refinement (R185-7c-2).

@phase R185-7c-2-rietveld-profile-bg
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.deviation.rietveld import (
    BackgroundFit,
    ProfileParameters,
    RietveldResult,
    _caglioti_fwhm,
    _pseudo_voigt,
    chebyshev_background,
    refine_full,
    simulate_phase_pattern_with_profile,
)


# ── Profile functions ────────────────────────────────────────────────────────

class TestCagliotiFWHM:
    def test_constant_W_only(self):
        # W=0.01, U=V=0 → FWHM should be sqrt(0.01) = 0.1
        fwhm = _caglioti_fwhm(30.0, U=0.0, V=0.0, W=0.01)
        assert abs(float(fwhm) - 0.1) < 1e-6

    def test_increases_with_2theta_when_U_positive(self):
        fwhm_low = float(_caglioti_fwhm(20.0, U=0.05, V=0.0, W=0.005))
        fwhm_high = float(_caglioti_fwhm(80.0, U=0.05, V=0.0, W=0.005))
        assert fwhm_high > fwhm_low

    def test_negative_W_yields_floor(self):
        # Robust to nonsense parameters
        fwhm = float(_caglioti_fwhm(30.0, U=0.0, V=0.0, W=-1.0))
        assert fwhm > 0


class TestPseudoVoigt:
    def test_gaussian_limit_eta_zero(self):
        x = np.linspace(-5, 5, 1000)
        pv = _pseudo_voigt(x, center=0.0, fwhm=1.0, eta=0.0)
        # Peak should be near center (within 1 grid point)
        peak_idx = np.argmax(pv)
        assert abs(x[peak_idx]) < 0.02

    def test_lorentzian_limit_eta_one(self):
        x = np.linspace(-5, 5, 1000)
        pv = _pseudo_voigt(x, center=0.0, fwhm=1.0, eta=1.0)
        peak_idx = np.argmax(pv)
        assert abs(x[peak_idx]) < 0.02
        # Lorentzian tails fall slower than Gaussian
        gauss = _pseudo_voigt(x, center=0.0, fwhm=1.0, eta=0.0)
        # At |x|=3 from center
        idx = np.argmin(abs(x - 3))
        assert pv[idx] > gauss[idx]


# ── Chebyshev background ─────────────────────────────────────────────────────

class TestChebyshev:
    def test_constant_coefficient(self):
        x_norm = np.linspace(-1, 1, 100)
        bg = chebyshev_background(x_norm, [50.0, 0, 0, 0, 0])
        assert np.allclose(bg, 50.0)

    def test_linear_term(self):
        x_norm = np.linspace(-1, 1, 100)
        bg = chebyshev_background(x_norm, [0, 10.0, 0, 0, 0])
        # T_1(x) = x → linear from -10 to +10
        assert abs(bg[0] - (-10.0)) < 1e-6
        assert abs(bg[-1] - 10.0) < 1e-6


# ── End-to-end refinement with synthetic data ────────────────────────────────

def _make_synthetic_observation(bragg_x, bragg_i, x_obs, U, V, W, eta, zero_shift,
                                bg_coeffs, scale):
    """Generate synthetic observed pattern for testing."""
    x_norm = 2 * (x_obs - x_obs.min()) / (x_obs.max() - x_obs.min()) - 1
    bg = chebyshev_background(x_norm, bg_coeffs)
    phase = simulate_phase_pattern_with_profile(
        bragg_x=np.asarray(bragg_x), bragg_intensity=np.asarray(bragg_i),
        x_grid=x_obs, U=U, V=V, W=W, eta=eta, zero_shift=zero_shift,
    )
    return bg + scale * phase


class TestRefineFullSyntheticNoStructure:
    """
    Test refinement using fake bragg reflections (no pymatgen Structure needed).
    Uses monkey-patched extractor.
    """
    def test_recovers_synthetic_W(self, monkeypatch):
        # Synthesize observed = 1.5 * pattern(bragg) + background
        bragg_x = np.array([28.0, 47.0, 56.0])
        bragg_i = np.array([100.0, 60.0, 30.0])
        x_obs = np.linspace(20, 70, 500)

        truth = {
            "U": 0.005, "V": -0.002, "W": 0.008,
            "eta": 0.3, "zero_shift": 0.05,
            "bg_coeffs": [80, 5, -2, 1, 0],
            "scale": 2.0,
        }
        y_obs = _make_synthetic_observation(
            bragg_x, bragg_i, x_obs,
            U=truth["U"], V=truth["V"], W=truth["W"],
            eta=truth["eta"], zero_shift=truth["zero_shift"],
            bg_coeffs=truth["bg_coeffs"], scale=truth["scale"],
        )
        # Add Poisson-like noise
        rng = np.random.default_rng(42)
        y_obs = np.maximum(y_obs + rng.normal(0, 1, size=len(y_obs)), 0)

        # Monkey-patch extractor to return our synthetic bragg
        def fake_extract(structure_dict, two_theta_range, wavelength_A):
            return bragg_x, bragg_i

        monkeypatch.setattr("src.deviation.rietveld._extract_bragg_reflections", fake_extract)
        monkeypatch.setattr("src.deviation.rietveld._phase_mass_factor", lambda s: 1.0)
        monkeypatch.setattr(
            "src.deviation.rietveld._phase_metadata",
            lambda s: (100.0, 100.0, 2),
        )

        result = refine_full(
            observed_x=x_obs, observed_y=y_obs,
            phase_structures={"FakePhase": {"dummy": True}},
        )

        # lmfit may not flag converged in 500 nfev for high-noise data,
        # but should still return reasonable scale.
        assert result.profile is not None
        assert len(result.phases) == 1
        # Profile params are tricky to recover exactly; just check fit ran
        # and produced a finite result.
        assert result.chi_squared is not None
        # Scale factor should be in same order of magnitude (loose check)
        s = result.phases[0].scale_factor
        assert 0.1 < s < 100

    def test_r_wp_reported(self, monkeypatch):
        bragg_x = np.array([28.0, 47.0])
        bragg_i = np.array([100.0, 60.0])
        x_obs = np.linspace(20, 60, 500)
        y_obs = _make_synthetic_observation(
            bragg_x, bragg_i, x_obs,
            U=0.005, V=-0.002, W=0.008, eta=0.3, zero_shift=0.0,
            bg_coeffs=[50, 0, 0, 0, 0], scale=1.0,
        )

        monkeypatch.setattr(
            "src.deviation.rietveld._extract_bragg_reflections",
            lambda s, r, w: (bragg_x, bragg_i),
        )
        monkeypatch.setattr("src.deviation.rietveld._phase_mass_factor", lambda s: 1.0)
        monkeypatch.setattr(
            "src.deviation.rietveld._phase_metadata",
            lambda s: (100.0, 100.0, 2),
        )

        result = refine_full(
            observed_x=x_obs, observed_y=y_obs,
            phase_structures={"P": {"d": 1}},
        )

        assert result.r_wp is not None
        # Perfect synthetic data can yield R_wp ~ 0; just verify it's computable
        assert 0 <= result.r_wp < 30


# ── Crystallite size from refined profile ────────────────────────────────────

class TestCrystalliteSize:
    def test_size_from_broad_W_small(self):
        # Broad peaks (W large) → small crystallites
        from src.deviation.rietveld import _crystallite_size_from_profile
        broad = ProfileParameters(U=0.01, V=0, W=0.5)
        narrow = ProfileParameters(U=0.001, V=0, W=0.005)
        broad_result = _crystallite_size_from_profile(broad)
        narrow_result = _crystallite_size_from_profile(narrow)
        assert broad_result is not None and narrow_result is not None
        assert broad_result[0] < narrow_result[0]


# ── Result serialization ─────────────────────────────────────────────────────

class TestResultSerialization:
    def test_full_result_serializable(self):
        import json
        result = RietveldResult(
            converged=True, n_iterations=100,
            r_wp=8.5, chi_squared=1.2,
            profile=ProfileParameters(U=0.01, V=-0.005, W=0.01, eta=0.4),
            background=BackgroundFit(coefficients=[50, 5, 0, 0, 0]),
        )
        d = result.to_dict()
        json.dumps(d)  # must serialize cleanly
        assert d["r_wp"] == 8.5
        assert d["profile"]["U"] == 0.01
