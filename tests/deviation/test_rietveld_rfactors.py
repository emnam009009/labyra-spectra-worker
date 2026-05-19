"""R-factors + difference plot + per-phase R_Bragg tests.

@phase R185-10d-1 / R185-7c-3
"""
from __future__ import annotations

import math
import numpy as np

from src.deviation.rietveld import (
    chebyshev_background,
    refine_full,
    simulate_phase_pattern_with_profile,
)


def _make_synthetic(bragg_x, bragg_i, x, U, V, W, eta, zs, bg, scale):
    x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    bg_curve = chebyshev_background(x_norm, bg)
    phase = simulate_phase_pattern_with_profile(
        bragg_x=np.asarray(bragg_x), bragg_intensity=np.asarray(bragg_i),
        x_grid=x, U=U, V=V, W=W, eta=eta, zero_shift=zs,
    )
    return bg_curve + scale * phase


class TestRFactors:
    def test_all_rfactors_computed(self, monkeypatch):
        bragg_x = np.array([28.0, 47.0, 56.0])
        bragg_i = np.array([100.0, 60.0, 30.0])
        x_obs = np.linspace(20, 70, 500)
        y_obs = _make_synthetic(
            bragg_x, bragg_i, x_obs,
            U=0.005, V=-0.002, W=0.008, eta=0.3, zs=0.0,
            bg=[50, 0, 0, 0, 0], scale=2.0,
        )
        rng = np.random.default_rng(7)
        y_obs = np.maximum(y_obs + rng.normal(0, 1, size=len(y_obs)), 0)

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
            phase_structures={"P": {"dummy": True}},
        )

        # All R-factors present
        assert result.r_wp is not None
        assert result.r_p is not None
        assert result.r_exp is not None
        assert result.gof is not None
        # Diff plot computed
        assert result.difference_plot is not None
        assert "x" in result.difference_plot
        assert "y_obs" in result.difference_plot
        assert "y_calc" in result.difference_plot
        assert "diff" in result.difference_plot
        # Phase contributions
        assert result.phase_contributions is not None
        assert "P" in result.phase_contributions

    def test_gof_acceptable_for_clean_data(self, monkeypatch):
        bragg_x = np.array([28.0, 47.0])
        bragg_i = np.array([100.0, 60.0])
        x_obs = np.linspace(20, 60, 500)
        y_obs = _make_synthetic(
            bragg_x, bragg_i, x_obs,
            U=0.005, V=-0.002, W=0.008, eta=0.3, zs=0.0,
            bg=[50, 0, 0, 0, 0], scale=1.0,
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

        # No noise → very low R-factors
        assert result.r_wp < 5
        # GoF should be finite + reasonable
        assert result.gof is not None
        assert result.gof < 100


class TestRBraggPerPhase:
    def test_r_bragg_populated_for_each_phase(self, monkeypatch):
        bragg_x = np.array([28.0, 47.0])
        bragg_i = np.array([100.0, 60.0])
        x_obs = np.linspace(20, 60, 500)
        y_obs = _make_synthetic(
            bragg_x, bragg_i, x_obs,
            U=0.005, V=-0.002, W=0.008, eta=0.3, zs=0.0,
            bg=[50, 0, 0, 0, 0], scale=1.5,
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
            phase_structures={"PhaseA": {"d": 1}},
        )

        assert len(result.phases) == 1
        assert result.phases[0].r_bragg is not None
        assert 0 <= result.phases[0].r_bragg <= 100


class TestDifferencePlotPayloadSize:
    def test_diff_plot_downsampled_to_about_200_pts(self, monkeypatch):
        bragg_x = np.array([28.0])
        bragg_i = np.array([100.0])
        x_obs = np.linspace(20, 60, 2000)  # 2000 points
        y_obs = _make_synthetic(
            bragg_x, bragg_i, x_obs,
            U=0.005, V=-0.002, W=0.008, eta=0.3, zs=0.0,
            bg=[50, 0, 0, 0, 0], scale=1.0,
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

        assert result.difference_plot is not None
        # ~200 points target
        assert 150 < len(result.difference_plot["x"]) < 300


class TestSerialization:
    def test_extended_result_json_safe(self):
        import json
        from src.deviation.rietveld import (
            BackgroundFit, ProfileParameters,
            RietveldPhaseResult, RietveldResult,
        )
        result = RietveldResult(
            converged=True, n_iterations=100,
            r_wp=8.5, r_p=6.2, r_exp=4.1, gof=2.07,
            chi_squared=1.2,
            phases=[RietveldPhaseResult(
                formula="P", scale_factor=1.0, scale_uncertainty=0.01,
                mass_fraction=1.0, mass_uncertainty=0.02, r_bragg=5.5,
            )],
            profile=ProfileParameters(),
            background=BackgroundFit(),
            difference_plot={"x": [1.0], "y_obs": [10.0], "y_calc": [9.5], "diff": [0.5]},
            phase_contributions={"P": [9.5]},
        )
        d = result.to_dict()
        json.dumps(d)
        assert d["r_p"] == 6.2
        assert d["gof"] == 2.07
        assert d["phases"][0]["r_bragg"] == 5.5
        assert d["difference_plot"]["diff"] == [0.5]
