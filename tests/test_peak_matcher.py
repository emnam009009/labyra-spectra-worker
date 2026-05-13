"""Test peak matcher scoring."""

from src.citation.peak_matcher import match_peaks


def test_match_perfect():
    """If user peaks exactly match simulated, score should be high."""
    user = [
        {"two_theta": 23.1, "intensity": 100, "fwhm": 0.1, "relative_intensity": 1.0},
        {"two_theta": 23.6, "intensity": 80, "fwhm": 0.1, "relative_intensity": 0.8},
        {"two_theta": 24.4, "intensity": 60, "fwhm": 0.1, "relative_intensity": 0.6},
    ]
    sim = [
        {"twotheta": 23.1, "intensity": 1000, "relative_intensity": 1.0, "multiplicity": 4, "hkl": [1,0,0]},
        {"twotheta": 23.6, "intensity": 800, "relative_intensity": 0.8, "multiplicity": 4, "hkl": [0,1,0]},
        {"twotheta": 24.4, "intensity": 600, "relative_intensity": 0.6, "multiplicity": 4, "hkl": [0,0,1]},
    ]
    result = match_peaks(user, sim)
    assert result["matched_count"] == 3
    assert result["match_ratio"] == 1.0
    assert result["score"] > 0.8


def test_match_no_overlap():
    user = [{"two_theta": 23.1, "intensity": 100, "fwhm": 0.1, "relative_intensity": 1.0}]
    sim = [{"twotheta": 50.0, "intensity": 1000, "relative_intensity": 1.0, "multiplicity": 1, "hkl": [1,1,1]}]
    result = match_peaks(user, sim)
    assert result["matched_count"] == 0
    assert result["score"] == 0.0


def test_match_partial():
    user = [
        {"two_theta": 23.1, "intensity": 100, "fwhm": 0.1, "relative_intensity": 1.0},
        {"two_theta": 50.0, "intensity": 50, "fwhm": 0.1, "relative_intensity": 0.5},
    ]
    sim = [
        {"twotheta": 23.1, "intensity": 1000, "relative_intensity": 1.0, "multiplicity": 1, "hkl": [1,0,0]},
    ]
    result = match_peaks(user, sim)
    assert result["matched_count"] == 1
    assert result["match_ratio"] == 0.5
    assert 0.3 < result["score"] < 0.5
