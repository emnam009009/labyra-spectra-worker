"""Test Dans_Diffraction XRD simulation."""

import pytest

WO3_CIF = """data_test
_cell_length_a 7.589
_cell_length_b 7.589
_cell_length_c 3.886
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
W1 W 0.5 0.0 0.5
O1 O 0.0 0.0 0.5
O2 O 0.5 0.5 0.5
O3 O 0.5 0.0 0.0
"""


def test_simulate_basic():
    from src.citation.xrd_simulator import simulate_powder_pattern

    peaks = simulate_powder_pattern(WO3_CIF, max_twotheta=80)
    assert len(peaks) > 5, f"Expected >5 peaks, got {len(peaks)}"
    # First peak fields
    p0 = peaks[0]
    assert "twotheta" in p0
    assert "intensity" in p0
    assert "relative_intensity" in p0
    # 2θ should be in range
    for p in peaks:
        assert 10 <= p["twotheta"] <= 80


def test_simulate_empty_on_bad_cif():
    from src.citation.xrd_simulator import simulate_powder_pattern

    # Use truly malformed CIF without _cell_length entries (Dans_Diffraction will fail loading)
    bad_cif = "data_test\nnot_a_valid_field bogus\n"
    peaks = simulate_powder_pattern(bad_cif)
    # Either empty or doesn't crash
    assert isinstance(peaks, list)
