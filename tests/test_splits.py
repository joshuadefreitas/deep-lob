import numpy as np

from deep_lob.splits import (
    chronological_split,
    n_windows_for,
    overlap_fraction,
    purged_embargoed_split,
    random_overlap_split,
)


WINDOW_SIZE = 20
HORIZON = 5
N_ROWS = 500


def _n_windows():
    return n_windows_for(N_ROWS, WINDOW_SIZE, HORIZON)


def test_n_windows_matches_build_lob_windows_formula():
    assert _n_windows() == N_ROWS - WINDOW_SIZE - HORIZON + 1


def test_chronological_split_is_strictly_time_ordered():
    n = _n_windows()
    split = chronological_split(n, train_frac=0.8)
    assert split.train_idx.max() < split.val_idx.min()
    assert len(split.train_idx) + len(split.val_idx) == n


def test_random_overlap_split_has_high_overlap():
    n = _n_windows()
    split = random_overlap_split(n, train_frac=0.8, seed=0)
    frac = overlap_fraction(split.train_idx, split.val_idx, WINDOW_SIZE, HORIZON)
    # With window_size + horizon = 25 and windows shuffled uniformly across
    # ~475 indices, the vast majority of val windows land within 25 raw
    # rows of some train window.
    assert frac > 0.9


def test_random_overlap_split_is_deterministic_given_seed():
    n = _n_windows()
    a = random_overlap_split(n, train_frac=0.8, seed=7)
    b = random_overlap_split(n, train_frac=0.8, seed=7)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.val_idx, b.val_idx)


def test_purged_embargoed_split_has_zero_overlap():
    n = _n_windows()
    split = purged_embargoed_split(n, WINDOW_SIZE, HORIZON, train_frac=0.8)
    frac = overlap_fraction(split.train_idx, split.val_idx, WINDOW_SIZE, HORIZON)
    assert frac == 0.0


def test_purged_embargoed_split_is_smaller_than_chronological():
    n = _n_windows()
    chrono = chronological_split(n, train_frac=0.8)
    purged = purged_embargoed_split(n, WINDOW_SIZE, HORIZON, train_frac=0.8)
    assert len(purged.train_idx) <= len(chrono.train_idx)
    assert len(purged.val_idx) <= len(chrono.val_idx)
    # Purging should actually remove something for these parameters.
    assert len(purged.train_idx) < len(chrono.train_idx)


def test_embargo_actually_removes_validation_windows():
    """
    The embargo is unenforced without this.

    purged_embargoed_split does two things: it PURGES training windows whose
    raw-row span reaches into the validation region, and it EMBARGOES a buffer
    of validation windows after the cut. Every other assertion in this file is
    satisfied by purging alone - an independent audit demonstrated it by setting
    embargo=0 and watching all 50 tests stay green.

    So this asserts the embargo's own effect: it must shrink the validation set
    relative to a plain chronological cut, and the first validation window must
    sit at least `horizon` windows past the cut.
    """
    n = _n_windows()
    chrono = chronological_split(n, train_frac=0.8)
    purged = purged_embargoed_split(n, WINDOW_SIZE, HORIZON, train_frac=0.8)
    cut = int(0.8 * n)

    assert len(purged.val_idx) < len(chrono.val_idx), (
        "the embargo removed no validation windows; it is not in effect"
    )
    assert int(purged.val_idx[0]) - cut >= HORIZON, (
        f"first validation window is {int(purged.val_idx[0]) - cut} past the cut, "
        f"expected at least {HORIZON}"
    )


def test_embargo_length_is_respected():
    """An explicit embargo must be honoured exactly, not approximately."""
    n = _n_windows()
    cut = int(0.8 * n)
    for embargo in (0, 5, 25):
        sp = purged_embargoed_split(n, WINDOW_SIZE, HORIZON, train_frac=0.8,
                                    embargo=embargo)
        assert int(sp.val_idx[0]) == cut + embargo, (
            f"embargo={embargo}: validation starts at {int(sp.val_idx[0])}, "
            f"expected {cut + embargo}"
        )


def test_chronological_split_can_still_have_boundary_overlap():
    # Sanity check that chronological-without-purge is a genuinely weaker
    # control than purged_embargoed: with window_size+horizon=25 windows
    # adjacent to the cut share raw rows.
    n = _n_windows()
    split = chronological_split(n, train_frac=0.8)
    frac = overlap_fraction(split.train_idx, split.val_idx, WINDOW_SIZE, HORIZON)
    assert frac > 0.0


def test_overlap_fraction_empty_inputs():
    assert overlap_fraction(np.array([]), np.array([1, 2, 3]), WINDOW_SIZE, HORIZON) == 0.0
    assert overlap_fraction(np.array([1, 2, 3]), np.array([]), WINDOW_SIZE, HORIZON) == 0.0
