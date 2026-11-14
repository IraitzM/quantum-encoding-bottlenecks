import numpy as np

from analysis import tables


def test_recording_bootstrap_resamples_whole_groups(monkeypatch):
    monkeypatch.setattr(tables, "N_BOOT", 257)
    groups = np.repeat(["recording-a", "recording-b", "recording-c"], [2, 3, 1])
    y = np.array([0, 0, 0, 1, 1, 1])

    weights = tables._test_bootstrap_weights(y, groups, np.random.default_rng(12))

    assert weights.shape == (tables.N_BOOT, len(y))
    np.testing.assert_array_equal(weights, np.floor(weights))
    unique = np.unique(groups)
    multiplicities = []
    for group in unique:
        rows = weights[:, groups == group]
        np.testing.assert_array_equal(rows, np.repeat(rows[:, :1], rows.shape[1], axis=1))
        multiplicities.append(rows[:, 0])
    # Three whole recordings are drawn per replicate, although their window counts differ.
    np.testing.assert_array_equal(np.stack(multiplicities).sum(axis=0), np.full(tables.N_BOOT, len(unique)))


def test_example_bootstrap_is_unchanged_without_groups(monkeypatch):
    monkeypatch.setattr(tables, "N_BOOT", 17)

    weights = tables._test_bootstrap_weights(np.array([0, 1, 1]), None, np.random.default_rng(5))
    expected = np.random.default_rng(5).multinomial(3, np.full(3, 1 / 3), size=17).astype(float)

    np.testing.assert_array_equal(weights, expected)
