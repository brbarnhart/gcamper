import numpy as np
import polars as pl
import pytest

from gcamper import (
    bleach_correct,
    bleach_correct_df,
    fit_bleach_curve,
    isosbestic_fit,
    regression_based_dff,
)


def _synthetic_bleach(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2000.0, n)
    true = 5.0 * np.exp(-t / 80.0) + 4.0 * np.exp(-t / 1500.0) + 40.0
    y = true + rng.normal(0.0, 0.03, size=n)
    return t, true, y


def test_fit_bleach_curve_recovers_double_exp_envelope():
    t, true, y = _synthetic_bleach()
    fit = fit_bleach_curve(t, y, bin_s=5.0)
    assert fit.shape == y.shape
    rmse = float(np.sqrt(np.mean((fit - true) ** 2)))
    assert rmse < 0.5


def test_bleach_correct_flattens_decay():
    t, _, y = _synthetic_bleach()
    bc = bleach_correct(t, y, bin_s=5.0)
    assert bc.shape == y.shape
    assert np.mean(bc) == pytest.approx(1.0, abs=0.05)
    slope = np.polyfit(t, bc, 1)[0]
    raw_slope = np.polyfit(t, y, 1)[0]
    assert abs(slope) < abs(raw_slope) / 10


def test_bleach_correct_df_adds_fit_and_bc_columns():
    t, _, gcamp = _synthetic_bleach(seed=1)
    _, _, isos = _synthetic_bleach(seed=2)
    df = pl.DataFrame(
        {
            "ID": ["A"] * t.size,
            "time": t,
            "gcamp": gcamp,
            "isos": isos,
        }
    )
    out = bleach_correct_df(df, by=["ID"])
    for col in ("gcamp_fit", "gcamp_bc", "isos_fit", "isos_bc"):
        assert col in out.columns
    assert out.height == df.height


def test_bleach_correct_df_is_per_session():
    t, _, y_a = _synthetic_bleach(n=400, seed=1)
    _, _, y_b = _synthetic_bleach(n=400, seed=3)
    df = pl.concat(
        [
            pl.DataFrame({"ID": ["A"] * t.size, "time": t, "gcamp": y_a, "isos": y_a}),
            pl.DataFrame({"ID": ["B"] * t.size, "time": t, "gcamp": y_b, "isos": y_b}),
        ]
    )
    out = bleach_correct_df(df, by=["ID"], cols=["gcamp"])
    fit_a = out.filter(pl.col("ID") == "A")["gcamp_fit"].to_numpy()
    fit_b = out.filter(pl.col("ID") == "B")["gcamp_fit"].to_numpy()
    assert not np.allclose(fit_a, fit_b)


def test_fit_bleach_curve_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        fit_bleach_curve(np.arange(10.0), np.arange(9.0))


def test_isosbestic_fit_matches_regression_dff_numerator():
    rng = np.random.default_rng(0)
    isos = rng.normal(30.0, 1.0, size=500)
    gcamp = 1.4 * isos + 2.0 + rng.normal(0.0, 0.1, size=500)
    y_fit = isosbestic_fit(isos, gcamp)
    dff = regression_based_dff(isos, gcamp)
    expected = (100 * (gcamp - y_fit) / y_fit).astype(np.float32)
    np.testing.assert_allclose(dff, expected, rtol=1e-5, atol=1e-5)
