from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from .helpers import check_required_cols


def _double_exp(t, a, tau_fast, b, delta_tau, c):
    return a * np.exp(-t / tau_fast) + b * np.exp(-t / (tau_fast + delta_tau)) + c


def _single_exp(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def _binned_median(
    t: np.ndarray, y: np.ndarray, bin_s: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Median fluorescence in successive time bins (slow envelope for the fit)."""
    t_max = float(t[-1])
    n_bins = max(int(np.ceil(t_max / bin_s)), 10)
    edges = np.linspace(0.0, max(t_max, bin_s), n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    t_out, y_out = [], []
    for i in range(n_bins):
        mask = idx == i
        if not np.any(mask):
            continue
        t_out.append(np.median(t[mask]))
        y_out.append(np.median(y[mask]))
    return np.asarray(t_out, dtype=np.float64), np.asarray(y_out, dtype=np.float64)


def _as_1d(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {out.shape})")
    return out


def fit_bleach_curve(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    bin_s: float = 5.0,
) -> np.ndarray:
    """Fit a double-exponential bleaching curve to a fluorescence trace.

    A double exponential plus a constant is fit to the slow envelope of
    ``signal`` (median in ``bin_s``-second bins) so brief transients do not
    pull the timescales:

        F(t) = a * exp(-t / tau_fast) + b * exp(-t / tau_slow) + c

    If that fit fails, a single exponential plus constant is used instead.
    Time is shifted to start at 0 internally. Fit on a single continuous
    recording, not on a stacked multi-session table.

    Parameters
    ----------
    time:
        1-D timestamps in seconds, same length as ``signal``.
    signal:
        1-D fluorescence (e.g. GCaMP or isosbestic).
    bin_s:
        Bin width in seconds for the median envelope (default 5).

    Returns
    -------
    np.ndarray
        Fitted bleach curve, same length as ``signal``.
    """
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive (got {bin_s})")

    time = _as_1d("time", time)
    signal = _as_1d("signal", signal)
    if time.shape != signal.shape:
        raise ValueError(
            f"time and signal must have the same length "
            f"(got {time.shape[0]} and {signal.shape[0]})"
        )
    if time.size < 2:
        raise ValueError("time and signal must contain at least two samples")

    t_rel = time - time[0]
    t_bin, y_bin = _binned_median(t_rel, signal, bin_s=bin_s)
    if y_bin.size < 3:
        raise ValueError(
            "not enough non-empty time bins to fit a bleach curve; "
            "check sampling and bin_s"
        )

    n_head = max(1, len(y_bin) // 20)
    y0 = float(np.median(y_bin[:n_head]))
    yend = float(np.median(y_bin[-n_head:]))
    drop = max(y0 - yend, float(np.max(y_bin) - np.min(y_bin)) * 0.05, 1e-3)
    y_max = float(np.max(y_bin))
    floor = max(yend, 1e-3)

    fit: np.ndarray | None = None
    if y_bin.size >= 5:
        p0 = (0.55 * drop, 90.0, 0.45 * drop, 1200.0, floor)
        bounds = (
            [0.0, 5.0, 0.0, 30.0, 0.0],
            [drop * 5.0, 800.0, drop * 5.0, 2e4, y_max * 1.2],
        )
        try:
            popt, _ = curve_fit(
                _double_exp, t_bin, y_bin, p0=p0, bounds=bounds, maxfev=20000
            )
            fit = _double_exp(t_rel, *popt)
        except (RuntimeError, ValueError):
            fit = None

    if fit is None:
        p0_s = (drop, 400.0, floor)
        bounds_s = ([0.0, 10.0, 0.0], [drop * 5.0, 2e4, y_max * 1.2])
        try:
            popt, _ = curve_fit(
                _single_exp, t_bin, y_bin, p0=p0_s, bounds=bounds_s, maxfev=20000
            )
            fit = _single_exp(t_rel, *popt)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("bleach curve fit failed") from exc

    return np.maximum(fit, np.max(np.abs(signal)) * 1e-6)


def bleach_correct(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    bin_s: float = 5.0,
) -> np.ndarray:
    """Divide a fluorescence trace by its fitted bleach curve.

    Parameters
    ----------
    time, signal, bin_s:
        Passed through to ``fit_bleach_curve``.

    Returns
    -------
    np.ndarray
        ``signal / fit``, same length as ``signal``.
    """
    fit = fit_bleach_curve(time, signal, bin_s=bin_s)
    return np.asarray(signal, dtype=np.float64) / fit


def bleach_correct_df(
    df: pl.DataFrame,
    *,
    by: Sequence[str] = (),
    time_col: str = "time",
    cols: Sequence[str] = ("gcamp", "isos"),
    bin_s: float = 5.0,
) -> pl.DataFrame:
    """Add bleach-fit and bleach-corrected columns for each requested signal.

    For every continuous recording (the whole table, or each ``by`` group)
    and each name in ``cols``, fits a double-exponential bleach curve and
    appends:

    - ``{col}_fit`` — the fitted envelope
    - ``{col}_bc`` — ``col / {col}_fit``

    Then run ``regression_based_dff`` on the ``_bc`` columns if you want
    isosbestic ΔF/F after bleach correction.

    Parameters
    ----------
    df:
        Photometry dataframe with a time column and the signal columns in
        ``cols``.
    by:
        Columns that uniquely identify one continuous recording. Empty
        (default) treats all of ``df`` as a single session. For
        ``create_master_df`` output, use e.g. ``("ID", "Extra")``.
    time_col:
        Timestamp column in seconds.
    cols:
        Signal columns to correct (default ``gcamp`` and ``isos``).
    bin_s:
        Envelope bin width in seconds, passed to ``fit_bleach_curve``.

    Returns
    -------
    pl.DataFrame
        Copy of ``df`` with ``{col}_fit`` and ``{col}_bc`` columns added.
    """
    by = list(by)
    cols = list(cols)
    if not cols:
        raise ValueError("cols must contain at least one signal column")
    check_required_cols(df, [time_col, *cols, *by])

    def _one(group: pl.DataFrame) -> pl.DataFrame:
        t = group[time_col].to_numpy()
        extras: dict[str, np.ndarray] = {}
        for col in cols:
            sig = group[col].to_numpy()
            fit = fit_bleach_curve(t, sig, bin_s=bin_s)
            extras[f"{col}_fit"] = fit
            extras[f"{col}_bc"] = np.asarray(sig, dtype=np.float64) / fit
        return group.with_columns(**extras)

    if not by:
        return _one(df)
    return df.group_by(by, maintain_order=True).map_groups(_one)
