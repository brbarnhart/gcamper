from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from .core import isosbestic_fit
from .helpers import check_required_cols


def annotate_events(
    ax: Axes,
    events: pl.DataFrame,
    palette: dict[str, tuple] | None = None,
    *,
    time_col: str = "time",
    note_col: str = "Note",
) -> None:
    """Draw a vertical line (and legend entry) for each note event.

    Repeated note labels share a color; only the first occurrence of each
    label is added to the legend.
    """
    if events.is_empty() or note_col not in events.columns:
        return

    notes = events[note_col].to_list()
    if palette is None:
        unique = list(dict.fromkeys(notes))
        palette = dict(zip(unique, sns.color_palette("tab10", n_colors=len(unique))))

    seen: set[str] = set()
    for row in events.iter_rows(named=True):
        note = row[note_col]
        label = note if note not in seen else None
        seen.add(note)
        ax.axvline(
            row[time_col],
            color=palette[note],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )
    ax.legend()


def _session_events(
    df: pl.DataFrame, time_col: str, note_col: str
) -> pl.DataFrame:
    if note_col not in df.columns:
        return pl.DataFrame()
    return (
        df.filter(pl.col(note_col).is_not_null())
        .select(time_col, note_col)
        .sort(time_col)
    )


def _event_palette(events: pl.DataFrame, note_col: str) -> dict[str, tuple]:
    if events.is_empty() or note_col not in events.columns:
        return {}
    notes = list(dict.fromkeys(events[note_col].to_list()))
    return dict(zip(notes, sns.color_palette("tab10", n_colors=len(notes))))


def _iter_groups(df: pl.DataFrame, by: Sequence[str]):
    by = list(by)
    if not by:
        yield {}, df
        return
    for group in df.partition_by(by, maintain_order=True):
        yield {c: group[c][0] for c in by}, group


def _group_stem(keys: dict) -> str:
    if not keys:
        return "session"
    return "_".join(f"{k}-{v}" for k, v in keys.items())


def _group_title(keys: dict) -> str:
    if not keys:
        return ""
    if len(keys) == 1:
        return str(next(iter(keys.values())))
    return ", ".join(f"{k}={v}" for k, v in keys.items())


def plot_raw_traces(
    df: pl.DataFrame,
    *,
    time_col: str = "time",
    gcamp_col: str = "gcamp",
    isos_col: str = "isos",
    note_col: str = "Note",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot raw GCaMP and isosbestic traces versus time.

    Note onsets are marked when ``note_col`` is present.
    """
    check_required_cols(df, [time_col, gcamp_col, isos_col])
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        df[time_col].to_numpy(),
        df[gcamp_col].to_numpy(),
        color="blue",
        linewidth=0.5,
        label="gcamp",
    )
    ax.plot(
        df[time_col].to_numpy(),
        df[isos_col].to_numpy(),
        color="green",
        linewidth=0.5,
        label="isos",
    )
    events = _session_events(df, time_col, note_col)
    annotate_events(
        ax,
        events,
        _event_palette(events, note_col),
        time_col=time_col,
        note_col=note_col,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fluorescence")
    if title:
        ax.set_title(title)
    return ax


def plot_isosbestic_fit(
    df: pl.DataFrame,
    *,
    time_col: str = "time",
    gcamp_col: str = "gcamp",
    isos_col: str = "isos",
    note_col: str = "Note",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot GCaMP with the fitted isosbestic estimate ``Y_fit`` overlaid.

    ``Y_fit`` is the linear map ``gcamp ≈ a * isos + b`` used by
    ``regression_based_dff``. If the control channel is a good match,
    orange should sit in the middle of blue for the whole recording.
    """
    check_required_cols(df, [time_col, gcamp_col, isos_col])
    if ax is None:
        _, ax = plt.subplots()

    time = df[time_col].to_numpy()
    gcamp = df[gcamp_col].to_numpy()
    y_fit = isosbestic_fit(df[isos_col].to_numpy(), gcamp)
    ax.plot(time, gcamp, color="blue", linewidth=0.5, label="gcamp")
    ax.plot(time, y_fit, color="orange", linewidth=1.0, label="Y_fit")
    events = _session_events(df, time_col, note_col)
    annotate_events(
        ax,
        events,
        _event_palette(events, note_col),
        time_col=time_col,
        note_col=note_col,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fluorescence")
    if title:
        ax.set_title(title)
    return ax


def plot_isosbestic_scatter(
    df: pl.DataFrame,
    *,
    time_col: str = "time",
    gcamp_col: str = "gcamp",
    isos_col: str = "isos",
    ax: Axes | None = None,
    title: str | None = None,
    max_points: int = 50_000,
) -> Axes:
    """Scatter GCaMP vs isosbestic, colored by time, with the linear fit.

    Early (purple) and late (yellow) points on opposite sides of the red
    line mean the 465–405 relationship is changing over the session
    (typically different bleaching rates).
    """
    check_required_cols(df, [time_col, gcamp_col, isos_col])
    if ax is None:
        _, ax = plt.subplots()

    time = df[time_col].to_numpy()
    isos = df[isos_col].to_numpy()
    gcamp = df[gcamp_col].to_numpy()
    bls = np.polyfit(isos, gcamp, 1)

    step = max(1, time.size // max_points) if max_points > 0 else 1
    sc = ax.scatter(
        isos[::step],
        gcamp[::step],
        c=time[::step],
        s=2,
        cmap="viridis",
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    x_line = np.linspace(float(np.min(isos)), float(np.max(isos)), 200)
    ax.plot(x_line, np.polyval(bls, x_line), color="red", linewidth=1.5, label="fit")
    ax.figure.colorbar(sc, ax=ax, label="time (s)")
    ax.set_xlabel("isos")
    ax.set_ylabel("gcamp")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_isosbestic_diagnostics(
    df: pl.DataFrame,
    *,
    by: Sequence[str] = (),
    time_col: str = "time",
    gcamp_col: str = "gcamp",
    isos_col: str = "isos",
    note_col: str = "Note",
    save_dir: str | Path | None = None,
    show: bool = False,
    max_scatter_points: int = 50_000,
    dpi: int = 300,
) -> list[dict[str, object]]:
    """Make the three isosbestic diagnostic figures for each recording.

    For every group in ``by`` (or the whole table if ``by`` is empty):

    1. Raw GCaMP and isosbestic vs time
    2. GCaMP with ``Y_fit`` overlaid
    3. GCaMP vs isosbestic scatter, colored by time

    Parameters
    ----------
    df:
        Photometry dataframe. Pass bleach-corrected columns via
        ``gcamp_col`` / ``isos_col`` (e.g. ``gcamp_bc``, ``isos_bc``) to
        inspect the traces after ``bleach_correct_df``.
    by:
        Columns that uniquely identify one continuous recording. Empty
        (default) treats all of ``df`` as a single session.
    save_dir:
        If set, write ``Raw_{stem}.png``, ``Fit_{stem}.png``, and
        ``Scatter_{stem}.png`` into this directory.
    show:
        If True, call ``plt.show()`` after each figure (notebook-friendly).
    max_scatter_points:
        Downsample the scatter to about this many points.
    dpi:
        Resolution used when ``save_dir`` is set.

    Returns
    -------
    list of dict
        One dict per recording with ``keys``, ``raw``, ``fit``, and
        ``scatter`` (matplotlib Figures).
    """
    by = list(by)
    check_required_cols(df, [time_col, gcamp_col, isos_col, *by])

    out_dir: Path | None = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for keys, group in _iter_groups(df, by):
        title = _group_title(keys)
        stem = _group_stem(keys)

        fig_raw, ax_raw = plt.subplots()
        plot_raw_traces(
            group,
            time_col=time_col,
            gcamp_col=gcamp_col,
            isos_col=isos_col,
            note_col=note_col,
            ax=ax_raw,
            title=title or None,
        )

        fig_fit, ax_fit = plt.subplots()
        plot_isosbestic_fit(
            group,
            time_col=time_col,
            gcamp_col=gcamp_col,
            isos_col=isos_col,
            note_col=note_col,
            ax=ax_fit,
            title=title or None,
        )

        fig_sc, ax_sc = plt.subplots()
        plot_isosbestic_scatter(
            group,
            time_col=time_col,
            gcamp_col=gcamp_col,
            isos_col=isos_col,
            ax=ax_sc,
            title=title or None,
            max_points=max_scatter_points,
        )

        if out_dir is not None:
            fig_raw.savefig(out_dir / f"Raw_{stem}.png", dpi=dpi, bbox_inches="tight")
            fig_fit.savefig(out_dir / f"Fit_{stem}.png", dpi=dpi, bbox_inches="tight")
            fig_sc.savefig(
                out_dir / f"Scatter_{stem}.png", dpi=dpi, bbox_inches="tight"
            )

        if show:
            plt.show()

        results.append(
            {"keys": keys, "raw": fig_raw, "fit": fig_fit, "scatter": fig_sc}
        )

    return results
