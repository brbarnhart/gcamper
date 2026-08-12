from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import tdt

from .helpers import check_required_cols


def get_file_metadata(path: Path, metadata_fields: list[str]) -> dict[str, str]:
    """Parse subject / session metadata from a TDT block folder name.

    The folder stem is split on ``_``. Leading tokens are mapped to
    ``metadata_fields`` in order; any remaining tokens are joined back with
    ``_`` and stored under ``"Extra"`` (often a date-time session stamp).

    Parameters
    ----------
    path:
        Path to a TDT block directory (stem is used, not the full path).
    metadata_fields:
        Ordered field names for the leading underscore-separated tokens,
        e.g. ``["ID", "Sex"]`` for a stem like ``UI128_M_260806-105325``.

    Returns
    -------
    dict[str, str]
        Mapping of each metadata field to its token, plus ``"Extra"`` for
        leftover path tokens (empty string if none).
    """
    file_fields = path.stem.split("_")
    file_fields = [s.strip() for s in file_fields]

    file_fields_main = file_fields[0 : len(metadata_fields)]

    file_fields_extra = file_fields[len(metadata_fields) :]
    file_fields_extra = "_".join(file_fields_extra)

    metadata = dict(zip(metadata_fields, file_fields_main))
    metadata["Extra"] = file_fields_extra

    return metadata


def create_time_array(tank: tdt.StructType, ref_stream: str) -> np.ndarray:
    """Build a sample-aligned time vector for a TDT stream.

    Uses the reference stream's ``start_time``, sample count, and sampling
    rate (``fs``) so the returned array has one timestamp per sample.

    Parameters
    ----------
    tank:
        TDT block structure (e.g. from ``tdt.read_block``).
    ref_stream:
        Name of the stream whose timing and length define the array
        (e.g. ``"_405A"``).

    Returns
    -------
    np.ndarray
        1-D ``float32`` array of times in seconds.
    """
    stream = tank.streams[ref_stream]
    start_time = stream.start_time
    n = len(stream.data)
    fs = stream.fs
    time_sequence = np.arange(start_time, n / fs, 1 / fs).astype(np.float32)

    return time_sequence


def create_generic_stream(
    tank: tdt.StructType, stream_data: np.ndarray, name: str, ref_stream: str
) -> tdt.StructType:
    """Wrap a numpy array as a TDT-like stream using another stream's timing.

    Copies ``fs`` and ``start_time`` from ``ref_stream`` so the new stream is
    sample-aligned with the photometry channels.

    Parameters
    ----------
    tank:
        TDT block structure providing the reference stream.
    stream_data:
        1-D data array for the new stream.
    name:
        Name assigned to the new stream (e.g. ``"time"``).
    ref_stream:
        Existing stream name used for ``fs`` and ``start_time``.

    Returns
    -------
    tdt.StructType
        Stream-like object with ``name``, ``data``, ``fs``, and ``start_time``.
    """
    stream = tank.streams[ref_stream]
    start_time = stream.start_time
    fs = stream.fs

    kwargs = {
        "name": name,
        "type_str": "streams",
        "data": stream_data,
        "fs": fs,
        "start_time": start_time,
    }

    return tdt.StructType(**kwargs)


def create_time_stream(tank: tdt.StructType, ref_stream: str) -> tdt.StructType:
    """Create a synthetic ``time`` stream aligned to a reference channel.

    Convenience wrapper around ``create_time_array`` and
    ``create_generic_stream``.

    Parameters
    ----------
    tank:
        TDT block structure.
    ref_stream:
        Stream whose sampling defines the time base (typically isosbestic).

    Returns
    -------
    tdt.StructType
        Stream named ``"time"`` with seconds since block start.
    """
    time_stream = create_generic_stream(
        tank,
        stream_data=create_time_array(tank, ref_stream),
        name="time",
        ref_stream=ref_stream,
    )

    return time_stream


def create_session_df(
    file: Path, metadata_fields: list[str], isos: str, gcamp: str
) -> pl.DataFrame:
    """Load one TDT block into a tidy photometry session dataframe.

    Reads isosbestic and GCaMP streams, builds a ``time`` column, attaches
    filename metadata, and left-joins Note epocs (if present) onto the nearest
    sample via ``join_asof``. Notes are sparse: only rows near note onsets
    have non-null ``Note`` / ``Onset`` / ``Offset``.

    Parameters
    ----------
    file:
        Path to a TDT block directory.
    metadata_fields:
        Ordered name tokens parsed from the folder stem (see
        ``get_file_metadata``).
    isos:
        TDT stream name for the isosbestic / control channel (e.g. ``"_405A"``).
    gcamp:
        TDT stream name for the functional channel (e.g. ``"_465A"``).

    Returns
    -------
    pl.DataFrame
        Sorted by ``time``, with metadata columns, ``isos``, ``gcamp``,
        ``time``, ``duration``, and note columns (``Note``, ``Onset``,
        ``Offset``; null-filled if the block has no Note epocs).
    """
    data = tdt.read_block(file)
    data.streams.time = create_time_stream(data, isos)

    metadata = get_file_metadata(file, metadata_fields)

    data_df = (
        pl.DataFrame(
            {
                **metadata,
                "isos": data.streams[isos].data,
                "gcamp": data.streams[gcamp].data,
                "time": data.streams.time.data,
            }
        )
        .with_columns(pl.duration(seconds=pl.col("time")).alias("duration"))
        .sort("time")
    )

    if hasattr(data.epocs, "Note"):
        notes_df = pl.DataFrame(
            {
                "Note": data.epocs.Note.notes,
                "Onset": data.epocs.Note.onset,
                "Offset": data.epocs.Note.offset,
            }
        ).sort("Onset")

        notes_df = notes_df.join_asof(
            data_df.select("time"),
            left_on="Onset",
            right_on="time",
            strategy="nearest",
        )

        final_df = data_df.join(
            notes_df,
            on="time",
            how="left",
        )
    else:
        notes_df = {
            "Note": pl.String,
            "Onset": pl.Float64,
            "Offset": pl.Float64,
        }

        final_df = data_df.with_columns(
            [pl.lit(None).cast(dtype).alias(name) for name, dtype in notes_df.items()]
        )

    return final_df


def create_master_df(
    data_files: list[Path],
    metadata_fields: list[str],
    isos: str,
    gcamp: str,
    experiment_length: pl.Expr | None = None,
    debug: bool = False,
) -> pl.DataFrame:
    """Stack multiple TDT sessions into one multi-recording dataframe.

    Calls ``create_session_df`` for each path and vertically concatenates the
    results. The stacked table is **not** a single continuous time series;
    group by recording keys (e.g. ``ID``, ``Extra``) for dFF fitting, event
    windows, and similar per-session operations.

    Parameters
    ----------
    data_files:
        Ordered list of TDT block directory paths.
    metadata_fields:
        Passed through to ``create_session_df`` / ``get_file_metadata``.
    isos, gcamp:
        TDT stream names for control and functional channels.
    experiment_length:
        Optional Polars expression compared against ``duration``. Rows with
        ``duration`` greater than this value are dropped per session before
        concat (e.g. ``pl.duration(minutes=20)``).
    debug:
        If True, print each file name and its session shape while loading.

    Returns
    -------
    pl.DataFrame
        Vertical concat of all session dataframes (same schema as
        ``create_session_df``).
    """
    df_list = []
    for file in data_files:
        if debug:
            print(file.name)

        _data = create_session_df(file, metadata_fields, isos=isos, gcamp=gcamp)

        if experiment_length is not None:
            _data = _data.filter(pl.col("duration") <= experiment_length)

        if debug:
            print(f"Shape: {_data.shape}")

        df_list.append(_data)

    master_df = pl.concat(df_list, how="vertical")

    return master_df


def regression_based_dff(isos: np.ndarray, gcamp: np.ndarray) -> np.ndarray:
    """Compute regression-based ΔF/F from isosbestic and GCaMP traces.

    Fits a linear model ``gcamp ≈ a * isos + b`` over the full provided
    arrays, subtracts the fitted isosbestic estimate from GCaMP, and
    expresses the residual as a percent of the fit:

        dFF = 100 * (gcamp - Y_fit) / Y_fit

    Fit on a single continuous recording (or a well-defined epoch), not on
    a multi-session master table as if it were one time series.

    Parameters
    ----------
    isos:
        1-D isosbestic / control fluorescence.
    gcamp:
        1-D functional fluorescence, same length as ``isos``.

    Returns
    -------
    np.ndarray
        1-D ``float32`` ΔF/F in percent, same length as the inputs.
    """
    bls = np.polyfit(isos, gcamp, 1)
    Y_fit_all = np.polyval(bls, isos)
    Y_dF_all = gcamp - Y_fit_all
    dFF = (100 * Y_dF_all / Y_fit_all).astype(np.float32)

    return dFF


def get_session_events(
    df: pl.DataFrame,
    *,
    by: Sequence[str] = (),
    time_col: str = "time",
    note_col: str = "Note",
    note: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return one row per note event.

    Notes are sparse in ``create_session_df`` / ``create_master_df`` output
    (null except near onset).

    Parameters
    ----------
    by:
        Columns that identify a continuous recording. Empty (default) treats
        ``df`` as a single session. For multi-animal / multi-session tables use
        e.g. ``("ID", "Extra")`` so ``event_id`` restarts within each recording
        and group keys are kept on the events table.
    """
    by = list(by)
    if by:
        check_required_cols(df, [*by, time_col, note_col])
    else:
        check_required_cols(df, [time_col, note_col])

    select_exprs: list[pl.Expr] = [pl.col(c) for c in by]
    select_exprs.extend(
        [
            pl.col(time_col).alias("event_time"),
            pl.col(note_col).alias("Note"),
        ]
    )
    unique_subset = [*by, "event_time", "Note"]
    sort_cols = [*by, "event_time"] if by else ["event_time"]

    events = (
        df.filter(pl.col(note_col).is_not_null())
        .select(select_exprs)
        .unique(subset=unique_subset)
        .sort(sort_cols)
    )

    if note is not None:
        notes = [note] if isinstance(note, str) else list(note)
        events = events.filter(pl.col("Note").is_in(notes))

    if events.is_empty():
        return events

    if by:
        return events.with_columns(pl.int_range(0, pl.len()).over(by).cast(pl.UInt32).alias("event_id"))
    return events.with_row_index("event_id")


def _prepare_events_table(
    events: pl.DataFrame,
    *,
    by: Sequence[str],
    note: str | Sequence[str] | None,
) -> pl.DataFrame:
    """Validate and normalize a caller-supplied peri-event table.

    Ensures ``event_time`` and ``Note`` (plus any ``by`` columns) are present,
    optionally filters by note label, and assigns ``event_id`` if missing
    (global or within each ``by`` group).
    """
    by = list(by)
    check_required_cols(events, ["event_time", "Note", *by])
    if note is not None:
        notes = [note] if isinstance(note, str) else list(note)
        events = events.filter(pl.col("Note").is_in(notes))
    if "event_id" not in events.columns:
        sort_cols = [*by, "event_time"] if by else ["event_time"]
        events = events.sort(sort_cols)
        if by:
            events = events.with_columns(
                pl.int_range(0, pl.len()).over(by).cast(pl.UInt32).alias("event_id")
            )
        else:
            events = events.with_row_index("event_id")
    return events


def _extract_event_windows_one(
    df: pl.DataFrame,
    events: pl.DataFrame,
    pre: float,
    post: float,
    *,
    value_cols: Sequence[str],
    time_col: str = "time",
    drop_incomplete: bool = True,
    group_label: str | None = None,
) -> pl.DataFrame:
    """Slice peri-event windows from a single continuous recording.

    Internal helper used by ``extract_event_windows``. Builds a shared
    sample-index ``rel_time`` grid from the recording's median ``dt``, maps
    each event to the nearest sample, and returns a long-format window table.
    ``group_label`` is only used to enrich error messages for multi-recording
    callers.
    """
    if events.is_empty():
        return pl.DataFrame()

    value_cols = list(value_cols)
    signal = df.select(time_col, *value_cols).sort(time_col)
    times = signal[time_col].to_numpy()
    value_arrays = {col: signal[col].to_numpy() for col in value_cols}
    context = f" ({group_label})" if group_label else ""

    if times.size < 2:
        raise ValueError(
            f"recording must contain at least two samples to estimate sampling "
            f"rate{context}"
        )

    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"could not estimate a positive sampling interval (got {dt}){context}"
        )

    pre_n = int(round(pre / dt))
    post_n = int(round(post / dt))
    rel_time = np.arange(-pre_n, post_n + 1, dtype=np.float32) * np.float32(dt)
    n_samples = times.size

    event_times = events["event_time"].to_numpy()
    event_idx = np.searchsorted(times, event_times, side="left")
    event_idx = np.clip(event_idx, 0, n_samples - 1)
    left = np.clip(event_idx - 1, 0, n_samples - 1)
    use_left = np.abs(times[left] - event_times) < np.abs(
        times[event_idx] - event_times
    )
    event_idx = np.where(use_left, left, event_idx)

    windows: list[pl.DataFrame] = []
    event_rows = list(events.iter_rows(named=True))

    for row, idx in zip(event_rows, event_idx, strict=True):
        start = int(idx) - pre_n
        end = int(idx) + post_n + 1  # exclusive

        if start < 0 or end > n_samples:
            if drop_incomplete:
                continue
            start_clamped = max(start, 0)
            end_clamped = min(end, n_samples)
            offset = start_clamped - start
            length = end_clamped - start_clamped
            sl = slice(start_clamped, end_clamped)
            rel = rel_time[offset : offset + length]
        else:
            sl = slice(start, end)
            rel = rel_time

        if sl.start >= sl.stop:
            continue

        data: dict[str, np.ndarray] = {
            "event_id": np.full(rel.shape[0], row["event_id"], dtype=np.int32),
            "Note": np.full(rel.shape[0], row["Note"]),
            "event_time": np.full(
                rel.shape[0], float(times[int(idx)]), dtype=np.float32
            ),
            "rel_time": rel,
            time_col: times[sl],
        }
        for col in value_cols:
            data[col] = value_arrays[col][sl]
        windows.append(pl.DataFrame(data))

    return pl.concat(windows) if windows else pl.DataFrame()


def extract_event_windows(
    df: pl.DataFrame,
    pre: float,
    post: float,
    *,
    by: Sequence[str] = (),
    note: str | Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
    time_col: str = "time",
    note_col: str = "Note",
    events: pl.DataFrame | None = None,
    drop_incomplete: bool = True,
    carry_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Slice peri-event windows around note onsets.

    Windows are built with a shared sample-index grid **within each continuous
    recording** so ``rel_time`` aligns across trials from that recording.
    ``pre`` / ``post`` are converted to sample counts from that recording's
    median sampling interval.

    Parameters
    ----------
    df:
        Photometry dataframe. For a single session this is one continuous
        time series. For multi-animal / multi-session data (e.g.
        ``create_master_df``), pass ``by`` so each recording is handled
        separately — do not treat a stacked master table as one time base.
    pre, post:
        Seconds before / after each event onset to include.
    by:
        Columns that uniquely identify one continuous recording. Empty
        (default) treats all of ``df`` as a single session. For
        ``create_master_df`` output, use e.g. ``("ID", "Extra")``.
        These columns are copied onto every window row.
    note:
        Optional note label(s) to keep. If ``None``, use all notes.
    value_cols:
        Signal columns to keep in each window. Defaults to common signals
        present in ``df`` (``dFF``, ``gcamp``, ``isos``).
    events:
        Optional pre-built events table with at least ``event_time`` and
        ``Note``. When ``by`` is set, ``events`` must also include those
        group columns so each event can be matched to its recording. If
        omitted, events are taken from non-null ``note_col`` rows in ``df``.
    drop_incomplete:
        If True, skip events whose full sample window falls outside the
        recording.
    carry_cols:
        Extra columns to copy from each recording onto its windows (e.g.
        ``("Sex",)``). Values are taken from the first row of the group;
        they should be constant within a recording.

    Returns
    -------
    pl.DataFrame
        Long-format table with one row per sample per event, including
        group keys from ``by``, ``event_id`` (unique within each ``by``
        group), ``Note``, ``event_time``, ``rel_time``, absolute ``time``,
        and the requested value columns.

    Notes
    -----
    ``event_id`` alone is not unique across recordings. Downstream
    z-scoring should use trial keys that include the same ``by`` columns,
    e.g. ``z_score_event_windows(..., by=["ID", "Extra", "event_id"])``.
    """
    if pre < 0 or post < 0:
        raise ValueError("pre and post must be non-negative")

    by = list(by)
    carry_cols = list(carry_cols)
    overlap = set(by) & set(carry_cols)
    if overlap:
        raise ValueError(
            f"carry_cols overlaps with by (already carried): {sorted(overlap)}"
        )

    if value_cols is None:
        candidates = ("dFF", "gcamp", "isos")
        value_cols = [c for c in candidates if c in df.columns]
        if not value_cols:
            raise ValueError(
                "No default value columns found; pass value_cols explicitly "
                f"(available: {df.columns})"
            )
    else:
        value_cols = list(value_cols)

    check_required_cols(df, [time_col, *value_cols, *by, *carry_cols])
    if note_col not in df.columns and events is None:
        raise ValueError(
            f"note column {note_col!r} not in df; pass events= explicitly "
            "or include notes in the session dataframe"
        )

    if events is not None:
        events = _prepare_events_table(events, by=by, note=note)

    if not by:
        if events is None:
            events = get_session_events(
                df, time_col=time_col, note_col=note_col, note=note
            )
        return _extract_event_windows_one(
            df,
            events,
            pre,
            post,
            value_cols=value_cols,
            time_col=time_col,
            drop_incomplete=drop_incomplete,
        )

    # Multi-recording: extract within each continuous group, then stack.
    if events is None:
        events = get_session_events(
            df, by=by, time_col=time_col, note_col=note_col, note=note
        )

    parts: list[pl.DataFrame] = []
    for group in df.partition_by(by, maintain_order=True):
        group_keys = {c: group[c][0] for c in by}
        label = ", ".join(f"{k}={v!r}" for k, v in group_keys.items())

        group_events = events
        for c, v in group_keys.items():
            group_events = group_events.filter(pl.col(c) == v)
        # Drop by-cols from the per-group events table; they are re-attached
        # from group_keys so we don't duplicate after concat.
        event_cols = [
            c for c in group_events.columns if c not in by
        ]
        group_events = group_events.select(event_cols)

        windows = _extract_event_windows_one(
            group,
            group_events,
            pre,
            post,
            value_cols=value_cols,
            time_col=time_col,
            drop_incomplete=drop_incomplete,
            group_label=label,
        )
        if windows.is_empty():
            continue

        attach = {**group_keys, **{c: group[c][0] for c in carry_cols}}
        windows = windows.with_columns(
            [pl.lit(v).alias(k) for k, v in attach.items()]
        )
        # Prefer group keys / metadata near the front for readability.
        front = [*by, *carry_cols]
        other = [c for c in windows.columns if c not in front]
        parts.append(windows.select([*front, *other]))

    return pl.concat(parts) if parts else pl.DataFrame()


def average_event_windows(
    windows: pl.DataFrame,
    value_cols: Sequence[str] | None = None,
    *,
    by: Sequence[str] = ("Note",),
    rel_time_col: str = "rel_time",
) -> pl.DataFrame:
    """Average peri-event windows across trials at each relative time.

    This is the **pooling** step: ``by`` chooses which trials are combined into
    the same mean trace (plus ``rel_time``). It is *not* the trial identity used
    for per-window z-scoring — see ``z_score_event_windows(..., by=...)``.

    Examples of ``by``:

    - ``("Note",)`` — grand mean over all trials of each note type (default)
    - ``("Note", "ID")`` — per-animal mean for each note type
    - ``("Note", "ID", "session")`` — per-session means before a later animal avg

    Returns mean, SD, SEM, and ``n_trials`` for each value column. Windows from
    ``extract_event_windows`` share a common ``rel_time`` grid within a session
    so samples stack cleanly; when pooling across sessions, ensure sampling
    rates (and thus ``rel_time`` grids) match or resample first.
    """
    if windows.is_empty():
        return windows

    if value_cols is None:
        candidates = ("dFF", "gcamp", "isos")
        value_cols = [c for c in candidates if c in windows.columns]
        if not value_cols:
            raise ValueError(
                "No default value columns found; pass value_cols explicitly "
                f"(available: {windows.columns})"
            )
    else:
        value_cols = list(value_cols)

    by = list(by)
    if not by:
        raise ValueError("by must contain at least one pooling column")

    group_cols = [*by, rel_time_col]
    check_required_cols(windows, [*group_cols, *value_cols])

    aggs: list[pl.Expr] = []
    for col in value_cols:
        aggs.extend(
            [
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
                (pl.col(col).std() / pl.col(col).count().sqrt()).alias(f"{col}_sem"),
            ]
        )
    aggs.append(pl.len().alias("n_trials"))

    return windows.group_by(group_cols).agg(aggs).sort(group_cols)


def z_score_event_windows(
    windows: pl.DataFrame,
    baseline: tuple[float, float],
    value_cols: Sequence[str],
    *,
    by: Sequence[str] = ("event_id",),
    rel_time_col: str = "rel_time",
    suffix: str = "_z",
) -> pl.DataFrame:
    """Z-score each event window using that trial's baseline mean and SD.

    This is the **per-trial** step: ``by`` must uniquely identify one peri-event
    window (one baseline µ/σ). That is different from ``average_event_windows``
    ``by``, which chooses how trials are pooled into a mean trace.

    For every group in ``by`` and each value column, computes:

        z = (x - mu_baseline) / sigma_baseline

    where the baseline is samples with ``rel_time`` in ``baseline``
    ``[start, end]`` (inclusive).

    Parameters
    ----------
    windows:
        Long-format peri-event table (e.g. from ``extract_event_windows``).
    baseline:
        Inclusive ``(start, end)`` in ``rel_time`` seconds used for µ and σ.
    value_cols:
        Signal columns to z-score (e.g. ``["dFF"]``).
    by:
        Columns that uniquely identify a trial. Default ``("event_id",)`` is
        enough for a single session. For multi-animal / multi-session tables,
        include animal and session keys so IDs do not collide, e.g.
        ``("ID", "session", "event_id")`` or a global ``("trial_id",)``.
    rel_time_col:
        Relative-time column (seconds from event onset).
    suffix:
        Appended to each value column name for the z-scored output
        (default ``dFF`` → ``dFF_z``).

    Returns
    -------
    pl.DataFrame
        Copy of ``windows`` with extra ``{col}{suffix}`` columns. Original
        columns are unchanged so you can average raw or z-scored traces with
        ``average_event_windows``.

    Notes
    -----
    Trials with zero or undefined baseline SD get null z-scores for that column.
    """
    if windows.is_empty():
        raise ValueError("dataframe is empty")

    value_cols = list(value_cols)
    by = list(by)
    if not by:
        raise ValueError(
            "by must contain at least one trial-identity column "
            "(e.g. ['event_id'] or ['ID', 'session', 'event_id'])"
        )
    if not value_cols:
        raise ValueError("value_cols must contain at least one signal column")

    check_required_cols(windows, [*by, rel_time_col, *value_cols])

    lo, hi = baseline
    if lo > hi:
        raise ValueError(f"baseline start must be <= end (got {baseline})")
    baseline_mask = pl.col(rel_time_col).is_between(lo, hi, closed="both")

    baseline_df = windows.filter(baseline_mask)
    if baseline_df.is_empty():
        raise ValueError(
            "no samples fall in the baseline window; check baseline bounds "
            f"and {rel_time_col} values"
        )

    stats_aggs: list[pl.Expr] = []
    for col in value_cols:
        stats_aggs.extend(
            [
                pl.col(col).mean().alias(f"__{col}_bl_mean"),
                pl.col(col).std().alias(f"__{col}_bl_std"),
            ]
        )
    baseline_stats = baseline_df.group_by(by).agg(stats_aggs)
    z_windows = windows.join(baseline_stats, on=by, how="left")

    z_exprs: list[pl.Expr] = []
    drop_cols: list[str] = []
    for col in value_cols:
        mean_col = f"__{col}_bl_mean"
        std_col = f"__{col}_bl_std"
        drop_cols.extend([mean_col, std_col])
        z_exprs.append(
            pl.when(pl.col(std_col).is_null() | (pl.col(std_col) == 0))
            .then(None)
            .otherwise((pl.col(col) - pl.col(mean_col)) / pl.col(std_col))
            .cast(pl.Float32)
            .alias(f"{col}{suffix}")
        )

    return z_windows.with_columns(z_exprs).drop(drop_cols)
