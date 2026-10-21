from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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


def _tdt_store_ids(*names: str) -> list[str]:
    """Map Python stream names to ids accepted by ``tdt.read_block(store=...)``.

    ``read_block`` filters on the raw 4-character store id. ``tdt.fix_var_name``
    inserts a leading underscore when that id starts with a digit, so callers
    pass ``"_405A"`` for a store named ``405A``. Both forms are returned.
    """
    store_ids: list[str] = []
    for name in names:
        if name not in store_ids:
            store_ids.append(name)
        if len(name) == 5 and name.startswith("_") and name[1].isdigit():
            raw_id = name[1:]
            if raw_id not in store_ids:
                store_ids.append(raw_id)
    return store_ids


# Synapse records these digital inputs with a trailing slash. ``tdt`` then
# exposes them as ``PC0_`` and ``PC1_`` because ``/`` is not a valid attribute.
_EPOC_RAW_ALIASES = {"PC0_": "PC0/", "PC1_": "PC1/"}
_EVENT_CANDIDATES = ("Note", "PC0/", "PC1/")


def _resolve_epoc_store(name: str) -> tuple[str, str]:
    """Return ``(raw_store_id, python_attribute_name)`` for an epoc name."""
    raw = _EPOC_RAW_ALIASES.get(name, name)
    if len(raw) == 5 and raw.startswith("_") and raw[1].isdigit():
        raw = raw[1:]
    return raw, tdt.fix_var_name(raw)


def _event_table_from_epoc(
    epoc: tdt.StructType,
    store_name: str,
    label: str | None,
) -> pl.DataFrame | None:
    """Build the sparse ``Note`` / ``Onset`` / ``Offset`` table for one epoc.

    A non-empty ``label`` is used for every event. Otherwise the epoc's own
    note text is kept when it has one entry per onset. A port epoc has no
    text, so the store name is used instead.
    """
    onset = np.atleast_1d(np.asarray(getattr(epoc, "onset", []), dtype=float))
    if onset.size == 0:
        return None

    offset = getattr(epoc, "offset", None)
    if offset is None:
        offset_values = np.full(onset.shape, np.nan)
    else:
        offset_values = np.atleast_1d(np.asarray(offset, dtype=float))
        if offset_values.shape[0] != onset.shape[0]:
            offset_values = np.full(onset.shape, np.nan)

    text = (label or "").strip()
    if text:
        notes = np.full(onset.shape, text)
    else:
        raw_notes = getattr(epoc, "notes", None)
        if raw_notes is None:
            notes = np.full(onset.shape, store_name)
        else:
            raw_notes = np.atleast_1d(np.asarray(raw_notes))
            if raw_notes.shape[0] == onset.shape[0]:
                notes = raw_notes.astype(str)
            else:
                notes = np.full(onset.shape, store_name)

    return pl.DataFrame(
        {"Note": notes, "Onset": onset, "Offset": offset_values}
    ).sort("Onset")


def create_session_df(
    file: Path,
    metadata_fields: list[str],
    isos: str,
    gcamp: str,
    event_store: str | None = None,
    event_label: str | None = None,
) -> pl.DataFrame:
    """Load one TDT block into a tidy photometry session dataframe.

    Reads the isosbestic stream, the GCaMP stream, and one epoc (other tank
    stores are skipped), builds a ``time`` column, attaches filename metadata,
    and left-joins that epoc's events onto the nearest sample via
    ``join_asof``. Metadata columns (``metadata_fields`` plus ``Extra``) are
    stored as categoricals: one code per row, with the label kept once.
    Events are sparse: only rows near onsets have non-null ``Note`` /
    ``Onset`` / ``Offset``.

    ``event_store`` selects the epoc. ``None`` reads ``Note``. ``PC0_`` and
    ``PC0/`` are the same digital-input store, and likewise for ``PC1_``.
    ``event_label``, when set, replaces the event text. A port epoc has no
    note text of its own, so with no label the store name is used.

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
    event_store:
        Epoc to join as the event column. ``None`` uses ``Note``.
    event_label:
        Text written into ``Note`` for every event from ``event_store``.
        ``None`` keeps the epoc's own note text, or the store name when the
        epoc has none.

    Returns
    -------
    pl.DataFrame
        Sorted by ``time``, with categorical metadata columns, ``isos``,
        ``gcamp``, ``time``, ``duration``, and note columns (``Note``,
        ``Onset``, ``Offset``; null-filled if the chosen epoc has no events).
    """
    raw_store, epoc_attr = _resolve_epoc_store(event_store or "Note")
    data = tdt.read_block(file, store=_tdt_store_ids(isos, gcamp, raw_store))
    data.streams.time = create_time_stream(data, isos)

    metadata = get_file_metadata(file, metadata_fields)

    data_df = (
        pl.DataFrame(
            {
                "isos": data.streams[isos].data,
                "gcamp": data.streams[gcamp].data,
                "time": data.streams.time.data,
            }
        )
        .with_columns(
            pl.duration(seconds=pl.col("time")).alias("duration"),
            *[
                pl.lit(value).cast(pl.Categorical).alias(name)
                for name, value in metadata.items()
            ],
        )
        .select(*metadata.keys(), "isos", "gcamp", "time", "duration")
        .sort("time")
    )

    epoc = data.epocs[epoc_attr] if epoc_attr in data.epocs.keys() else None
    events = (
        None
        if epoc is None
        else _event_table_from_epoc(epoc, event_store or "Note", event_label)
    )
    if events is None:
        final_df = data_df.with_columns(
            pl.lit(None).cast(pl.String).alias("Note"),
            pl.lit(None).cast(pl.Float64).alias("Onset"),
            pl.lit(None).cast(pl.Float64).alias("Offset"),
        )
    else:
        events = events.join_asof(
            data_df.select("time"),
            left_on="Onset",
            right_on="time",
            strategy="nearest",
        )
        final_df = data_df.join(events, on="time", how="left")

    return final_df


_TRACE_COLS = ("isos", "gcamp", "time", "duration", "Note", "Onset", "Offset")


def _downsample_to_sample_rate(df: pl.DataFrame, sample_rate: float) -> pl.DataFrame:
    """Average recordings into bins of width ``1 / sample_rate`` seconds.

    ``isos`` and ``gcamp`` become the mean of the samples in each bin, and
    ``time`` becomes the mean of those sample times. ``duration`` is rebuilt
    from that time. Each note is moved onto the nearest output sample. A
    target at the recording rate returns ``df`` unchanged. A higher target
    raises ``ValueError``. So does a target coarse enough that two notes
    would share one output sample, which would duplicate rows in the trace.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be a positive number of Hz, got {sample_rate}"
        )
    if df.height < 2:
        return df

    dt = df.select(pl.col("time").diff().drop_nulls().median()).item()
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"could not estimate a positive sampling interval (got {dt})"
        )
    native = 1.0 / float(dt)
    if sample_rate > native * (1 + 1e-3):
        raise ValueError(
            f"sample_rate ({sample_rate:g} Hz) is above the recording rate "
            f"({native:.4g} Hz)"
        )
    if sample_rate >= native * (1 - 1e-3):
        return df

    group_cols = [col for col in df.columns if col not in _TRACE_COLS]
    df = df.sort([*group_cols, "time"] if group_cols else ["time"])
    # Bin by sample index within each recording. ``time`` is float32, so
    # flooring ``time * sample_rate`` puts edge samples in the wrong bin.
    index = pl.int_range(pl.len())
    span = pl.col("time").max() - pl.col("time").min()
    count = pl.len()
    if group_cols:
        index = index.over(group_cols)
        span = span.over(group_cols)
        count = count.over(group_cols)
    binned = (
        df.with_columns(
            _bin=(index.cast(pl.Float64) * sample_rate / ((count - 1) / span))
            .floor()
            .cast(pl.Int64)
        )
        .group_by(*group_cols, "_bin", maintain_order=True)
        .agg(
            pl.col("isos").mean().cast(pl.Float32),
            pl.col("gcamp").mean().cast(pl.Float32),
            pl.col("time").mean().cast(pl.Float32),
        )
        .drop("_bin")
        .with_columns(pl.duration(seconds=pl.col("time")).alias("duration"))
        .select(*group_cols, "isos", "gcamp", "time", "duration")
    )

    notes = df.filter(pl.col("Note").is_not_null()).select(
        *group_cols, "Note", "Onset", "Offset"
    )
    if notes.is_empty():
        return binned.with_columns(
            pl.lit(None).cast(pl.String).alias("Note"),
            pl.lit(None).cast(pl.Float64).alias("Onset"),
            pl.lit(None).cast(pl.Float64).alias("Offset"),
        )

    sort_keys = [*group_cols, "time"]
    binned = binned.sort(sort_keys)
    asof_by = {"by": group_cols} if group_cols else {}
    snapped = notes.sort([*group_cols, "Onset"]).join_asof(
        binned.select(*sort_keys),
        left_on="Onset",
        right_on="time",
        strategy="nearest",
        check_sortedness=False,
        **asof_by,
    )
    clash = (
        snapped.group_by(sort_keys)
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if clash.height:
        raise ValueError(
            "sample_rate is too coarse to keep every note on its own sample "
            f"({clash.height} output sample(s) are nearest for more than one note)"
        )
    return binned.join(snapped, on=sort_keys, how="left")


def _epoc_event_counts(path: Path, candidates: Sequence[str]) -> dict[str, int]:
    """Count onsets for each candidate epoc without reading photometry streams."""
    raw_ids = [_resolve_epoc_store(name)[0] for name in candidates]
    data = tdt.read_block(path, store=raw_ids)
    counts: dict[str, int] = {}
    for name in candidates:
        attr = _resolve_epoc_store(name)[1]
        if attr not in data.epocs.keys():
            counts[name] = 0
            continue
        onset = np.atleast_1d(np.asarray(data.epocs[attr].onset, dtype=float))
        counts[name] = int(onset.size)
    return counts


def _choose_event_store(counts: dict[str, int], previous: str) -> str:
    """Keep a store that was already chosen. Otherwise fill in an unambiguous one."""
    if previous.strip():
        return previous.strip()
    present = [name for name, count in counts.items() if count > 0]
    if len(present) == 1:
        return present[0]
    return ""


def _merge_event_table(
    scanned: list[dict[str, object]],
    existing: pl.DataFrame | None,
    candidates: Sequence[str],
) -> pl.DataFrame:
    """Refresh counts, keep prior ``store`` and ``label`` choices, append new files."""
    previous: dict[str, tuple[str, str]] = {}
    leftovers: list[dict[str, object]] = []
    if existing is not None:
        seen: set[str] = set()
        for row in existing.iter_rows(named=True):
            stem = Path(str(row["file"])).stem
            if stem in seen:
                raise ValueError(f"events table lists {stem!r} more than once")
            seen.add(stem)
            previous[stem] = (
                "" if row["store"] is None else str(row["store"]).strip(),
                "" if row["label"] is None else str(row["label"]).strip(),
            )
        scanned_stems = {str(row["file"]) for row in scanned}
        for row in existing.iter_rows(named=True):
            stem = Path(str(row["file"])).stem
            if stem not in scanned_stems:
                kept_counts = {
                    name: 0 if row[name] is None else int(row[name])
                    for name in candidates
                }
                leftovers.append(
                    {
                        "file": stem,
                        **kept_counts,
                        "store": previous[stem][0],
                        "label": previous[stem][1],
                    }
                )

    rows: list[dict[str, object]] = []
    for row in scanned:
        stem = str(row["file"])
        counts = {name: int(row[name]) for name in candidates}
        old_store, old_label = previous.get(stem, ("", ""))
        rows.append(
            {
                "file": stem,
                **counts,
                "store": _choose_event_store(counts, old_store),
                "label": old_label,
            }
        )
    rows.extend(leftovers)
    return pl.DataFrame(
        rows,
        schema={
            "file": pl.String,
            **{name: pl.Int64 for name in candidates},
            "store": pl.String,
            "label": pl.String,
        },
    )


def write_events_csv(
    data_files: list[Path],
    path: str | Path,
    candidates: Sequence[str] = _EVENT_CANDIDATES,
) -> pl.DataFrame:
    """Write an event-source sheet for ``create_master_df(..., events=path)``.

    Each row is one block folder. The count columns are how many onsets
    ``Note``, ``PC0/``, and ``PC1/`` contain. ``store`` is filled in when
    exactly one of those counts is non-zero. It is left blank when none or
    several have events, which is the case to edit by hand. ``label`` is
    optional text used for a port's events in place of the store name.

    Running this again refreshes the counts and adds new folders. A ``store``
    or ``label`` that already has text is left as written. Clear ``store``
    before re-running if that row should be chosen automatically again.
    Rows for folders not in ``data_files`` are kept.

    Parameters
    ----------
    data_files:
        TDT block directories to scan.
    path:
        CSV path to write. An existing file at this path is updated.
    candidates:
        Epoc names to count. The default is ``Note``, ``PC0/``, and ``PC1/``.
        ``PC0_`` and ``PC1_`` name the same stores.

    Returns
    -------
    pl.DataFrame
        The table that was written.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates must name at least one epoc")
    stems = [file.stem for file in data_files]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"data_files has repeated folder names: {duplicates}")

    destination = Path(path)
    existing = None
    if destination.exists():
        existing = pl.read_csv(
            destination,
            schema_overrides={
                "file": pl.String,
                "store": pl.String,
                "label": pl.String,
                **{name: pl.Int64 for name in candidates},
            },
        )
        missing = {"file", "store", "label", *candidates} - set(existing.columns)
        if missing:
            raise ValueError(
                f"{destination} is missing columns: {sorted(missing)}"
            )

    scanned: list[dict[str, object]] = []
    for file in data_files:
        counts = _epoc_event_counts(file, candidates)
        scanned.append({"file": file.stem, **counts})

    table = _merge_event_table(scanned, existing, candidates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination)
    return table


def _load_events_table(
    path: str | Path,
    data_files: list[Path],
) -> dict[str, tuple[str | None, str | None]]:
    """Map a block folder name to ``(event_store, event_label)`` from a CSV."""
    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(f"events table not found: {destination}")
    table = pl.read_csv(
        destination,
        schema_overrides={"file": pl.String, "store": pl.String, "label": pl.String},
    )
    missing = {"file", "store", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"{destination} is missing columns: {sorted(missing)}")

    chosen: dict[str, tuple[str | None, str | None]] = {}
    for row in table.iter_rows(named=True):
        stem = Path(str(row["file"])).stem
        if stem in chosen:
            raise ValueError(f"events table lists {stem!r} more than once")
        store = "" if row["store"] is None else str(row["store"]).strip()
        label = "" if row["label"] is None else str(row["label"]).strip()
        chosen[stem] = (store or None, label or None)

    file_stems = [file.stem for file in data_files]
    repeated = sorted({stem for stem in file_stems if file_stems.count(stem) > 1})
    if repeated:
        raise ValueError(f"data_files has repeated folder names: {repeated}")
    absent = [stem for stem in file_stems if stem not in chosen]
    if absent:
        raise ValueError(
            f"events table {destination} has no row for: {absent}"
        )
    return chosen


def _load_session_frame(
    file: Path,
    metadata_fields: list[str],
    isos: str,
    gcamp: str,
    experiment_length: pl.Expr | None,
    sample_rate: float | None,
    debug: bool,
    event_store: str | None = None,
    event_label: str | None = None,
) -> pl.DataFrame:
    """Load one block, then apply the length filter and optional downsample."""
    if debug:
        print(file.name)

    frame = create_session_df(
        file,
        metadata_fields,
        isos=isos,
        gcamp=gcamp,
        event_store=event_store,
        event_label=event_label,
    )

    if experiment_length is not None:
        frame = frame.filter(pl.col("duration") <= experiment_length)

    if sample_rate is not None:
        try:
            frame = _downsample_to_sample_rate(frame, sample_rate)
        except ValueError as exc:
            raise ValueError(f"{file.name}: {exc}") from exc

    if debug:
        print(f"Shape: {frame.shape}")
        print(f"Columns: {frame.columns}")

    return frame


def create_master_df(
    data_files: list[Path],
    metadata_fields: list[str],
    isos: str,
    gcamp: str,
    experiment_length: pl.Expr | None = None,
    sample_rate: float | None = None,
    n_workers: int = 1,
    debug: bool = False,
    events: str | Path | None = None,
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
    sample_rate:
        Optional target sample rate in Hz. After the length filter, each
        session is averaged into bins of width ``1 / sample_rate``. ``time``
        is the mean time of the samples in the bin, and each note is snapped
        to the nearest output sample. Omit to keep the acquisition rate.
    n_workers:
        How many sessions to load at once. ``1`` (default) reads one file
        at a time. Above 1, a thread pool reads several tanks at once and
        the result stays in ``data_files`` order. Threads fit this work
        because the tank read is mostly file I/O. ``debug=True`` prints
        from different files can interleave.
    debug:
        If True, print each file name and its session shape while loading.
    events:
        Optional CSV from ``write_events_csv``. Each block folder is matched
        on its name. A blank ``store`` reads ``Note``. A filled ``store``
        (``Note``, ``PC0/``, ``PC1_``, and so on) is the epoc joined for that
        file, and a filled ``label`` replaces that epoc's event text. Omit
        to read ``Note`` for every file.

    Returns
    -------
    pl.DataFrame
        Vertical concat of all session dataframes (same schema as
        ``create_session_df``).
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")

    event_map = (
        None if events is None else _load_events_table(events, data_files)
    )

    def load(file: Path) -> pl.DataFrame:
        event_store, event_label = (
            (None, None) if event_map is None else event_map[file.stem]
        )
        return _load_session_frame(
            file,
            metadata_fields,
            isos,
            gcamp,
            experiment_length,
            sample_rate,
            debug,
            event_store,
            event_label,
        )

    if n_workers == 1 or len(data_files) < 2:
        df_list = [load(file) for file in data_files]
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(data_files))) as pool:
            df_list = list(pool.map(load, data_files))

    master_df = pl.concat(df_list, how="vertical")

    return master_df


def isosbestic_fit(isos: np.ndarray, gcamp: np.ndarray) -> np.ndarray:
    """Linear isosbestic estimate of GCaMP: ``Y_fit = a * isos + b``.

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
        1-D fitted GCaMP estimate, same length as the inputs.
    """
    isos = np.asarray(isos)
    gcamp = np.asarray(gcamp)
    if isos.shape != gcamp.shape:
        raise ValueError(
            "isos and gcamp must have the same shape "
            f"(got {isos.shape} and {gcamp.shape})"
        )
    if isos.size < 2:
        raise ValueError("isos and gcamp must contain at least two samples")
    bls = np.polyfit(isos, gcamp, 1)
    return np.polyval(bls, isos)


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
    y_fit = isosbestic_fit(isos, gcamp)
    dFF = (100 * (gcamp - y_fit) / y_fit).astype(np.float32)

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
        return events.with_columns(
            pl.int_range(0, pl.len()).over(by).cast(pl.UInt32).alias("event_id")
        )
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
        event_cols = [c for c in group_events.columns if c not in by]
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
        windows = windows.with_columns([pl.lit(v).alias(k) for k, v in attach.items()])
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
