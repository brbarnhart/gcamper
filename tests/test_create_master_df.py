from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gcamper import create_master_df


def _fake_session(
    file: Path,
    metadata_fields,
    isos,
    gcamp,
    event_store=None,
    event_label=None,
) -> pl.DataFrame:
    del metadata_fields, isos, gcamp, event_store, event_label
    animal = file.name
    height = 20
    time = (np.arange(height, dtype=np.float64) / 100.0).astype(np.float32)
    return pl.DataFrame(
        {
            "ID": pl.Series([animal] * height, dtype=pl.Categorical),
            "isos": np.full(height, float(animal), dtype=np.float32),
            "gcamp": np.arange(height, dtype=np.float32),
            "time": time,
        }
    ).with_columns(
        pl.duration(seconds=pl.col("time")).alias("duration"),
        pl.lit(None).cast(pl.String).alias("Note"),
        pl.lit(None).cast(pl.Float64).alias("Onset"),
        pl.lit(None).cast(pl.Float64).alias("Offset"),
    )


def test_parallel_load_matches_serial_order(monkeypatch):
    monkeypatch.setattr("gcamper.core.create_session_df", _fake_session)
    files = [Path("2"), Path("0"), Path("1")]
    serial = create_master_df(files, ["ID"], "_405A", "_465A")
    parallel = create_master_df(files, ["ID"], "_405A", "_465A", n_workers=3)
    assert parallel["ID"].cast(pl.String).to_list() == (
        ["2"] * 20 + ["0"] * 20 + ["1"] * 20
    )
    assert parallel.equals(serial)


def test_parallel_downsample_matches_serial(monkeypatch):
    monkeypatch.setattr("gcamper.core.create_session_df", _fake_session)
    files = [Path("3"), Path("1")]
    kwargs = {"sample_rate": 10, "n_workers": 2}
    parallel = create_master_df(files, ["ID"], "_405A", "_465A", **kwargs)
    serial = create_master_df(files, ["ID"], "_405A", "_465A", sample_rate=10)
    assert parallel.height == 4
    assert parallel.equals(serial)


def test_n_workers_must_be_positive():
    with pytest.raises(ValueError, match="n_workers"):
        create_master_df([], ["ID"], "_405A", "_465A", n_workers=0)
