import numpy as np
import polars as pl
import pytest

from gcamper.core import _downsample_to_sample_rate


def _session(
    n: int = 100,
    fs: float = 100.0,
    animal: str = "A",
    gcamp: np.ndarray | None = None,
    notes: list[tuple[float, str]] | None = None,
) -> pl.DataFrame:
    time = np.arange(n, dtype=np.float64) / fs
    signal = np.arange(n, dtype=np.float32) if gcamp is None else gcamp
    df = pl.DataFrame(
        {
            "ID": pl.Series([animal] * n, dtype=pl.Categorical),
            "Sex": pl.Series(["M"] * n, dtype=pl.Categorical),
            "Extra": pl.Series(["s1"] * n, dtype=pl.Categorical),
            "isos": np.full(n, 2.0, dtype=np.float32),
            "gcamp": signal.astype(np.float32),
            "time": time.astype(np.float32),
        }
    ).with_columns(
        pl.duration(seconds=pl.col("time")).alias("duration"),
        pl.lit(None).cast(pl.String).alias("Note"),
        pl.lit(None).cast(pl.Float64).alias("Onset"),
        pl.lit(None).cast(pl.Float64).alias("Offset"),
    )
    if not notes:
        return df

    note_rows = pl.DataFrame(
        {
            "Onset": [onset for onset, _ in notes],
            "Note": [label for _, label in notes],
            "Offset": [onset + 0.1 for onset, _ in notes],
        }
    ).sort("Onset")
    snapped = note_rows.join_asof(
        df.select("time"),
        left_on="Onset",
        right_on="time",
        strategy="nearest",
    )
    return df.update(snapped, on="time")


def test_downsample_averages_bins_and_sets_output_rate():
    out = _downsample_to_sample_rate(_session(), sample_rate=10)
    assert out.height == 10
    assert out.schema["ID"] == pl.Categorical
    assert out.schema["isos"] == pl.Float32
    assert out.schema["gcamp"] == pl.Float32
    assert out["isos"].to_numpy() == pytest.approx(2.0)
    # Bins of samples 0..9 and 10..19.
    assert out["gcamp"].to_numpy()[:2] == pytest.approx([4.5, 14.5])
    dt = np.diff(out["time"].to_numpy())
    assert dt == pytest.approx(0.1)
    assert out["Note"].null_count() == out.height


def test_downsample_keeps_notes_on_the_nearest_sample():
    out = _downsample_to_sample_rate(
        _session(notes=[(0.2, "food"), (0.81, "cue")]),
        sample_rate=10,
    )
    kept = out.filter(pl.col("Note").is_not_null()).select("Note", "Onset", "time")
    assert kept["Note"].to_list() == ["food", "cue"]
    assert kept["Onset"].to_list() == pytest.approx([0.2, 0.81])
    assert out.height == 10


def test_downsample_does_not_mix_recordings():
    sessions = pl.concat(
        [
            _session(animal="A", gcamp=np.zeros(100)),
            _session(animal="B", gcamp=np.full(100, 10.0), notes=[(0.42, "food")]),
        ]
    )
    out = _downsample_to_sample_rate(sessions, sample_rate=10)
    assert out.filter(pl.col("ID") == "A")["gcamp"].mean() == pytest.approx(0.0)
    assert out.filter(pl.col("ID") == "B")["gcamp"].mean() == pytest.approx(10.0)
    notes = out.filter(pl.col("Note").is_not_null())
    assert notes["ID"].to_list() == ["B"]
    assert notes["Note"].to_list() == ["food"]


def test_downsample_matching_the_recording_rate_is_unchanged():
    source = _session(notes=[(0.25, "food")])
    out = _downsample_to_sample_rate(source, sample_rate=100)
    assert out.height == source.height
    assert out["gcamp"].to_numpy()[0] == pytest.approx(0.0)
    assert out.filter(pl.col("Note").is_not_null())["Onset"].item() == pytest.approx(
        0.25
    )


def test_downsample_rejects_a_rate_above_the_recording():
    with pytest.raises(ValueError, match="above the recording rate"):
        _downsample_to_sample_rate(_session(), sample_rate=200)


def test_downsample_rejects_a_non_positive_rate():
    with pytest.raises(ValueError, match="positive"):
        _downsample_to_sample_rate(_session(), sample_rate=0)


def test_downsample_rejects_notes_that_would_share_a_sample():
    source = _session(notes=[(0.10, "food"), (0.12, "cue")])
    with pytest.raises(ValueError, match="too coarse"):
        _downsample_to_sample_rate(source, sample_rate=1)
