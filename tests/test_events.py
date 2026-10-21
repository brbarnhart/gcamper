from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from gcamper import create_master_df, write_events_csv
from gcamper.core import (
    _event_table_from_epoc,
    _merge_event_table,
    _resolve_epoc_store,
)


def test_resolve_epoc_store_accepts_slash_and_underscore_names():
    assert _resolve_epoc_store("Note") == ("Note", "Note")
    assert _resolve_epoc_store("PC0/") == ("PC0/", "PC0_")
    assert _resolve_epoc_store("PC0_") == ("PC0/", "PC0_")
    assert _resolve_epoc_store("PC1_") == ("PC1/", "PC1_")
    assert _resolve_epoc_store("_405A") == ("405A", "_405A")


def test_event_table_uses_note_text_label_or_store_name():
    noted = SimpleNamespace(onset=[1.0, 2.0], offset=[1.5, 2.5], notes=["food", "cue"])
    table = _event_table_from_epoc(noted, "Note", None)
    assert table is not None
    assert table["Note"].to_list() == ["food", "cue"]
    assert table["Onset"].to_list() == [1.0, 2.0]

    port = SimpleNamespace(onset=[3.0], offset=[3.2])
    assert _event_table_from_epoc(port, "PC0/", None)["Note"].to_list() == ["PC0/"]
    assert _event_table_from_epoc(port, "PC0/", "lick")["Note"].to_list() == ["lick"]
    labeled = _event_table_from_epoc(noted, "Note", "tone")
    assert labeled is not None
    assert labeled["Note"].to_list() == ["tone", "tone"]
    assert _event_table_from_epoc(SimpleNamespace(onset=[]), "Note", None) is None


def test_merge_event_table_prefills_only_unambiguous_stores_and_keeps_edits():
    candidates = ("Note", "PC0/", "PC1/")
    scanned = [
        {"file": "a", "Note": 1, "PC0/": 0, "PC1/": 0},
        {"file": "b", "Note": 2, "PC0/": 4, "PC1/": 0},
        {"file": "c", "Note": 0, "PC0/": 3, "PC1/": 0},
    ]
    first = _merge_event_table(scanned, None, candidates)
    assert first["store"].to_list() == ["Note", "", "PC0/"]
    assert first["label"].to_list() == ["", "", ""]

    edited = first.with_columns(
        pl.when(pl.col("file") == "b")
        .then(pl.lit("PC1/"))
        .otherwise(pl.col("store"))
        .alias("store"),
        pl.when(pl.col("file") == "b")
        .then(pl.lit("cue"))
        .otherwise(pl.col("label"))
        .alias("label"),
    )
    edited = pl.concat(
        [
            edited,
            pl.DataFrame(
                {
                    "file": ["d"],
                    "Note": [5],
                    "PC0/": [0],
                    "PC1/": [0],
                    "store": ["Note"],
                    "label": [""],
                }
            ),
        ]
    )
    rescanned = [
        {"file": "a", "Note": 0, "PC0/": 1, "PC1/": 1},
        {"file": "b", "Note": 9, "PC0/": 9, "PC1/": 9},
        {"file": "c", "Note": 0, "PC0/": 8, "PC1/": 0},
    ]
    second = _merge_event_table(rescanned, edited, candidates)
    by_file = {row["file"]: row for row in second.iter_rows(named=True)}
    assert by_file["a"]["store"] == "Note"
    assert by_file["a"]["PC0/"] == 1
    assert by_file["b"]["store"] == "PC1/"
    assert by_file["b"]["label"] == "cue"
    assert by_file["c"]["store"] == "PC0/"
    assert by_file["d"]["store"] == "Note"
    assert by_file["d"]["Note"] == 5


def _fake_session(file, metadata_fields, isos, gcamp, event_store=None, event_label=None):
    del metadata_fields, isos, gcamp
    return pl.DataFrame(
        {
            "ID": [file.name],
            "store": ["" if event_store is None else event_store],
            "label": ["" if event_label is None else event_label],
            "isos": [1.0],
            "gcamp": [1.0],
            "time": [0.0],
            "duration": [0],
            "Note": [None],
            "Onset": [None],
            "Offset": [None],
        }
    )


def test_create_master_df_reads_store_and_label_per_file(monkeypatch, tmp_path):
    monkeypatch.setattr("gcamper.core.create_session_df", _fake_session)
    sheet = tmp_path / "events.csv"
    pl.DataFrame(
        {
            "file": ["mouse_a", "mouse_b", "mouse_c"],
            "Note": [1, 0, 4],
            "PC0/": [0, 3, 2],
            "PC1/": [0, 0, 0],
            "store": ["", "PC0/", "PC1_"],
            "label": ["", "lick", ""],
        }
    ).write_csv(sheet)
    files = [Path("mouse_b"), Path("mouse_a"), Path("mouse_c")]
    out = create_master_df(files, ["ID"], "_405A", "_465A", events=sheet)
    assert out["ID"].to_list() == ["mouse_b", "mouse_a", "mouse_c"]
    assert out["store"].to_list() == ["PC0/", "", "PC1_"]
    assert out["label"].to_list() == ["lick", "", ""]


def test_create_master_df_requires_a_row_for_every_file(tmp_path):
    sheet = tmp_path / "events.csv"
    pl.DataFrame(
        {"file": ["mouse_a"], "store": ["Note"], "label": [""]}
    ).write_csv(sheet)
    with pytest.raises(ValueError, match="mouse_b"):
        create_master_df(
            [Path("mouse_b")],
            ["ID"],
            "_405A",
            "_465A",
            events=sheet,
        )


def test_write_events_csv_preserves_manual_choices(monkeypatch, tmp_path):
    counts = {
        "one": {"Note": 2, "PC0/": 0, "PC1/": 0},
        "two": {"Note": 1, "PC0/": 4, "PC1/": 0},
    }

    def fake_counts(path, candidates):
        del candidates
        return dict(counts[path.stem])

    monkeypatch.setattr("gcamper.core._epoc_event_counts", fake_counts)
    destination = tmp_path / "events.csv"
    files = [Path("one"), Path("two")]
    first = write_events_csv(files, destination)
    assert first["store"].to_list() == ["Note", ""]

    edited = pl.read_csv(destination).with_columns(
        pl.when(pl.col("file") == "two")
        .then(pl.lit("PC0/"))
        .otherwise(pl.col("store"))
        .alias("store"),
        pl.when(pl.col("file") == "two")
        .then(pl.lit("cue"))
        .otherwise(pl.col("label"))
        .alias("label"),
    )
    edited.write_csv(destination)
    counts["two"] = {"Note": 8, "PC0/": 8, "PC1/": 1}
    second = write_events_csv(files, destination)
    two = second.filter(pl.col("file") == "two")
    assert two["store"].item() == "PC0/"
    assert two["label"].item() == "cue"
    assert two["Note"].item() == 8
