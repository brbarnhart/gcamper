import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from gcamper import plot_isosbestic_diagnostics, plot_raw_traces


def _session_df(n: int = 200, animal: str = "UI128") -> pl.DataFrame:
    t = np.linspace(0.0, 100.0, n)
    isos = 30.0 - 0.02 * t + 0.2 * np.sin(t / 5.0)
    gcamp = 1.3 * isos + 5.0 + 0.3 * np.sin(t / 3.0)
    note = [None] * n
    note[10] = "Rest"
    note[50] = "Walk"
    return pl.DataFrame(
        {
            "ID": [animal] * n,
            "time": t,
            "gcamp": gcamp,
            "isos": isos,
            "Note": note,
        }
    )


def test_plot_raw_traces_returns_axes():
    ax = plot_raw_traces(_session_df())
    assert ax.get_xlabel() == "time (s)"
    plt.close(ax.figure)


def test_plot_isosbestic_diagnostics_per_animal(tmp_path):
    df = pl.concat([_session_df(animal="UI128"), _session_df(animal="UI129")])
    results = plot_isosbestic_diagnostics(df, by=["ID"], save_dir=tmp_path, show=False)
    assert len(results) == 2
    assert {r["keys"]["ID"] for r in results} == {"UI128", "UI129"}
    for name in (
        "Raw_ID-UI128.png",
        "Fit_ID-UI128.png",
        "Scatter_ID-UI128.png",
        "Raw_ID-UI129.png",
        "Fit_ID-UI129.png",
        "Scatter_ID-UI129.png",
    ):
        assert (tmp_path / name).is_file()
    for r in results:
        for fig in (r["raw"], r["fit"], r["scatter"]):
            plt.close(fig)
