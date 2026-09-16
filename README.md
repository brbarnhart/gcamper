# gcamper

Fiber photometry analysis toolkit (TDT-first).

Load TDT blocks into tidy session dataframes, compute regression-based ΔF/F, bleach-correct traces, and extract event-aligned windows.

## Install

```bash
uv sync
```

## Quick start

```python
from gcamper import (
    bleach_correct_df,
    create_master_df,
    plot_isosbestic_diagnostics,
    regression_based_dff,
)

df = bleach_correct_df(df, by=["ID"])
df = df.group_by("ID", maintain_order=True).map_groups(
    lambda g: g.with_columns(
        dFF=regression_based_dff(g["isos_bc"].to_numpy(), g["gcamp_bc"].to_numpy())
    )
)
plot_isosbestic_diagnostics(df, by=["ID"], save_dir="plots")
# After bleach correction, pass the corrected columns:
plot_isosbestic_diagnostics(
    df, by=["ID"], gcamp_col="gcamp_bc", isos_col="isos_bc", save_dir="plots"
)
```

## License

Proprietary / TBD.
