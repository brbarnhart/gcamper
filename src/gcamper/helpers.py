import polars as pl


def check_required_cols(df: pl.DataFrame, required_cols: list[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in dataframe: {sorted(missing)}")
