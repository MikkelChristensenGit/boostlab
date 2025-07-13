import pandas as pd


# --- Strategy functions ---
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Basic drop-rows with any NA."""
    return df.dropna()


def fill_zero(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values with zero in the given columns."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(0)
    return df


def scale_minmax(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale columns to [0,1]."""
    df = df.copy()
    for col in columns:
        col_min, col_max = df[col].min(), df[col].max()
        df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


# --- Context that applies a chosen strategy ---
def preprocess(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    """Apply the named preprocessor to df."""
    STRATEGIES = {
        "drop": drop_missing,
        "zero": fill_zero,
        "minmax": scale_minmax,
    }
    func = STRATEGIES[strategy]
    return func(df, **kwargs)


# --- Usage example ---
if __name__ == "__main__":
    raw = pd.DataFrame(
        {
            "age": [25, None, 40],
            "income": [50000, 60000, None],
        }
    )

    # Choose strategy by name
    cleaned = preprocess(raw, strategy="zero", columns=["age", "income"])
    print(cleaned)
