import pandas as pd

from abc import ABC, abstractmethod

# Put in __init__:
# - Anything that won't change between calls to your core method
# - Algorithm hyper-parameters, column lists, thresholds, external resources

# Pass to apply():
# - Everything that's different on each invocation
# - The actual dataset or chunk you need to process

# Guidelines:
### Constructor (__init__) - "What I am":
## Immutable configuration
# - Anything that defines how this object behaves, and won’t change after construction.
# - Algorithm hyperparameters (e.g. tree depth, learning rate)
# - Column names or feature lists
# - File paths, connection strings, external service clients
## Dependencies
# - Any collaborators or services this object requires to do its job.
# - A database or API client
# - A logger instance
# - A metrics/monitoring hook

### Public Methods - "What I do":
## Data Entry Points
# - Any method that acts on the changing payload (e.g. a new DataFrame, a batch of records)


class Preprocessor(ABC):
    """
    Base Class for preprocessing the dataset
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """apply method"""


class DropMissingPreprocessor(Preprocessor):
    """Drop missing values"""

    def apply(self, df: pd.DataFrame):
        return df.dropna()


class ZeroFillerPreprocessor(Preprocessor):
    """Fill missing values with zero in the given columns"""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            df[col] = df[col].fillna(0)
        return df


class MinMaxScalerPreprocessor(Preprocessor):
    """Scale columns to [0,1]"""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            col_min, col_max = df[col].min(), df[col].max()
            df[col] = (df[col] - col_min) / (col_max - col_min)
        return df


class PreprocessingPipeline:
    """
    Pipeline
    name: selects which strategy
    **cfg (e.g. columns=['age', 'income']) goes directly into the right constructor
    """

    STRATEGIES = {
        "drop": DropMissingPreprocessor,
        "zero": ZeroFillerPreprocessor,
        "minmax": MinMaxScalerPreprocessor,
    }

    @classmethod
    def build_preprocessor(cls, name: str, **cfg) -> Preprocessor:
        try:
            strategy_cls = cls.STRATEGIES[name]
        except KeyError:
            raise ValueError(f"Unknown preprocessing strategy: {name!r}")
        return strategy_cls(**cfg)

    def __init__(self, strategy: str, **cfg) -> None:
        self.processor = self.build_preprocessor(strategy, **cfg)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.processor.apply(df)
