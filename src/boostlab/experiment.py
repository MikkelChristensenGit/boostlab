from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from boostlab.model import XGBModel

# Script to orchestrate a full end-to-end experiment.


class Experiment(ABC):

    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame: ...

    @abstractmethod
    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...

    @abstractmethod
    def run_pipeline(self) -> dict[str, Any]: ...


class XGBExperiment(Experiment):

    def __init__(
        self,
        path: str,
        params: dict[str, Any],
        target_col: str,
        test_size: int,
        num_rounds: int,
        random_state: int,
    ) -> None:
        self.path = path
        self.params = params
        self.target_col = target_col
        self.test_size = test_size
        self.num_rounds = num_rounds
        self.random_state = random_state
        self.df: pd.DataFrame
        self.model: XGBModel

    def load_data(self, path: str) -> pd.DataFrame:
        self.df = pd.read_csv(path)
        return self.df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def run_pipeline(self) -> dict[str, Any]:
        df = self.load_data(self.path)
        X_train, X_test, y_train, y_test = self.split_data(
            df, self.target_col, self.test_size, self.random_state
        )
        self.model = XGBModel(params=self.params, num_rounds=self.num_rounds)
        train_df = pd.concat([X_train, y_train.rename(self.target_col)], axis=1)
        self.model.fit(train_df, self.target_col)
        preds = self.model.predict(X_test)
        return self.model.evaluate(y_test, preds)
