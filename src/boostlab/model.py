from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb

# Configuration goes into __init__
# Data is passed into fit
# features into predict
# true/false into evaluate

OBJECTIVE_MAP = {
    "binary": {"objective": "binary:logistic"},
    "multiclass": {"objective": "multi:softprob", "num_class_key": "num_class"},
}


class Model(ABC):

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]: ...


class XGBModel(Model):

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_rounds: int = 10,
        problem_type: str | None = None,
    ) -> None:
        """Holds configurations for Model"""
        defaults = {"objective": "binary:logistic", "tree_method": "hist"}
        self.params = {**defaults, **(params or {})}
        self.num_rounds = num_rounds
        self.problem_type = problem_type

    def fit(self, data: pd.DataFrame, target_col: str) -> None:
        """Train the XGB booster on the provided DataFrame and store it on self.booster."""
        # Set up the instance's state.
        # Following libraries, fit methods don't return the model - they mutate it in place
        # Training data = Data - Target col. Last argument specifies the labels
        y = data[target_col]
        unique = pd.unique(y)
        self.problem_type = "multiclass" if len(unique) > 2 else "binary"

        mapping = OBJECTIVE_MAP[self.problem_type]
        self.params["objective"] = mapping["objective"]

        if self.problem_type == "multiclass":
            num_classes = len(unique)
            self.params["num_class"] = num_classes

        dtrain = xgb.DMatrix(data.drop(columns=[target_col]), data[target_col])
        self.booster = xgb.train(self.params, dtrain, self.num_rounds)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the input features using the trained XGBoost model.
        Returns:
        pd.Series
            Predicted values for each input sample, indexed by the input DataFrame's index.
        """
        dtest = xgb.DMatrix(X)
        if not hasattr(self, "booster"):
            raise RuntimeError("Call fit() before predict()")
        raw = self.booster.predict(dtest)
        if self.problem_type == "multiclass":
            labels = raw.argmax(axis=1)
        else:
            labels = (raw > 0.5).astype(int)
        return pd.Series(labels, index=X.index)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
        """Tiny wrapper for evaluating accuracy"""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)
        return {
            "acc": acc,
            "f1": f1,
            "cm": cm,
        }
