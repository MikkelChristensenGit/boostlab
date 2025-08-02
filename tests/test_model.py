import pandas as pd
from boostlab.model import XGBModel


def test_xgb_model_fit_predict_eval():
    df = pd.read_csv("data/dataset.csv")
    model = XGBModel(num_rounds=2)
    model.fit(df, "target")
    preds = model.predict(df.drop(columns=["target"]))
    assert isinstance(preds, pd.Series)
    metrics = model.evaluate(df["target"], preds)
    assert isinstance(metrics, dict), "metrics is not a dict"
    assert 0 <= metrics["acc"] <= 1.0, "accuracy is not between 0 and 1"
    assert 0 <= metrics["f1"] <= 1.0, "f1 is not between 0 and 1"
    # Check if confusion matrix is square and matches number of unique classes
    n_classes = df["target"].nunique()
    assert metrics["cm"].shape == (
        n_classes,
        n_classes,
    ), "Confusion matrix dimensions are incorrect"
