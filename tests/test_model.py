import pandas as pd
from src.boostlab.model import XGBModel


def test_xgb_model_fit_predict_eval():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    model = XGBModel(num_rounds=2)
    model.fit(df, "y")
    preds = model.predict(df[["x"]])
    assert isinstance(preds, pd.Series)
    acc = model.evaluate(df["y"], preds)
    assert 0.0 <= acc <= 1.0
