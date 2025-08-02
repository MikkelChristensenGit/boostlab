from boostlab.experiment import XGBExperiment


def test_experiment():
    exp = XGBExperiment(
        path="data/dataset.csv",
        params={},
        target_col="target",
        test_size=0.9,
        num_rounds=2,
        random_state=42,
    )
    metrics = exp.run_pipeline()
    assert isinstance(metrics, dict), "metrics is not a dict"
    assert 0 <= metrics["acc"] <= 1.0, "accuracy is not between 0 and 1"
    assert 0 <= metrics["f1"] <= 1.0, "f1 is not between 0 and 1"
    # Check if confusion matrix is square and matches number of unique classes
    # Infer number of classes from the confusion matrix shape
    cm = metrics["cm"]
    assert cm.shape[0] == cm.shape[1], "Confusion matrix is not square"
    n_classes = cm.shape[0]
    assert cm.shape == (
        n_classes,
        n_classes,
    ), "Confusion matrix dimensions are incorrect"
