import hashlib
import json
from pathlib import Path

expected = json.loads((ROOT / "scripts/expected_lab3.json").read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["data"]["csv_sha256"]
assert len(y_train) == 13832 and len(y_test) == 3458
assert set(train_idx).isdisjoint(test_idx)
assert "Price" not in FEATURES
for name, values in expected["metrics"].items():
    for metric, value in values.items():
        assert np.isclose(RESULTS[name][metric], value, atol=1e-10, rtol=0), (
            name,
            metric,
        )
assert lr_search.best_params_["model__C"] == svm_search.best_params_["model__C"] == 1
assert np.array_equal(Y43, [[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0]])
assert np.array_equal(
    S43, [[0.7, 0.6, 0.5], [0.3, 0.4, 0.6], [0.5, 0.9, 0.2], [0.4, 0.5, 0.1]]
)
assert class_aucs == [Fraction(3, 4), Fraction(2, 3), Fraction(1)]
assert [macro43, weighted43, micro43, samples43] == [
    Fraction(29, 36),
    Fraction(3, 4),
    Fraction(5, 6),
    Fraction(7, 8),
]
for labels, scores in (
    [(Y43[:, j], S43[:, j]) for j in range(3)]
    + list(zip(Y43, S43))
    + [(Y43.ravel(), S43.ravel())]
):
    points, area = manual_roc(labels, scores)
    assert area == manual_pair_auc(labels, scores)[0]
    assert (points[0]["FPR"], points[0]["TPR"]) == (0, 0)
    assert (points[-1]["FPR"], points[-1]["TPR"]) == (1, 1)
for scores, area in [
    ([0.8, 0.2], Fraction(1)),
    ([0.2, 0.8], Fraction(0)),
    ([0.5, 0.5], Fraction(1, 2)),
]:
    assert manual_pair_auc([1, 0], scores)[0] == manual_roc([1, 0], scores)[1] == area
for fn in [manual_pair_auc, manual_roc]:
    try:
        fn([1, 1], [0.2, 0.8])
    except ValueError:
        pass
    else:
        raise AssertionError("Undefined AUC must be rejected")
assert np.array_equal(
    threshold_table43[["TP", "FP", "FN", "TN"]],
    [[3, 1, 0, 0], [2, 1, 1, 0], [1, 0, 2, 1]],
)
for table, labels, scores in [
    (threshold_table43, Y43[:, 1], S43[:, 1]),
]:
    for row in table.itertuples():
        predictions = scores >= row.threshold
        assert np.isclose(row.accuracy, accuracy_score(labels, predictions))
        assert np.isclose(row.precision, precision_score(labels, predictions))
        assert np.isclose(row.recall, recall_score(labels, predictions))
assert len(anomalies) == len(y_test) and int(anomalies.outlier.sum()) == 188
assert np.isfinite(anomalies.score).all()
assert np.array_equal(anomalies.score < 0, anomalies.outlier)
assert np.isclose(
    sensitivity.loc[sensitivity.nu == 0.05, "test_outlier_share"].iloc[0],
    anomalies.outlier.mean(),
)
print(
    "PASS: current kernel data, 21 metrics, manual ROC/AUC, thresholds and anomalies verified."
)
