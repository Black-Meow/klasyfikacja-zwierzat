import numpy as np
import pandas as pd

CLASS_NAMES = [
    "antelope_duiker", "bird", "blank", "civet_genet",
    "hog", "leopard", "monkey_prosimian", "rodent",
]


def test_submission_probability_format():
    probabilities = np.full((3, len(CLASS_NAMES)), 1 / len(CLASS_NAMES))
    frame = pd.DataFrame(probabilities, columns=CLASS_NAMES)
    assert np.isfinite(frame.to_numpy()).all()
    assert np.all((frame.to_numpy() >= 0) & (frame.to_numpy() <= 1))
    assert np.allclose(frame.sum(axis=1).to_numpy(), 1.0)


def test_class_names_are_unique():
    assert len(CLASS_NAMES) == 8
    assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)
