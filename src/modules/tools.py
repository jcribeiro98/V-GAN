import numpy as np
import pandas as pd
import random


def aggregator_funct(decision_function: np.array, type: str = "avg", weights: np.ndarray = None) -> np.ndarray:
    assert type in ["avg", "exact"], f"{type} aggregation not found"

    if type == "avg":
        return np.average(decision_function, axis=1, weights=weights)

    if type == "exact":
        weights = weights/weights.sum()
        aggregated_scores = [random.choices(scores, weights=weights)
                             for scores in decision_function]
        return aggregated_scores
