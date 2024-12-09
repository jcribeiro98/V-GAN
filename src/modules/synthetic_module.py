import numpy as np
import pandas as pd


def generate_data(feature_count: int, datapoints: int = 1000):
    x_data = np.random.normal(
        size=datapoints*feature_count).reshape(datapoints, feature_count)
    return x_data
