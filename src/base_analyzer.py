"""
common functionality for all analyzers
"""
import numpy as np

class BaseAnalyzer:
    def __init__(self):
        pass
    
    def handle_nan_values(self, matrix: np.ndarray) -> np.ndarray:
        if np.isnan(matrix).any():
            print(f"Warning: NaN values found in matrix. Replacing with zeros.")
            matrix = np.nan_to_num(matrix)
        return matrix
