import statistics
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import linregress, pearsonr
    

class PlanfulNativeStatisticsService:

    def __init__(self):
        pass

    def get_median(self, input: List[float]) -> float:
        if len(input) == 0:
            return 0
        return statistics.median(input)

    def get_percentile(self, input: List[float], percentile: float) -> float:
        if percentile < 0 or percentile > 1:
            raise Exception("Percentile should be within 0 and 1")
        
        input.sort()
        arr_length = len(input)
        n = (arr_length - 1) * percentile + 1
        
        if n == 1:
            return input[0]
        
        if n == arr_length:
            return input[arr_length - 1]
        
        k = int(n)
        d = n - k
        return input[k - 1] + d * (input[k] - input[k - 1])

    def get_inverse_td_value(self, min: float, max: float, most_likely: float, probability: float) -> float:
        if probability <= 0 or probability >= 1:
            raise Exception("Probability should be within 0 and 1")
        
        if max == min:
            return max
        
        if probability < (most_likely - min) / (max - min):
            return min + np.sqrt(probability * (most_likely - min) * (max - min))
        else:
            return max - np.sqrt((1 - probability) * (max - min) * (max - most_likely))

    def get_inverse_td_value_dict(self, min: float, max: float, most_likely: float, probability: List[float]) -> Dict[float, float]:
        if any(x <= 0 or x >= 1 for x in probability):
            raise Exception("Probability should be within 0 and 1")
        
        result = {}
        for prob in probability:
            result[prob] = self.get_inverse_td_value(min, max, most_likely, prob)
        
        return result

    def get_standard_deviation(self, input: List[float]) -> float:
        mean = self.get_mean(input)
        return np.sqrt(np.mean([(x - mean) ** 2 for x in input]))

    def get_distribution_coefficient(self, input: List[float], default_coeff: float = 1.5) -> float:
        mean = self.get_mean(input)
        if mean == 0:
            return default_coeff
        
        standard_deviation = self.get_standard_deviation(input)
        if standard_deviation == 0:
            return default_coeff
        
        coefficient_of_variation = standard_deviation / mean
        return max(np.log2(abs(1 / coefficient_of_variation)), default_coeff)

    def get_mean(self, input: List[float]) -> float:
        if input is None:
            raise ValueError("Input cannot be null")
        
        return np.mean(input)

    def get_coefficient_of_variation(self, input: List[float]) -> float:
        mean = self.get_mean(input)
        standard_deviation = self.get_standard_deviation(input)
        
        if standard_deviation == 0:
            return 0
        
        if mean == 0:
            return float('inf')
        
        return standard_deviation / mean

    def run_linear_regression(x_vals: List[float], y_vals: List[float]) -> Tuple[float, float, float]:
        
        if len(x_vals) != len(y_vals):
            raise Exception("Input values should be of the same length")
        
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        r_squared = r_value ** 2
        
        return r_squared, intercept, slope

    def pearson(self, input: List[float]) -> float:
        x_vals = list(range(1, len(input) + 1))
        return self.pearson_with_x(x_vals, input)

    def pearson_with_x(self, x: List[float], y: List[float]) -> float:
        if len(x) != len(y):
            raise Exception("Input values should be of the same length")
        if len(set(y)) == 1:
            return np.nan

        correlation, _ = pearsonr(x, y)
        return correlation