from typing import List

from backend.PredictV3EngineConfig import SSASeriesForecast

class PredictEngineHelper:

    def __init__(self):
        pass

    def handle_negatives_5(self, input_bucketized: List[List[float]], values_to_handle: List[float], ssa: SSASeriesForecast, hw: List[float], v2: List[List[float]]):
        for k in range(len(input_bucketized)):
            filtered_input = input_bucketized[k]

            total_value_count = len(values_to_handle)
            filtered_handle_value = []
            filtered_ssa_value = []
            filtered_hw_value = []
            filtered_v2_value = []
            count = 0
            for l in range(k, total_value_count, len(input_bucketized)):
                filtered_handle_value.append(values_to_handle[l])
                filtered_ssa_value.append(ssa.forecast[l])
                filtered_hw_value.append(hw[l])
                filtered_v2_value.append(v2[k][count])
                count += 1

            if any(x < 0 for x in filtered_input):
                continue

            if all(x >= 0 for x in filtered_ssa_value) and all(x >= 0 for x in filtered_hw_value) and all(x >= 0 for x in filtered_v2_value) and all(x >= 0 for x in filtered_handle_value):
                continue

            # If all inputs are positive and If any of HW, SSA, V2, Ensemble output is negative then it needs to be handled
            for i in range(len(filtered_handle_value)):
                if filtered_ssa_value[i] < 0 or filtered_hw_value[i] < 0:
                    if filtered_v2_value[i] > 0:
                        values_to_handle[i * len(input_bucketized) + k] = filtered_v2_value[i]
                    else:
                        values_to_handle[i * len(input_bucketized) + k] = max(filtered_input)
                else:
                    if filtered_v2_value[i] < 0:
                        values_to_handle[i * len(input_bucketized) + k] = max(filtered_input)


    def get_multiplication_factor(self, projection_multiplication_factor: float, SignalSensitivity: float) -> float:
        return projection_multiplication_factor * (1 + SignalSensitivity)

    def handle_negatives(self, input_bucketized: List[List[float]], values_to_handle: List[float], normal_upper_bounds: List[float] = None):
        for k in range(len(input_bucketized)):
            filtered_input = input_bucketized[k]

            total_value_count = len(values_to_handle)
            filtered_handle_value = []

            for l in range(k, total_value_count, len(input_bucketized)):
                filtered_handle_value.append(values_to_handle[l])

            if any(x < 0 for x in filtered_input):
                continue

            if all(x >= 0 for x in filtered_handle_value):
                continue

            min_projection = min(filtered_handle_value)

            shifted_values = [(x - min_projection) ** (1 / 1.1) for x in filtered_handle_value]

            for i in range(len(shifted_values)):
                if shifted_values[i] == 0:
                    upperRange = normal_upper_bounds[i * len(input_bucketized) + k] if normal_upper_bounds else None
                    if i > 0 and (shifted_values[i - 1] / 2 < upperRange):
                        shifted_values[i] = shifted_values[i - 1] / 2
                    elif normal_upper_bounds and (filtered_input[-1] / 2 < upperRange):
                        shifted_values[i] = filtered_input[-1] / 2  # Brute Force way. We need to think of better approach
                    elif (filtered_input[0] / 2 < upperRange):
                        shifted_values[i] = filtered_input[0] / 2
                    else:
                        shifted_values[i] = 0
                values_to_handle[i * len(input_bucketized) + k] = shifted_values[i]

    def round_off(self, values: List[float], round_off_digits: int):
        if values is None or len(values) == 0:
            return

        for i in range(len(values)):
            values[i] = round(values[i], round_off_digits)
            # Why round is chosen:
            # Rounding to nearest even is the standard in financial and statistical operations. It conforms to IEEE Standard 754, section 4.
            # See: https://docs.python.org/3/library/functions.html#round

def mean_absolute_percentage_error(actual: List[float], predicted: List[float]) -> float:
    if predicted is None:
        return 99999
    
    sum_error = 0

    for i in range(len(actual)):
        sum_error += abs((actual[i] - predicted[i]) / actual[i])
    
    mean_absolute_percentage_error_value = (sum_error / len(actual)) * 100

    return mean_absolute_percentage_error_value