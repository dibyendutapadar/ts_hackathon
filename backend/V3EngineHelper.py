from typing import List, Optional, Dict, Tuple


class V3EngineHelper:

    def __init__(self):
        pass

    def interprete_missing_value(self, filtered_input: List[List[Optional[float]]]) -> List[List[float]]:
        yearly_sum_dictionary = self.get_yearly_sum_dictionary(filtered_input)
        monthly_series_dictionary = self.get_monthly_series_dictionary(filtered_input)
        period_count = len(filtered_input)
        interpreted_input = []

        for i in range(period_count):
            monthly_series_count = len(filtered_input[i])
            interpreted_input.append([])
            for j in range(monthly_series_count):
                temp_value = filtered_input[i][j] if filtered_input[i][j] is not None else 0
                if filtered_input[i][j] is None:
                    monthly_series_average = monthly_series_dictionary[i]
                    yearly_series_average = yearly_sum_dictionary[j][0] / yearly_sum_dictionary[j][1]
                    if monthly_series_average == 0:
                        temp_value = yearly_series_average
                    else:
                        temp_value = 0.6 * yearly_series_average + 0.4 * monthly_series_average
                interpreted_input[i].append(temp_value)
        return interpreted_input

        
    def is_two_pass_approach_required(self, period_count: int, flat_input_series: List[float]) -> bool:
        return False if period_count == 1 else (len(flat_input_series) % period_count != 0)


    def get_monthly_series_dictionary(self, filtered_input: List[List[Optional[float]]]) -> Dict[int, float]:
        monthly_sum_dictionary = {}
        period_count = len(filtered_input)
        for i in range(period_count):
            total_sum = sum(x for x in filtered_input[i] if x is not None) or 0
            total_elements = sum(1 for x in filtered_input[i] if x is not None)
            average = total_sum / total_elements if total_elements > 0 else 0
            if average != average:  # Check for NaN
                average = 0
            monthly_sum_dictionary[i] = average
        return monthly_sum_dictionary

    def get_yearly_sum_dictionary(self, filtered_input: List[List[Optional[float]]]) -> Dict[int, Tuple[float, int]]:
        yearly_sum_dictionary = {}
        period_count = len(filtered_input)
        for i in range(period_count):
            monthly_series_count = len(filtered_input[i])
            for j in range(monthly_series_count):
                if filtered_input[i][j] is not None:
                    if j in yearly_sum_dictionary:
                        yearly_sum_dictionary[j] = (yearly_sum_dictionary[j][0] + filtered_input[i][j], yearly_sum_dictionary[j][1] + 1)
                    else:
                        yearly_sum_dictionary[j] = (filtered_input[i][j], 1)
        return yearly_sum_dictionary

    def trim_zero_to_null(self, input_bucketized: List[List[Optional[float]]]):
        for index in range(len(input_bucketized)):
            input_bucketized[index] = [None if x == 0 or (x is not None and (x == float('inf') or x != x)) else x for x in input_bucketized[index]]

    def trim_yearly_empty_data(self, flat_list: List[Optional[float]], period_count: int) -> List[Optional[float]]:
        result = []
        counter = 0
        total_elements = len(flat_list)
        while counter < total_elements:
            current_period_count = period_count if counter + period_count <= total_elements else total_elements - counter
            if current_period_count > 0:
                yearly_data = flat_list[counter:counter + current_period_count]
                if not all(x is None for x in yearly_data):
                    result.extend(yearly_data)
            counter += current_period_count
        return result