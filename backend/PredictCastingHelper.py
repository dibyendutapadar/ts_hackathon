from typing import List, Optional, Dict

class PredictCastingHelper:
     
    def __init__(self):
           pass

    def transform_to_flat_series(self, fiscal_year_bucketize_input: Dict[int, List[Optional[float]]]) -> List[Optional[float]]:
            flat_time_series = []
            for yearly_data in fiscal_year_bucketize_input.values():
                flat_time_series.extend(yearly_data)
            return flat_time_series

    def transform_to_flat_series(self, input_bucketized: List[List[Optional[float]]]) -> List[Optional[float]]:
            flat_input = []
            series_size = len(input_bucketized[0])

            for i in range(series_size):
                for input in input_bucketized:
                    if len(input) > i:
                        flat_input.append(input[i])
            return flat_input

    def transform_to_flat_series(self, input_bucketized: List[List[float]]) -> List[float]:
            flat_input = []
            series_size = len(input_bucketized[0])
            for i in range(series_size):
                for input in input_bucketized:
                    if len(input) > i:
                        flat_input.append(input[i])
            return flat_input

    def transform_to_bucketize_series(self, flat_time_series: List[Optional[float]], period_count: int) -> List[List[Optional[float]]]:
            result = []
            for i in range(len(flat_time_series)):
                if i < period_count:
                    result.append([flat_time_series[i]])
                else:
                    index = i % period_count
                    result[index].append(flat_time_series[i])
            return result
