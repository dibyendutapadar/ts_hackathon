import math
from typing import List, Optional
from backend.PredictCastingHelper import PredictCastingHelper
from backend.V3EngineHelper import V3EngineHelper
from backend.PredictV3EngineConfig import ForecastBundle

class PredictV3EngineGuard:
    def __init__(self):
        pass
    DECIMAL18DIGITSQLMIN = -83333333333333333
    DECIMAL18DIGITSQLMAX = 83333333333333333
    MINIMUM_INPUT_SIZE = 3

    def validate_data_count_for_dynamic_engine_selection(self, input_bucketized: List[List[Optional[float]]], minimum_years_for_dynamic_engine: int = 4) -> bool:
        predictCastingHelper = PredictCastingHelper()
        flat_list =   predictCastingHelper.transform_to_flat_series(input_bucketized)
        minimum_required_data = (len(input_bucketized) * minimum_years_for_dynamic_engine - 3)
        filtered_data = sum(1 for x in flat_list if x is not None)
        
        if filtered_data < minimum_required_data:
            return False
        
        return True

    def ensure_forecast_is_valid(self, forecast: ForecastBundle):
        """
        Guard clauses exceptions should never be caught.
        If you encounter exceptions here, then V3 Engine needs to be fixed.
        """
        sql_decimal_min = self.DECIMAL18DIGITSQLMIN
        sql_decimal_max = self.DECIMAL18DIGITSQLMAX
        n = len(forecast.predictions)
        
        for i in range(n):
            predicted_value = forecast.predictions[i]

            lowLowerBound = forecast.low_lower_bounds[i]
            lowUpperBound = forecast.low_upper_bounds[i]

            lowerRange = forecast.normal_lower_bounds[i]
            upperRange = forecast.normal_upper_bounds[i]

            mediumLowerBound = forecast.medium_lower_bounds[i]
            mediumUpperBound = forecast.medium_upper_bounds[i]

            message = []

            if upperRange < lowerRange:
                message.append("Normal Upper Bound should be greater than or equal to Normal Lower Bound.")

            if lowUpperBound < lowLowerBound:
                message.append("Low Upper Bound should be greater than or equal to Low Lower Bound.")

            if mediumUpperBound < mediumLowerBound:
                message.append("Medium Upper Bound should be greater than or equal to Medium Lower Bound.")

            if predicted_value < lowerRange or predicted_value > upperRange:
                message.append("Predicted value is outside of normal boundaries.")

            if lowLowerBound > lowerRange:
                message.append("Low Lower bound should be smaller than normal lower bound.")

            if mediumLowerBound > lowLowerBound:
                message.append("Medium lower bound should be smaller than low lower bound.")

            if upperRange > lowUpperBound:
                message.append("Low upper bound should be greater normal upper bound.")

            if lowUpperBound > mediumUpperBound:
                message.append("Medium upper bound should be greater than low upper bound.")

            if message:
                forecast.low_lower_bounds[i] = forecast.predictions[i] - 0.2 * abs(forecast.predictions[i])
                forecast.low_upper_bounds[i] = forecast.predictions[i] + 0.2 * abs(forecast.predictions[i])

                forecast.normal_lower_bounds[i] = forecast.predictions[i] - 0.1 * abs(forecast.predictions[i])
                forecast.normal_upper_bounds[i] = forecast.predictions[i] + 0.1 * abs(forecast.predictions[i])

                forecast.medium_lower_bounds[i] = forecast.predictions[i] - 0.3 * abs(forecast.predictions[i])
                forecast.medium_upper_bounds[i] = forecast.predictions[i] + 0.3 * abs(forecast.predictions[i])

                combined_message = f"Projection Ranges Failed for {i}th time period failed. Reason: {' '.join(message)}."
                #logger.info(combined_message)
                message = []

            if math.isnan(predicted_value):
                message.append("Prediction Value should not be NaN.")
            if math.isnan(lowLowerBound) or math.isnan(lowUpperBound):
                message.append("Lower Boundaries should not be NaN.")
            if math.isnan(lowerRange) or math.isnan(upperRange):
                message.append("Normal Boundaries should not be NaN.")
            if math.isnan(mediumLowerBound) or math.isnan(mediumUpperBound):
                message.append("Medium Boundaries should not be NaN.")
            if predicted_value > sql_decimal_max or predicted_value < sql_decimal_min:
                message.append("Prediction Value Exceed the Decimal Boundaries.")
            if upperRange > sql_decimal_max or lowerRange < sql_decimal_min:
                message.append("Normal Boundaries Exceed the Decimal Boundaries.")
            if lowUpperBound > sql_decimal_max or lowLowerBound < sql_decimal_min:
                message.append("Lower Boundaries Exceed the Decimal Boundaries.")
            if mediumUpperBound > sql_decimal_max or mediumLowerBound < sql_decimal_min:
                message.append("Medium Boundaries Exceed the Decimal Boundaries.")
            if message:
                combined_message = f"Projection for {i}th time period failed. Reason: {' '.join(message)}."
                #logger.error(combined_message)
                raise Exception(combined_message)
            
    def valid_series_count(self, input_bucketized: List[List[Optional[float]]]):
        if len(input_bucketized) == 0:
            raise ProjectionException("PredictV3Engine", ProjectionError(ProjectionErrorCode.NotEnoughHistoricalData))       

    def validate_series_bucket(self, input_bucketized: List[List[Optional[float]]]):
        input_bucketized_count = len(input_bucketized[0])
        first_series_count = input_bucketized_count
        valid_series_count = input_bucketized_count

        for input in input_bucketized:
            series_count = len(input)
            if series_count == valid_series_count and valid_series_count == first_series_count:
                continue
            elif series_count == first_series_count - 1:
                valid_series_count = first_series_count - 1
                continue
            else:
                raise ProjectionException("PredictV3Engine", ProjectionError(ProjectionErrorCode.InValidInputRequest)) 
            
    def trim_last_zeroes(self, input_bucketized: List[List[Optional[float]]], period_count: int) -> List[List[Optional[float]]]:
        predictCastingHelper = PredictCastingHelper()
        flat_list = predictCastingHelper.transform_to_flat_series(input_bucketized)
        total_elements = len(flat_list)

        for index in range(total_elements - 1, -1, -1):
            if flat_list[index] == 0.0 or flat_list[index] is None:
                flat_list.pop(index)
            else:
                break

        if len(flat_list) == 0:
            raise ProjectionException("PredictV3Engine", ProjectionError(ProjectionErrorCode.NotEnoughHistoricalData))

        predictCastingHelper = PredictCastingHelper()
        return predictCastingHelper.transform_to_bucketize_series(flat_list, period_count)

    def validate_valid_data_count(self, flat_list: List[Optional[float]], series_count: int):
        minimum_required_data = self.minimum_required_input_for_predict_v3_engine(series_count)
        filtered_data = sum(1 for x in flat_list if x is not None)
        if filtered_data < minimum_required_data:
            raise ProjectionException("PredictV3Engine", ProjectionError(ProjectionErrorCode.NotEnoughHistoricalData))

    def minimum_required_input_for_predict_v3_engine(self, period_count: int) -> int:
        return self.MINIMUM_INPUT_SIZE if period_count == 1 else (period_count * self.MINIMUM_INPUT_SIZE - self.MINIMUM_INPUT_SIZE)


    def ensure_input_is_valid(self, input_bucketized: List[List[Optional[float]]]) -> List[List[Optional[float]]]:
        self.valid_series_count(input_bucketized)
        self.validate_series_bucket(input_bucketized)
        input_bucketized = self.trim_last_zeroes(input_bucketized, len(input_bucketized))
        v3EngineHelper = V3EngineHelper()
        v3EngineHelper.trim_zero_to_null(input_bucketized)
        predictCastingHelper = PredictCastingHelper()
        flat_list = predictCastingHelper.transform_to_flat_series(input_bucketized)
        flat_list = v3EngineHelper.trim_yearly_empty_data(flat_list, len(input_bucketized))
        self.validate_valid_data_count(flat_list, len(input_bucketized))
        predictCastingHelper = PredictCastingHelper()
        return predictCastingHelper.transform_to_bucketize_series(flat_list, len(input_bucketized))


class ProjectionException(Exception):
    def __init__(self, message, error):
        super().__init__(message)
        self.error = error

class ProjectionError:
    def __init__(self, code):
        self.code = code

class ProjectionErrorCode:
    def __init__(self):
        pass
    NotEnoughHistoricalData = "NotEnoughHistoricalData"
    InValidInputRequest = "InValidInputRequest"