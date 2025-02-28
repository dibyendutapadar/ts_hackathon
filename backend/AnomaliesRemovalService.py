from typing import List, Optional, Tuple

from backend.PredictV3EngineConfig import StatisticalDistributionConfig, TDFPrediction
from backend.TDFForecast import TDFForecast
from backend.PredictHelper import PredictHelper

class AnomaliesRemovalService:
    def __init__(self):
          pass

    def remove_anomaly_from_raw_input(self, input: List[float], boundaries: TDFPrediction, minimum_input_size: int) -> List[Optional[float]]:
        if len(input) == minimum_input_size:
            predictHelper = PredictHelper()
            return predictHelper.get_nullable_cast(input)

        # Try to remove anomalies in data
        filtered = [i for i in input if boundaries.riskBoundary.lowLowerBound <= i <= boundaries.riskBoundary.lowUpperBound]
        filtered_element_with_null = []
        for element in input:
            if boundaries.riskBoundary.lowLowerBound <= element <= boundaries.riskBoundary.lowUpperBound:
                filtered_element_with_null.append(element)
            else:
                filtered_element_with_null.append(None)

        if len(filtered) < minimum_input_size:
            predictHelper = PredictHelper()
            return predictHelper.get_nullable_cast(input)

        return filtered_element_with_null

    def remove_anomaly_from_raw_input_overload(self, input: List[float], boundaries: TDFPrediction) -> List[Optional[float]]:
        # Try to remove anomalies in data
        filtered = [i for i in input if boundaries.riskBoundary.lowLowerBound <= i <= boundaries.riskBoundary.lowUpperBound]
        filtered_element_with_null = []
        for element in input:
            if boundaries.riskBoundary.lowLowerBound <= element <= boundaries.riskBoundary.lowUpperBound:
                filtered_element_with_null.append(element)
            else:
                filtered_element_with_null.append(None)

        return filtered_element_with_null

    def remove_anomalies_from_raw_input(self, interprete_input: List[List[float]], stat_dist_config: StatisticalDistributionConfig) -> Tuple[List[List[Optional[float]]], List[TDFPrediction]]:
            result = []
            tdf_prediction_list = []
            total_periods = len(interprete_input)
            
            for i in range(total_periods):
                tdf_forecast = TDFForecast()
                tdf_prediction = tdf_forecast.get_prediction(interprete_input[i], stat_dist_config)
                filtered_input = self.remove_anomaly_from_raw_input(interprete_input[i], tdf_prediction, 3)
                last_element = filtered_input[-1] if filtered_input else None
                
                if last_element is None:
                    tdf_prediction = tdf_forecast.get_prediction(interprete_input[i], stat_dist_config, is_modified_outlier_logic=True)
                    filtered_input = self.remove_anomaly_from_raw_input(interprete_input[i], tdf_prediction, 3)
                    
                    if filtered_input[-1] is None:
                        last_index = len(filtered_input) - 1
                        filtered_input[last_index] = interprete_input[i][last_index]
                        tdf_prediction.riskBoundary.lowerRange = min(interprete_input[i]) / 2
                
                tdf_prediction_list.append(tdf_prediction)
                result.append(filtered_input)
            
            return result, tdf_prediction_list

    def remove_anomalies_from_raw_input_overload(self, raw_input: List[float], stat_dist_config: StatisticalDistributionConfig) -> Tuple[List[Optional[float]], TDFPrediction]:
            tdf_forecast = TDFForecast()
            tdf_prediction = tdf_forecast.get_predictions(raw_input, stat_dist_config)
            filtered_input = self.remove_anomaly_from_raw_input_overload(raw_input, tdf_prediction)
            last_element = filtered_input[-1] if filtered_input else None

            if last_element is None:
                tdf_prediction = tdf_forecast.get_prediction(raw_input, stat_dist_config, is_modified_outlier_logic=True)
                filtered_input = self.remove_anomaly_from_raw_input_overload(raw_input, tdf_prediction)

            return filtered_input, tdf_prediction