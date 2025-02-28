from typing import List, Optional, Tuple

from backend.PredictCastingHelper import PredictCastingHelper
from backend.PredictV3EngineConfig import StatisticalDistributionConfig, TDFPrediction
from backend.PredictV3EngineGuard import PredictV3EngineGuard
from backend.AnomaliesRemovalService import AnomaliesRemovalService
from backend.PredictHelper import PredictHelper
from backend.V3EngineHelper import V3EngineHelper
from backend.StatsModelProfile import StatsModelProfile

class InterpreterService:
    def __init__(self):
        pass

    def update_anomaly_filtered_input_trim_yearly_empty_data(self, interpreted_flat_list: List[float], flat_list: List[Optional[float]], period_count: int) -> List[Optional[float]]:
        result = []
        counter = 0
        total_elements = len(flat_list)
        
        while counter < total_elements:
            current_period_count = period_count if (counter + period_count) <= total_elements else total_elements - counter
            if current_period_count > 0:
                yearly_data = flat_list[counter:counter + current_period_count]
                if all(x is None for x in yearly_data):
                    result.extend([float(x) for x in interpreted_flat_list[counter:counter + current_period_count]])
                else:
                    result.extend(yearly_data)
            counter += current_period_count
        
        return result


    def handling_input_for_greater_than_two_year_history(self, input_bucketized: List[List[Optional[float]]], stat_dist_config: StatisticalDistributionConfig) -> Tuple[List[List[float]], List[TDFPrediction], List[List[float]], List[float]]:
        predictV3EngineGuard = PredictV3EngineGuard()
        filtered_input = predictV3EngineGuard.ensure_input_is_valid(input_bucketized)
        v3EngineHelper = V3EngineHelper()
        interprete_input = v3EngineHelper.interprete_missing_value(filtered_input)

        anomaliesRemovalService = AnomaliesRemovalService()
        anamoly_filtered_input, tdf_prediction = anomaliesRemovalService.remove_anomalies_from_raw_input(interprete_input, stat_dist_config)

        predictCastingHelper = PredictCastingHelper()
        flat_list = predictCastingHelper.transform_to_flat_series(anamoly_filtered_input)
        interpreted_flat_list = predictCastingHelper.transform_to_flat_series(interprete_input)
        updated_flat_series = self.update_anomaly_filtered_input_trim_yearly_empty_data(interpreted_flat_list, flat_list, len(input_bucketized))

        anamoly_filtered_input = predictCastingHelper.transform_to_bucketize_series(updated_flat_series, len(input_bucketized))
        bucketize_input_series = v3EngineHelper.interprete_missing_value(anamoly_filtered_input)
        flat_input_series_for_ssa = predictCastingHelper.transform_to_flat_series(bucketize_input_series)

        return interprete_input, tdf_prediction, bucketize_input_series, flat_input_series_for_ssa

    def handling_input_for_two_year_history(self, input_bucketized: List[List[Optional[float]]], stat_dist_config: StatisticalDistributionConfig) -> Tuple[List[List[float]], List[TDFPrediction], List[List[float]], List[float]]:
        v3EngineHelper = V3EngineHelper()
        predictV3EngineGuard = PredictV3EngineGuard()
        filtered_input = predictV3EngineGuard.ensure_input_is_valid(input_bucketized)
        interprete_input = v3EngineHelper.interprete_missing_value(filtered_input)
        predictCastingHelper = PredictCastingHelper()
        flat_input_series_for_ssa = predictCastingHelper.transform_to_flat_series(interprete_input)
        anomaliesRemovalService = AnomaliesRemovalService()
        anamoly_filtered_input_set, tdf_predictions = anomaliesRemovalService.remove_anomalies_from_raw_input(interprete_input, stat_dist_config)
        return interprete_input, tdf_predictions, interprete_input, flat_input_series_for_ssa


    def validate_and_interprete_input(self, input_bucketized: List[List[Optional[float]]], is_consider_two_year_history: bool = False) -> Tuple[List[List[float]], List[TDFPrediction], List[List[float]], List[float]]:

        StatsModelProfile.initialize_default()
        predictHelper = PredictHelper()
        stat_dist_config = predictHelper.get_statistical_dist_config(StatsModelProfile._default)

        if is_consider_two_year_history:
            return self.handling_input_for_two_year_history(input_bucketized, stat_dist_config)
        else:
            return self.handling_input_for_greater_than_two_year_history(input_bucketized, stat_dist_config)
