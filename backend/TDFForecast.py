from typing import List, Optional, Dict, Tuple
import numpy as np

from  backend.PlanfulNativeStatisticsService import PlanfulNativeStatisticsService
from  backend.PredictV3EngineConfig import RiskBoundary, TDFPrediction
from  backend.PredictHelper import PredictHelper

class TDFForecast:

    def __init__(self):
        return
    def predict(self, inputs: List[List[Optional[float]]], StatisticalDistributionConfig) -> List[TDFPrediction]:
        count = len(inputs)
        if count < StatisticalDistributionConfig.MinimumHistoryPeriods:
            return None

        output = []
        predictHelper = PredictHelper()
        month_dict = predictHelper.convert_to_month_level_data(inputs)
        for entry in month_dict.items():
            output.append(self.get_prediction(predictHelper.get_non_nullable_cast(entry[1]), StatisticalDistributionConfig))

        return output

    def get_predictions(self, inputs: List[List[float]], StatisticalDistributionConfig) -> List[TDFPrediction]:
        output = []
        for i in range(len(inputs)):
            output.append(self.get_prediction(inputs[i], StatisticalDistributionConfig))
        return output

    def get_prediction(self, input: List[float], config, is_modified_outlier_logic=False) -> TDFPrediction:
        dynamic_distribution_enabled = config.DynamicDistribution

        input_arr = np.array(input)
        boundary_probabilities = [
            config.NormalBoundary.low,
            config.NormalBoundary.high,
            config.LowerRiskBoundary.low,
            config.LowerRiskBoundary.high,
            config.MediumRiskBoundary.low,
            config.MediumRiskBoundary.high
        ]

        lower_percentile = 0.25
        upper_percentile = 0.75
        if is_modified_outlier_logic:
            upper_percentile = 0.95
        if dynamic_distribution_enabled:
            lower_percentile = 0.2
            upper_percentile = 0.8
            if is_modified_outlier_logic:
                upper_percentile = 0.95


        planfulNativeStatisticsService = PlanfulNativeStatisticsService()
        median = planfulNativeStatisticsService.get_median(input_arr)
        percentile25 = planfulNativeStatisticsService.get_percentile(input_arr, lower_percentile)
        percentile75 = planfulNativeStatisticsService.get_percentile(input_arr, upper_percentile)
        iqr = percentile75 - percentile25

        if iqr == 0:
            dedup_arr = np.unique(input_arr)
            if len(dedup_arr) == 1:
                output = TDFPrediction(
                    median=median,
                    percentile25=percentile25,
                    percentile75=percentile75,
                    inter_quartile_range=iqr,
                    min_value=dedup_arr[0],
                    max_value=dedup_arr[0],
                    predicted_value=None,
                    riskBoundary=RiskBoundary(
                        lowerRange=dedup_arr[0] * (1 + config.AltNormalBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 - config.AltNormalBoundary),
                        upperRange=dedup_arr[0] * (1 - config.AltNormalBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 + config.AltNormalBoundary),
                        lowLowerBound=dedup_arr[0] * (1 + config.AltLowRiskBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 - config.AltLowRiskBoundary),
                        lowUpperBound=dedup_arr[0] * (1 - config.AltLowRiskBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 + config.AltLowRiskBoundary),
                        mediumLowerBound=dedup_arr[0] * (1 + config.AltMediumRiskBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 - config.AltMediumRiskBoundary),
                        mediumUpperBound=dedup_arr[0] * (1 - config.AltMediumRiskBoundary) if dedup_arr[0] < 0 else dedup_arr[0] * (1 + config.AltMediumRiskBoundary)
                    )
                )
                self.add_padding(output, config.padding)
                return output
            else:
                planfulNativeStatisticsService = PlanfulNativeStatisticsService()
                median = planfulNativeStatisticsService.get_median(dedup_arr)
                percentile25 = planfulNativeStatisticsService.get_percentile(dedup_arr, lower_percentile)
                percentile75 = planfulNativeStatisticsService.get_percentile(dedup_arr, upper_percentile)
                iqr = percentile75 - percentile25

        DistributionCoefficient = config.DistributionCoefficient
        if dynamic_distribution_enabled:
            DistributionCoefficient =planfulNativeStatisticsService.get_distribution_coefficient(input_arr)

        upper_limit = percentile75 + DistributionCoefficient * iqr
        lower_limit = percentile25 - DistributionCoefficient * iqr

        predictions = planfulNativeStatisticsService.get_inverse_td_value_dict(lower_limit, upper_limit, median, boundary_probabilities)
        output = TDFPrediction(
            median=median,
            percentile25=percentile25,
            percentile75=percentile75,
            inter_quartile_range=iqr,
            min_value=lower_limit,
            max_value=upper_limit,
            predicted_value=None,
            riskBoundary=RiskBoundary(
                lowerRange=predictions[boundary_probabilities[0]],
                upperRange=predictions[boundary_probabilities[1]],
                lowLowerBound=predictions[boundary_probabilities[2]],
                lowUpperBound=predictions[boundary_probabilities[3]],
                mediumLowerBound=predictions[boundary_probabilities[4]],
                mediumUpperBound=predictions[boundary_probabilities[5]]
            )
        )
        self.add_padding(output, config.padding)
        return output

    def add_padding(self, output: TDFPrediction, padding: float):
        if output.riskBoundary is None:
            return

        output.riskBoundary.lowerRange -= padding
        output.riskBoundary.upperRange += padding
        output.riskBoundary.lowLowerBound -= padding
        output.riskBoundary.lowUpperBound += padding
        output.riskBoundary.mediumLowerBound -= padding
        output.riskBoundary.mediumUpperBound += padding