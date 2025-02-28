from dataclasses import dataclass
from typing import List, Optional


class PredictSameMonthOnMonthConfig:
    
    def __init__(self, normalBoundsAdjustFactor: float, lowBoundsAdjustFactor: float, mediumBoundsAdjustFactor: float):
        self.normalBoundsAdjustFactor = normalBoundsAdjustFactor
        self.lowBoundsAdjustFactor = lowBoundsAdjustFactor
        self.mediumBoundsAdjustFactor = mediumBoundsAdjustFactor

class PredictEngineWeightageConfig:
    
    def __init__(self, SSA_Weightage: float, HOLTWINTER_Weightage: float, V2_Weightage: float):
        self.SSA_Weightage = SSA_Weightage
        self.HOLTWINTER_Weightage = HOLTWINTER_Weightage
        self.V2_Weightage = V2_Weightage

class HoltWinterConfig:
    
    def __init__(self, Alpha: float, Phi: float, Beta: float):
        self.Alpha = Alpha
        self.Phi = Phi
        self.Beta = Beta

class SSAForecastConfig:
    
    def __init__(self, WindowSize: int, SeriesLength: int, TrainSize: int):
        self.WindowSize = WindowSize
        self.SeriesLength = SeriesLength
        self.TrainSize = TrainSize

class StatisticalDistributionRiskBoundary:
    
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

class StatisticalDistributionConfig:
    
    def __init__(self, NormalBoundary: StatisticalDistributionRiskBoundary, LowerRiskBoundary: StatisticalDistributionRiskBoundary, 
                 MediumRiskBoundary: StatisticalDistributionRiskBoundary, MinimumHistoryPeriods: int, DistributionCoefficient: float, 
                 AltNormalBoundary: float, AltLowRiskBoundary: float, AltMediumRiskBoundary: float, padding: float, 
                 DynamicDistribution: bool):
        self.NormalBoundary = NormalBoundary
        self.LowerRiskBoundary = LowerRiskBoundary
        self.MediumRiskBoundary = MediumRiskBoundary
        self.MinimumHistoryPeriods = MinimumHistoryPeriods
        self.DistributionCoefficient = DistributionCoefficient
        self.AltNormalBoundary = AltNormalBoundary
        self.AltLowRiskBoundary = AltLowRiskBoundary
        self.AltMediumRiskBoundary = AltMediumRiskBoundary
        self.padding = padding
        self.DynamicDistribution = DynamicDistribution

class PredictionConfig:
    pass

class ProjectionEngineBaseConfig(PredictionConfig):
    
    def __init__(self, SignalSensitivity: float, CalculateMultiplicationFactorDynamically: bool, 
                 DefaultDistributionCoefficient: float, StaticProjectionMultiplicationFactor: float, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SignalSensitivity = SignalSensitivity
        self.CalculateMultiplicationFactorDynamically = CalculateMultiplicationFactorDynamically
        self.DefaultDistributionCoefficient = DefaultDistributionCoefficient
        self.StaticProjectionMultiplicationFactor = StaticProjectionMultiplicationFactor

class ProjectionViaHoltWinterConfig(ProjectionEngineBaseConfig):
    
    def __init__(self, Alpha: float, Phi: float, StatisticalDistributionConfig: StatisticalDistributionConfig, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Alpha = Alpha
        self.Phi = Phi
        self.StatisticalDistributionConfig = StatisticalDistributionConfig

class PredictV3EngineConfig(PredictionConfig):
    
    def __init__(self, is_consider_two_year_history: bool, ml_round_off_digits_count: int, dynamic_engine_algorithm: int,
                 predictSameMonthOnMonthConfig: PredictSameMonthOnMonthConfig, 
                 PredictEngineWeightageConfig: PredictEngineWeightageConfig,
                 HoltWinterConfig: HoltWinterConfig,
                 SSAForecastConfig: SSAForecastConfig,
                 ProjectionViaHoltWinterConfig: ProjectionViaHoltWinterConfig,
                 *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.IsConsiderTwoYearHistory = is_consider_two_year_history
        self.MLRoundOffDigitsCount = ml_round_off_digits_count
        self.DynamicEngineAlgorithm = dynamic_engine_algorithm
        self.PredictSameMonthOnMonthConfig = predictSameMonthOnMonthConfig
        self.PredictEngineWeightageConfig = PredictEngineWeightageConfig
        self.HoltWinterConfig = HoltWinterConfig
        self.SSAForecastConfig = SSAForecastConfig
        self.ProjectionViaHoltWinterConfig = ProjectionViaHoltWinterConfig

class RiskBoundary:
    
    def __init__(self, lowerRange: Optional[float], upperRange: Optional[float],
                    lowLowerBound: Optional[float], lowUpperBound: Optional[float],
                 mediumLowerBound: Optional[float], mediumUpperBound: Optional[float]): 
        self.lowerRange = lowerRange
        self.upperRange = upperRange
        self.lowLowerBound = lowLowerBound
        self.lowUpperBound = lowUpperBound
        self.mediumLowerBound = mediumLowerBound
        self.mediumUpperBound = mediumUpperBound
    
    def to_dict(self):
        return {
            'lowerRange': self.lowerRange,
            'upperRange': self.upperRange,
            'lowLowerBound': self.lowLowerBound,
            'lowUpperBound': self.lowUpperBound,
            'mediumLowerBound': self.mediumLowerBound,
            'mediumUpperBound': self.mediumUpperBound
        }

class StatisticalForecastInput:
    
    def __init__(self, MonthNum: int, Value: Optional[float],
                    TimeId: int, FiscalYear: int): 
        self.MonthNum = MonthNum
        self.Value = Value
        self.TimeId = TimeId
        self.FiscalYear = FiscalYear

class Prediction:
    def __init__(self, predictedValue: float, riskBoundary: RiskBoundary):
        self.predictedValue = predictedValue
        self.riskBoundary = riskBoundary
    def to_dict(self):
        return {
            'predictedValue': self.predictedValue,
            'riskBoundary': self.riskBoundary.to_dict() if hasattr(self.riskBoundary, 'to_dict') else self.riskBoundary
        }
    def __str__(self):
        return f"Predicted Value: {self.predictedValue}"
    def __repr__(self):  # Optional but useful for debugging
        return self.__str__()
class TDFPrediction(Prediction):
    
    def __init__(self, median: float, minValue: float, maxValue: float,
                 percentile25: float, percentile75: float, interQuartileRange: float,
                 *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.median = median
        self.minValue = minValue
        self.maxValue = maxValue
        self.percentile25 = percentile25
        self.percentile75 = percentile75
        self.interQuartileRange = interQuartileRange

class SSASeriesForecast:
    
    def __init__(self, forecast: List[float]):
        self.forecast = forecast

class ForecastStatisticalParameter:
    
    def __init__(self, DistributionCoefficient: float, projection_multiplication_factor: float, error_of_standard_deviation: float, 
                 pearson_correlation_coefficient: float, mean: float, median: float, standard_deviation: float):
        self.DistributionCoefficient = DistributionCoefficient
        self.projection_multiplication_factor = projection_multiplication_factor
        self.error_of_standard_deviation = error_of_standard_deviation
        self.pearson_correlation_coefficient = pearson_correlation_coefficient
        self.mean = mean
        self.median = median
        self.standard_deviation = standard_deviation

    @property
    def coefficient_of_variation(self) -> float:
        if self.mean == 0:
            return float('inf')
        return self.standard_deviation / self.mean

class ForecastBundle:
    
    def __init__(self, predictions: List[float], normal_upper_bounds: List[float], normal_lower_bounds: List[float],
                 low_upper_bounds: List[float], low_lower_bounds: List[float], 
                 medium_upper_bounds: List[float], medium_lower_bounds: List[float], metadata: List[ForecastStatisticalParameter]):
        self.predictions = predictions
        self.normal_upper_bounds = normal_upper_bounds
        self.normal_lower_bounds = normal_lower_bounds
        self.low_upper_bounds = low_upper_bounds
        self.low_lower_bounds = low_lower_bounds
        self.medium_upper_bounds = medium_upper_bounds
        self.medium_lower_bounds = medium_lower_bounds
        self.metadata = metadata

class PredictHelper:
    
    @staticmethod
    def is_equal(value1: float, value2: float) -> bool:
        # Compare till 2 decimal places
        epsilon = 0.001
        return abs(value1 - value2) <= epsilon
    
    @staticmethod
    def is_two_pass_approach_required(period_count: int, flat_input_series: List[float]) -> bool:
        if period_count == 1:
            return False
        return len(flat_input_series) % period_count != 0
    
class TDFPrediction:
    def __init__(self, median, percentile25, percentile75, inter_quartile_range, min_value, max_value, predicted_value, riskBoundary):
        self.median = median
        self.percentile25 = percentile25
        self.percentile75 = percentile75
        self.inter_quartile_range = inter_quartile_range
        self.min_value = min_value
        self.max_value = max_value
        self.predicted_value = predicted_value
        self.riskBoundary = riskBoundary

class StatisticalPrediction:
    def __init__(self, predicted_value: float, ssa_predicted_value: Optional[float], holt_winter_predicted_value: Optional[float], v2_predicted_value: Optional[float]):
        self.predicted_value = predicted_value
        self.ssa_predicted_value = ssa_predicted_value
        self.holt_winter_predicted_value = holt_winter_predicted_value
        self.v2_predicted_value = v2_predicted_value

class EnginePredictions:
    def __init__(self, v3: List[float], ssa: List[float], holt_linear: List[float], v2: List[float]):
        self.v3 = v3
        self.ssa = ssa
        self.holt_linear = holt_linear
        self.v2 = v2

class MLAlgorithm:
    SSA = "SSA"
    HoltWinter = "HoltWinter"
    PredictV2 = "PredictV2"
    PredictV3 = "PredictV3"
