from collections import defaultdict
from typing import Dict, List, Optional
from backend.StatsModelProfile import StatsModelProfile
from backend.PredictV3EngineConfig import StatisticalDistributionConfig, StatisticalDistributionRiskBoundary, StatisticalForecastInput

class PredictHelper:

    def __init__(self):
        pass

    def get_statistical_dist_config(self, profile: StatsModelProfile) -> StatisticalDistributionConfig:
        return StatisticalDistributionConfig(
            NormalBoundary=StatisticalDistributionRiskBoundary(low=profile.normal_boundary_low / 100.0, high=profile.normal_boundary_high / 100.0),
            LowerRiskBoundary=StatisticalDistributionRiskBoundary(low=profile.low_risk_boundary_low / 100.0, high=profile.low_risk_boundary_high / 100.0),
            MediumRiskBoundary=StatisticalDistributionRiskBoundary(low=profile.medium_risk_boundary_low / 100.0, high=profile.medium_risk_boundary_high / 100.0),
            MinimumHistoryPeriods=profile.MinimumHistoryPeriods,
            DistributionCoefficient=profile.DistributionCoefficient,
            AltNormalBoundary=profile.alternate_boundaries[0] / 100.0,
            AltLowRiskBoundary=profile.alternate_boundaries[1] / 100.0,
            AltMediumRiskBoundary=profile.alternate_boundaries[2] / 100.0,
            padding=profile.padding,
            DynamicDistribution=profile.DynamicDistribution
        )

    def convert_to_month_level_data(self, inputs: List[StatisticalForecastInput]) -> Dict[int, List[Optional[float]]]:
        if not inputs:
            return {}

        month_dict = defaultdict(list)

        for input in inputs:
            month_dict[input.MonthNum].append(input.Value)

        return dict(sorted(month_dict.items()))

    def get_non_nullable_cast(self, input: List[Optional[float]]) -> List[float]:
        return [x for x in input if x is not None]

    def get_nullable_cast(self, input: List[float]) -> List[Optional[float]]:
        return [x for x in input]