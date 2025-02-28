from datetime import datetime
from typing import List

from backend.BaseProfile import BaseProfile
from backend.ProjectionConstants import ProjectionConstants

class StatsModelProfile(BaseProfile):
    _default = None

    def __init__(self, id: int, name: str, base_scenario: int, MinimumHistoryPeriods: int, normal_boundary_low: int,
                 normal_boundary_high: int, low_risk_boundary_low: int, low_risk_boundary_high: int, medium_risk_boundary_low: int,
                 medium_risk_boundary_high: int, DistributionCoefficient: float, alternate_boundaries: List[int], padding: float,
                 prediction_count: int, projection_start_year: int, DynamicDistribution: bool = False):
        super().__init__(id, name, base_scenario, MinimumHistoryPeriods)
        self.normal_boundary_low = normal_boundary_low
        self.normal_boundary_high = normal_boundary_high
        self.low_risk_boundary_low = low_risk_boundary_low
        self.low_risk_boundary_high = low_risk_boundary_high
        self.medium_risk_boundary_low = medium_risk_boundary_low
        self.medium_risk_boundary_high = medium_risk_boundary_high
        self.DistributionCoefficient = DistributionCoefficient
        self.alternate_boundaries = alternate_boundaries
        self.padding = padding
        self.DynamicDistribution = DynamicDistribution
        self.prediction_count = prediction_count
        self.projection_start_year = projection_start_year

    @staticmethod
    def initialize_default():
        StatsModelProfile._default = StatsModelProfile(
            id=2,
            name="Default TDF Profile",
            base_scenario=1,
            MinimumHistoryPeriods=36,
            normal_boundary_low=30,
            normal_boundary_high=80,
            low_risk_boundary_low=25,
            low_risk_boundary_high=85,
            medium_risk_boundary_low=20,
            medium_risk_boundary_high=90,
            DistributionCoefficient=1.5,
            alternate_boundaries=[5, 10, 20],
            padding=10,
            prediction_count=ProjectionConstants.MINIMUM_PREDICTION_COUNT,
            projection_start_year=datetime.utcnow().year
        )
        StatsModelProfile.Current = StatsModelProfile._default

    def __str__(self):
        return (f"{super().__str__()}, Normal Boundary : Low {self.normal_boundary_low} - High {self.normal_boundary_high}, "
                f"Low Risk Boundary : Low {self.low_risk_boundary_low} - High {self.low_risk_boundary_high}, "
                f"Medium Risk Boundary : Low {self.medium_risk_boundary_low} - High {self.medium_risk_boundary_high}, "
                f"Distribution Coefficient : {self.DistributionCoefficient}, Padding - {self.padding}, "
                f"Dynamic Distribution Enabled : {self.DynamicDistribution}, PredictionCount {self.prediction_count}, "
                f"ProjectionStartYear {self.projection_start_year}")

    def shallow_copy(self):
        return self

    def deep_copy(self):
        other = StatsModelProfile(
            id=self.id,
            name=self.name,
            base_scenario=self.base_scenario,
            MinimumHistoryPeriods=self.MinimumHistoryPeriods,
            normal_boundary_low=self.normal_boundary_low,
            normal_boundary_high=self.normal_boundary_high,
            low_risk_boundary_low=self.low_risk_boundary_low,
            low_risk_boundary_high=self.low_risk_boundary_high,
            medium_risk_boundary_low=self.medium_risk_boundary_low,
            medium_risk_boundary_high=self.medium_risk_boundary_high,
            DistributionCoefficient=self.DistributionCoefficient,
            alternate_boundaries=self.alternate_boundaries[:],
            padding=self.padding,
            prediction_count=self.prediction_count,
            projection_start_year=self.projection_start_year,
            DynamicDistribution=self.DynamicDistribution
        )
        return other

# Initialize the default profile
StatsModelProfile.initialize_default()