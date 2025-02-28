class BaseProfile:
    def __init__(self, id: int, name: str, base_scenario: int, MinimumHistoryPeriods: int):
        self.id = id
        self.name = name
        self.base_scenario = base_scenario
        self.MinimumHistoryPeriods = MinimumHistoryPeriods

    def __str__(self):
        return f"Profile Id : {self.id}, Name : {self.name}, Base Scenario : {self.base_scenario}, Minimum History Periods : {self.MinimumHistoryPeriods}"