import json
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

from backend.PredictCastingHelper import PredictCastingHelper
from backend.PredictV3EngineConfig import EnginePredictions, MLAlgorithm, PredictV3EngineConfig, Prediction, StatisticalPrediction
from backend.LinearExponentialSmoothingModel import LinearExponentialSmoothingModel
from backend.PlanfulNativeStatisticsService import PlanfulNativeStatisticsService
from backend.PredictEngineHelper import PredictEngineHelper, mean_absolute_percentage_error
from backend.PredictV3EngineGuard import PredictV3EngineGuard
from backend.SSA import SSAForecastModel
from backend.PredictV3EngineConfig import HoltWinterConfig, PredictEngineWeightageConfig, PredictSameMonthOnMonthConfig, PredictV3EngineConfig, Prediction, ProjectionEngineBaseConfig, ProjectionViaHoltWinterConfig, RiskBoundary, SSAForecastConfig, SSASeriesForecast, StatisticalDistributionConfig, StatisticalDistributionRiskBoundary, TDFPrediction, ForecastBundle, ForecastStatisticalParameter
from backend.PredictV3EngineConfig import PredictHelper
from backend.InterpreterService import InterpreterService

def dict_to_namespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def GetPredictions(bucketizedInput: List[List[Optional[float]]], 
                 predictionCount: int,
                config : PredictV3EngineConfig) -> List[Prediction]:
    
    # Convert JSON string to Python object
    if isinstance(config, dict):
        config_dict = json.loads(json.dumps(config))
        config = dict_to_namespace(config_dict)
        #enabling dynamic engine algorithm by default
        config.DynamicEngineAlgorithm = 1
    if predictionCount <= 0:
        return []

    projection_upper_bounds: List[float] = []
    projection_lower_bounds: List[float] = []
    cor_dispersion: List[float] = []

    interprete_input: List[List[float]] = []
    bucketize_input_series: List[List[float]] = []
    tdf_prediction_list: List[PredictV3EngineConfig.TDFPrediction] = []
    flat_input_series_for_ssa: List[float] = []
    predictions = []
    
    interpreterService = InterpreterService()
    interprete_input, tdf_prediction_list, bucketize_input_series, flat_input_series_for_ssa = interpreterService.validate_and_interprete_input(bucketizedInput, config.IsConsiderTwoYearHistory)

    forecast_response, ssa_series_forecast, holtwinterprediction, predict_v2_prediction, tdf_predictions, is_same_for_all_values = get_forecast(
    bucketizedInput, predictionCount, config, projection_upper_bounds, projection_lower_bounds, cor_dispersion,
    interprete_input, tdf_prediction_list, bucketize_input_series, flat_input_series_for_ssa
    )

    predictions = transform_to_predictions(forecast_response)

    return predictions

def get_forecast(input_bucketized: List[List[Optional[float]]], prediction_count: int, prediction_config: PredictV3EngineConfig, 
                 projection_upper_bounds: List[float], projection_lower_bounds: List[float], cor_dispersion: List[float], 
                 interprete_input: List[List[float]], tdf_prediction_list: List[TDFPrediction], 
                 bucketize_input_series: List[List[float]], flat_input_series_for_ssa: List[float], 
                 sensitivity: Optional[float] = None) -> Tuple[ForecastBundle, SSASeriesForecast, List[float], List[List[float]], List[TDFPrediction], bool]:
    
    # Initialize the return values
    double_array = []
    list_of_double_arrays = []
    tdf_prediction_list = []
    boolean_value = False
    
    period_count = len(input_bucketized)
    number_of_predictions = period_count * prediction_count

    normal_upper_bounds = [0.0] * number_of_predictions
    normal_lower_bounds = [0.0] * number_of_predictions
    low_upper_bounds = [0.0] * number_of_predictions
    low_lower_bounds = [0.0] * number_of_predictions
    medium_upper_bounds = [0.0] * number_of_predictions
    medium_lower_bounds = [0.0] * number_of_predictions
    ensemble_forecast_list = [0.0] * number_of_predictions
    

    prediction_config.ProjectionViaHoltWinterConfig.SignalSensitivity = sensitivity if sensitivity is not None else prediction_config.ProjectionViaHoltWinterConfig.SignalSensitivity
    isSameForAllMonthOnMonthValues = check_all_month_on_month_value_equal(interprete_input, period_count)

    #region Pre-checking if all Month-on-Month values are same
    if isSameForAllMonthOnMonthValues:
        adjust_boundaries_and_forecast_for_same_month_on_month_values(prediction_config.PredictSameMonthOnMonthConfig, period_count, prediction_count,
                                                                      normal_upper_bounds, normal_lower_bounds, low_upper_bounds,
                                                                      low_lower_bounds, medium_upper_bounds, medium_lower_bounds,
                                                                      interprete_input, ensemble_forecast_list)
        
        forecast_bundle = ForecastBundle(
        predictions=ensemble_forecast_list,
        normal_upper_bounds=normal_upper_bounds,
        normal_lower_bounds=normal_lower_bounds,
        low_upper_bounds=low_upper_bounds,
        low_lower_bounds=low_lower_bounds,
        medium_upper_bounds=medium_upper_bounds,
        medium_lower_bounds=medium_lower_bounds,
        metadata=[ForecastStatisticalParameter(
            DistributionCoefficient=0.0,
            projection_multiplication_factor=0.0,
            error_of_standard_deviation=0.0,
            pearson_correlation_coefficient=0.0,
            mean=0.0,
            median=0.0,
            standard_deviation=0.0
            ) for _ in range(number_of_predictions)]
        )

        ssa_series_forecast = SSASeriesForecast(forecast=[0.0] * number_of_predictions)

        double_array = [0.0] * number_of_predictions

        list_of_double_arrays = [[0.0] * prediction_count for _ in range(period_count)]

        tdf_prediction_list = [TDFPrediction(
             median=0,
            percentile25=0,
            percentile75=0,
            inter_quartile_range=0,
            min_value=0,
            max_value=0,
            predicted_value=None,
            riskBoundary=RiskBoundary(
                lowLowerBound=0.0,
                lowUpperBound=0.0,
                mediumLowerBound=0.0,
                mediumUpperBound=0.0,
                lowerRange=0.0,
                upperRange=0.0
            )
        ) for _ in range(period_count)]

        boolean_value = True

        return forecast_bundle, ssa_series_forecast, double_array, list_of_double_arrays, tdf_prediction_list, boolean_value
    #endregion

    #region Config Section
    
    projectionConfig = prediction_config.ProjectionViaHoltWinterConfig
    projectionViaHoltWinterConfig = projectionConfig
    statDistConfig = projectionViaHoltWinterConfig.StatisticalDistributionConfig

    #endregion

    isTwoPassRequired = PredictHelper.is_two_pass_approach_required(period_count, flat_input_series_for_ssa)

    #region STEP 1: SSA Projection

    ssaPrediction = get_ssa_forecast(isTwoPassRequired, prediction_count, period_count, flat_input_series_for_ssa, prediction_config.SSAForecastConfig);

    #endregion STEP 1: SSA Projection

    #region STEP 2: Holt Winter

    holt_winter_predictions, yhat = get_holt_winter_forecast(isTwoPassRequired, prediction_config, period_count, prediction_count, flat_input_series_for_ssa)

    #endregion

    #region  STEP 3: Month on Month Prediction - Holt Winter

    predict_forecast_list = []
    predict_forecast_statistical_parameter = []

    month_on_month_holt_winter_forecast(prediction_count, period_count, projectionConfig, statDistConfig, flat_input_series_for_ssa,
                                    bucketize_input_series, predict_forecast_list, predict_forecast_statistical_parameter)

    #endregion

    #region STEP 4: Ensemble Forecast - pending

    #region Dynamic Algorithm selection
    predictV3EngineGuard = PredictV3EngineGuard()
    if prediction_config.DynamicEngineAlgorithm == 1 and predictV3EngineGuard.validate_data_count_for_dynamic_engine_selection(input_bucketized):
        ensemble_forecast_list = get_dynamic_engine_forecast(prediction_count, bucketize_input_series, flat_input_series_for_ssa, period_count, prediction_config, ssaPrediction, holt_winter_predictions, predict_forecast_list)
    else:
        ensemble_forecast_list = calculate_ensemble_forecast(prediction_count, period_count, ssaPrediction, holt_winter_predictions, predict_forecast_list, prediction_config.PredictEngineWeightageConfig)
    #endregion
    #endregion

    predictEngineHelper = PredictEngineHelper()
    predictEngineHelper.handle_negatives_5(interprete_input, ensemble_forecast_list, ssaPrediction, holt_winter_predictions, predict_forecast_list)

    #region STEP 5: Risk Boundaries

    normal_upper_bounds, normal_lower_bounds, low_upper_bounds, low_lower_bounds, medium_upper_bounds, medium_lower_bounds = calculate_risk_boundaries(period_count, projectionConfig, predict_forecast_statistical_parameter, ensemble_forecast_list, interprete_input)
    
    #endregion

    #region Calculate distance between risk boundaries

    dist = [[0.0 for _ in range(4)] for _ in range(prediction_count * period_count)]

    for i in range(prediction_count * period_count):
        dist[i][0] = low_lower_bounds[i] - medium_lower_bounds[i]
        dist[i][1] = normal_lower_bounds[i] - low_lower_bounds[i]
        dist[i][2] = low_upper_bounds[i] - normal_upper_bounds[i]
        dist[i][3] = medium_upper_bounds[i] - low_upper_bounds[i]

    #endregion

     #region Negative Handling of Normal Lower Boundaries

    predictEngineHelper.handle_negatives(interprete_input, normal_lower_bounds, normal_upper_bounds)

    #endregion

    #region Adjust Lower & Medium Lower side Boundaries due to Negative Handling of Normal Lower Boundaries in Previous Step

    for i in range(prediction_count * period_count):
        low_lower_bounds[i] = normal_lower_bounds[i] - abs(dist[i][1])
        medium_lower_bounds[i] = low_lower_bounds[i] - abs(dist[i][0])

    #endregion

    #region Tune Normal Boundaries

    projection_upper_bounds.extend(normal_upper_bounds)
    projection_lower_bounds.extend(normal_lower_bounds)

    cor_dispersion = tune_boundaries(interprete_input, normal_upper_bounds, normal_lower_bounds, tdf_prediction_list, predict_forecast_statistical_parameter, ensemble_forecast_list)

    predictEngineHelper.handle_negatives(interprete_input, normal_lower_bounds, normal_upper_bounds)

    #endregion

    #region Derive Low & Medium Boundaries using tuned Normal Boundaries & honoring initial gap

    for i in range(prediction_count * period_count):
        low_lower_bounds[i] = normal_lower_bounds[i] - abs(dist[i][1])
        medium_lower_bounds[i] = low_lower_bounds[i] - abs(dist[i][0])
        low_upper_bounds[i] = normal_upper_bounds[i] + abs(dist[i][2])
        medium_upper_bounds[i] = low_upper_bounds[i] + abs(dist[i][3])

    #endregion

    #region Fix Predicted Value (If Predicted Value is not within normal bounds then forcefully bring it within normal bounds)

    fix_predicted_value_outside_of_bounds(ensemble_forecast_list, normal_upper_bounds, normal_lower_bounds);

    #endregion

    #region Round Off Predictions & Boundaries

    predictEngineHelper = PredictEngineHelper()
    predictEngineHelper.round_off(ensemble_forecast_list, prediction_config.MLRoundOffDigitsCount);

    predictEngineHelper.round_off(normal_upper_bounds, prediction_config.MLRoundOffDigitsCount);
    predictEngineHelper.round_off(normal_lower_bounds, prediction_config.MLRoundOffDigitsCount);

    predictEngineHelper.round_off(low_upper_bounds, prediction_config.MLRoundOffDigitsCount);
    predictEngineHelper.round_off(low_lower_bounds, prediction_config.MLRoundOffDigitsCount);

    predictEngineHelper.round_off(medium_upper_bounds, prediction_config.MLRoundOffDigitsCount);
    predictEngineHelper.round_off(medium_lower_bounds, prediction_config.MLRoundOffDigitsCount);

    #endregion

    #region Overriding the ensemble forecast and ranges when we have same values for Month-on-Month

    
    adjust_boundaries_and_forecast_for_same_month_on_month_values(prediction_config.PredictSameMonthOnMonthConfig, period_count, prediction_count,
                                                                      normal_upper_bounds, normal_lower_bounds, low_upper_bounds,
                                                                      low_lower_bounds, medium_upper_bounds, medium_lower_bounds,
                                                                      interprete_input, ensemble_forecast_list)
    #endregion

    forecast_bundle = ForecastBundle(
        predictions=ensemble_forecast_list,
        normal_upper_bounds=normal_upper_bounds,
        normal_lower_bounds=normal_lower_bounds,
        low_upper_bounds=low_upper_bounds,
        low_lower_bounds=low_lower_bounds,
        medium_upper_bounds=medium_upper_bounds,
        medium_lower_bounds=medium_lower_bounds,
        metadata=[ForecastStatisticalParameter(
            DistributionCoefficient=0.0,
            projection_multiplication_factor=0.0,
            error_of_standard_deviation=0.0,
            pearson_correlation_coefficient=0.0,
            mean=0.0,
            median=0.0,
            standard_deviation=0.0
            ) for _ in range(number_of_predictions)]
        )

    predictV3EngineGuard = PredictV3EngineGuard()
    predictV3EngineGuard.ensure_forecast_is_valid(forecast_bundle)

    return forecast_bundle, ssaPrediction, holt_winter_predictions, predict_forecast_list, tdf_prediction_list, False

def check_all_month_on_month_value_equal(interprete_input: List[List[float]], period_count: int) -> bool:
    for bucket_index in range(period_count):
        first = interprete_input[bucket_index][0]
        if not all(PredictHelper.is_equal(x, first) for x in interprete_input[bucket_index]):
            return False
    return True

def adjust_boundaries_and_forecast_for_same_month_on_month_values(predict_same_month_on_month_config, period_count, prediction_count,
                                                                  normal_upper_bounds, normal_lower_bounds, low_upper_bounds,
                                                                  low_lower_bounds, medium_upper_bounds, medium_lower_bounds,
                                                                  interprete_input, ensemble_forecast_list):
    for bucket_index in range(period_count):
        first = interprete_input[bucket_index][0]

        if all(PredictHelper.is_equal(x, first) for x in interprete_input[bucket_index]):
            for p_count in range(prediction_count):
                index = bucket_index + period_count * p_count

                ensemble_forecast_list[index] = first
                normal_upper_bounds[index] = first + (abs(first) * 0.1)
                normal_lower_bounds[index] = first - (abs(first) * 0.1)

                low_upper_bounds[index] = first + (abs(first) * 0.2)
                low_lower_bounds[index] = first - (abs(first) * 0.2)

                medium_upper_bounds[index] = first + (abs(first) * 0.3)
                medium_lower_bounds[index] = first - (abs(first) * 0.3)

def get_ssa_forecast(is_two_pass_required: bool, prediction_count: int, period_count: int, flat_input_series_for_ssa: List[float], ssa_forecast_config: SSAForecastConfig) -> SSASeriesForecast:
    ssa_forecast_model = SSAForecastModel()
    
    if not is_two_pass_required:
        return ssa_forecast_model.forecast(prediction_count * period_count, flat_input_series_for_ssa, ssa_forecast_config)
    else:
        extra_periods_count = len(flat_input_series_for_ssa) % period_count
        first_prediction_count = extra_periods_count
        second_prediction_count = (period_count * prediction_count) - extra_periods_count

        # Forecasting for closed periods
        first_output = ssa_forecast_model.forecast(first_prediction_count, flat_input_series_for_ssa[:-extra_periods_count], ssa_forecast_config)

        # Forecasting for open periods
        second_output = ssa_forecast_model.forecast(second_prediction_count, flat_input_series_for_ssa, ssa_forecast_config)

        combined_forecast = first_output.forecast + second_output.forecast
        return SSASeriesForecast(forecast=combined_forecast)

def get_holt_winter_forecast(is_two_pass_required: bool, predict_v3_engine_config: PredictV3EngineConfig, period_count: int, 
                             prediction_count: int, flat_input_series_for_ssa: List[float]) -> Tuple[List[float], List[float]]:
    holt_winter_config = predict_v3_engine_config.HoltWinterConfig
    smoothing_model = LinearExponentialSmoothingModel(holt_winter_config.Phi, holt_winter_config.Alpha, holt_winter_config.Beta)

    if not is_two_pass_required:
        return smoothing_model.forecast(flat_input_series_for_ssa, period_count * prediction_count)
    else:
        extra_periods_count = len(flat_input_series_for_ssa) % period_count
        first_prediction_count = extra_periods_count
        second_prediction_count = (period_count * prediction_count) - extra_periods_count

        # Forecasting for closed periods
        first_output_projection, first_output_yhat = smoothing_model.forecast(flat_input_series_for_ssa[:len(flat_input_series_for_ssa) - extra_periods_count], first_prediction_count)
        
        # Forecasting for open periods
        second_output_projection, second_output_yhat = smoothing_model.forecast(flat_input_series_for_ssa, second_prediction_count)

        combined_projection = first_output_projection + second_output_projection
        combined_yhat = first_output_yhat + second_output_yhat

        return combined_projection, combined_yhat
    
def make_forecast_for_input(filtered_input: List[float], prediction_count: int, projection_config: ProjectionEngineBaseConfig, 
                            stat_dist_config: StatisticalDistributionConfig) -> Tuple[List[float], ForecastStatisticalParameter]:
    # Get predictions for Filtered Input via Holt's Winter Forecasting method
    holt_winter_config = projection_config  # Assuming projection_config is of type ProjectionViaHoltWinterConfig

    smoothing_model = LinearExponentialSmoothingModel(Phi=holt_winter_config.Phi, Alpha=holt_winter_config.Alpha, Beta=0.1)

    predictions, yhat = smoothing_model.forecast(filtered_input, prediction_count)

    # Calculate Statistical Parameters
    error_sum = 0
    for j in range(len(yhat) - 2):
        error = filtered_input[j] - yhat[j]
        error_sum += error * error

    planfulNativeStatisticsService = PlanfulNativeStatisticsService();
    DistributionCoefficient = planfulNativeStatisticsService.get_distribution_coefficient(filtered_input, projection_config.DefaultDistributionCoefficient)

    forecast_statistical_parameter = ForecastStatisticalParameter(
        DistributionCoefficient=DistributionCoefficient,
        pearson_correlation_coefficient=planfulNativeStatisticsService.pearson(filtered_input),
        error_of_standard_deviation=(error_sum / len(filtered_input)) ** 0.5,
        projection_multiplication_factor=min(DistributionCoefficient, 15) if projection_config.CalculateMultiplicationFactorDynamically else projection_config.StaticProjectionMultiplicationFactor,
        standard_deviation=planfulNativeStatisticsService.get_standard_deviation(filtered_input),
        mean=planfulNativeStatisticsService.get_mean(filtered_input),
        median=planfulNativeStatisticsService.get_median(filtered_input)
    )

    return predictions, forecast_statistical_parameter

def month_on_month_holt_winter_forecast(prediction_count: int, period_count: int, projection_config: ProjectionEngineBaseConfig, 
                                        stat_dist_config: StatisticalDistributionConfig, flat_input_series_for_ssa: List[float], 
                                        bucketize_input_series: List[List[float]], predict_forecast_list: List[List[float]], 
                                        predict_forecast_statistical_parameter: List[ForecastStatisticalParameter]):
    extra_periods_count = len(flat_input_series_for_ssa) % period_count
    for index in range(period_count):
        is_two_pass_period_available = index < extra_periods_count
        if is_two_pass_period_available:
            second_pass_input = bucketize_input_series[index]
            first_pass_input = second_pass_input[:-1]

            # For Closed Periods Prediction: We are not using StatisticalParameter (as it is not considering full data series Input)
            a_forecast, _ = make_forecast_for_input(first_pass_input, 1, projection_config, stat_dist_config)
            b_forecast, next_statistical_parameter = make_forecast_for_input(second_pass_input, prediction_count - 1, projection_config, stat_dist_config)

            predict_forecast_list.append(a_forecast + b_forecast)
            predict_forecast_statistical_parameter.append(next_statistical_parameter)
        else:
            input = bucketize_input_series[index]
            a_forecast, statistical_parameter = make_forecast_for_input(input, prediction_count, projection_config, stat_dist_config)
            predict_forecast_list.append(a_forecast)
            predict_forecast_statistical_parameter.append(statistical_parameter)    

def validate_predict_v2_fallback_probability(prediction_count: int, period_count: int, ssa_prediction: SSASeriesForecast, 
                                             holt_winter_predictions: List[float], predict_forecast_list: List[List[float]]) -> bool:
    for j in range(prediction_count):
        for i in range(period_count):
            v2_forecast = abs(predict_forecast_list[i][j])
            ssa_forecast = abs(ssa_prediction.forecast[(j * period_count) + i])
            hw_forecast = abs(holt_winter_predictions[(j * period_count) + i])
            if ssa_forecast >= 2 * v2_forecast or hw_forecast >= 2 * v2_forecast:
                return True
    return False

def calculate_ensemble_forecast(prediction_count: int, period_count: int, ssa_prediction: SSASeriesForecast, 
                                holt_winter_predictions: List[float], predict_forecast_list: List[List[float]], 
                                predict_engine_weightage_config: PredictEngineWeightageConfig) -> List[float]:
    """
    To Get the Ensemble Forecast using the formula
    Ensemble Forecast = (0.5 * predict Projection) + (0.25 * SSA Forecast) + (0.25 * HoltWinter Forecast)
    """
    is_fallback_to_v2 = validate_predict_v2_fallback_probability(prediction_count, period_count, ssa_prediction, holt_winter_predictions, predict_forecast_list)

    ensemble_forecast_list = []
    for j in range(prediction_count):
        for i in range(period_count):
            if is_fallback_to_v2:
                ensemble_forecast = predict_forecast_list[i][j]
            else:
                ensemble_forecast = (predict_engine_weightage_config.V2_Weightage * predict_forecast_list[i][j] + 
                                     predict_engine_weightage_config.SSA_Weightage * ssa_prediction.forecast[(j * period_count) + i] + 
                                     predict_engine_weightage_config.HOLTWINTER_Weightage * holt_winter_predictions[(j * period_count) + i])
            ensemble_forecast_list.append(ensemble_forecast)
    
    return ensemble_forecast_list

def calculate_risk_boundaries(period_count: int, projection_config: ProjectionEngineBaseConfig, 
                              predict_forecast_statistical_parameter: List[ForecastStatisticalParameter], 
                              ensemble_forecast_list: List[float], interprete_input: List[List[float]]) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    # Variables description
    nbpfl = []
    lbpfl = []
    mbpfl = []

    nbpfl_ma = []
    lbpfl_ma = []
    mbpfl_ma = []

    enbpfl = []
    enbpfl_ma = []

    is_normal_lower_bound_calibrated = []

    for i in range(period_count):
        forecast = ensemble_forecast_list[i]
        forecast_statistical_parameter = predict_forecast_statistical_parameter[i]

        normal_signal_sensitivity = projection_config.SignalSensitivity
        predictEngineHelper = PredictEngineHelper()
        normal_multiplication_factor = predictEngineHelper.get_multiplication_factor(forecast_statistical_parameter.projection_multiplication_factor, normal_signal_sensitivity)

        nbpfl.append(normal_multiplication_factor * forecast_statistical_parameter.error_of_standard_deviation)
        nbpfl_ma.append(sum(nbpfl[-3:]) / len(nbpfl[-3:]))

        width = normal_multiplication_factor * forecast_statistical_parameter.error_of_standard_deviation
        if (forecast - (normal_multiplication_factor * forecast_statistical_parameter.error_of_standard_deviation) < 0) and (not any(x < 0 for x in interprete_input[i])):
            is_normal_lower_bound_calibrated.append(True)
            if forecast_statistical_parameter.pearson_correlation_coefficient > 0:
                width = forecast - min(interprete_input[i])
                width = max(width, 0)  # This will fix LB > UB
            else:
                width = forecast - 0.0
        else:
            is_normal_lower_bound_calibrated.append(False)

        enbpfl.append(width)
        enbpfl_ma.append(sum(enbpfl[-3:]) / len(enbpfl[-3:]))

        low_signal_sensitivity = normal_signal_sensitivity + 0.2
        predictEngineHelper = PredictEngineHelper()
        low_multiplication_factor = predictEngineHelper.get_multiplication_factor(forecast_statistical_parameter.projection_multiplication_factor, low_signal_sensitivity)
        lbpfl.append(low_multiplication_factor * forecast_statistical_parameter.error_of_standard_deviation)
        lbpfl_ma.append(sum(lbpfl[-3:]) / len(lbpfl[-3:]))

        medium_signal_sensitivity = low_signal_sensitivity + 0.2
        medium_multiplication_factor = predictEngineHelper.get_multiplication_factor(forecast_statistical_parameter.projection_multiplication_factor, medium_signal_sensitivity)
        mbpfl.append(medium_multiplication_factor * forecast_statistical_parameter.error_of_standard_deviation)
        mbpfl_ma.append(sum(mbpfl[-3:]) / len(mbpfl[-3:]))

    normal_upper_bounds = []
    normal_lower_bounds = []
    low_upper_bounds = []
    low_lower_bounds = []
    medium_upper_bounds = []
    medium_lower_bounds = []

    for i in range(len(ensemble_forecast_list)):
        forecast = ensemble_forecast_list[i]
        first_forecast = ensemble_forecast_list[i % period_count]

        normal_upper_bounds.append(forecast + nbpfl_ma[i % period_count])
        if is_normal_lower_bound_calibrated[i % period_count]:
            if forecast - nbpfl[i % period_count] < 0:
                if first_forecast - enbpfl_ma[i % period_count] < 0:
                    normal_lower_bounds.append(first_forecast - enbpfl[i % period_count])
                else:
                    if first_forecast - enbpfl_ma[i % period_count] > forecast + nbpfl_ma[i % period_count]:
                        if min(interprete_input[i % period_count]) / 2 < forecast + nbpfl_ma[i % period_count]:
                            normal_lower_bounds.append(min(interprete_input[i % period_count]) / 2)
                        else:
                            normal_lower_bounds.append(0)
                    else:
                        normal_lower_bounds.append(first_forecast - enbpfl_ma[i % period_count])
            else:
                normal_lower_bounds.append(forecast - nbpfl_ma[i % period_count])
        else:
            normal_lower_bounds.append(forecast - nbpfl_ma[i % period_count])

        low_upper_bounds.append(forecast + lbpfl_ma[i % period_count])
        low_lower_bounds.append(forecast - lbpfl_ma[i % period_count])

        medium_upper_bounds.append(forecast + mbpfl_ma[i % period_count])
        medium_lower_bounds.append(forecast - mbpfl_ma[i % period_count])

    return normal_upper_bounds, normal_lower_bounds, low_upper_bounds, low_lower_bounds, medium_upper_bounds, medium_lower_bounds

def handle_tdf_calibration(tdf_lower_bound: float, period_index: int, interprete_input: List[List[float]], 
                           predict_forecast_statistical_parameter: List[ForecastStatisticalParameter]) -> float:
    curr_min_interprete_input = min(interprete_input[period_index])
    curr_pearson_correlation_coefficient = predict_forecast_statistical_parameter[period_index].pearson_correlation_coefficient

    return curr_min_interprete_input if curr_min_interprete_input > 0 else (curr_min_interprete_input if curr_pearson_correlation_coefficient > 0 else 0) if curr_min_interprete_input > 0 else tdf_lower_bound

def tune_boundaries_internal(projection_upper_bounds: List[float], projection_lower_bounds: List[float], month_index: int, total_periods: int, 
                    tdf_upper_bound: float, tdf_lower_bound: float, forecasting_parameter: ForecastStatisticalParameter, 
                    cor_dispersion: List[float], ensemble_forecast_list: List[float]):
    if len(projection_upper_bounds) != len(projection_lower_bounds):
        raise ValueError("Bounds need to be of same length")

    correlation = abs(forecasting_parameter.pearson_correlation_coefficient)

    if forecasting_parameter.mean == 0 or forecasting_parameter.standard_deviation == 0:
        cor_disp = correlation
    else:
        coeff_of_variation = abs(forecasting_parameter.standard_deviation / forecasting_parameter.mean)
        if coeff_of_variation > 0.3:
            cor_disp = correlation
        else:
            cor_disp = correlation * (1 - coeff_of_variation)

    cor_disp = abs(cor_disp)
    cor_dispersion.append(cor_disp)

    def bound_tuning_formula(projection_bound: float, tdf_bound: float) -> float:
        return (projection_bound * cor_disp) + tdf_bound * (1 - cor_disp)

    tdf_boundary_diff = tdf_upper_bound - tdf_lower_bound

    for i in range(month_index, len(projection_upper_bounds), total_periods):
        projection_boundary_diff = projection_upper_bounds[i] - projection_lower_bounds[i]
        original_projection_upper_bound = projection_upper_bounds[i]
        original_projection_lower_bound = projection_lower_bounds[i]
        original_projection = ensemble_forecast_list[i]

        if projection_boundary_diff >= (1.32 * tdf_boundary_diff):
            continue

        projection_upper_bounds[i] = bound_tuning_formula(projection_upper_bounds[i], tdf_upper_bound)
        projection_lower_bounds[i] = bound_tuning_formula(projection_lower_bounds[i], tdf_lower_bound)

        if original_projection > projection_upper_bounds[i] or original_projection < projection_lower_bounds[i]:
            projection_upper_bounds[i] = original_projection_upper_bound
            projection_lower_bounds[i] = original_projection_lower_bound

def tune_boundaries(interprete_input: List[List[float]], normal_upper_bounds: List[float], normal_lower_bounds: List[float],
                    tdf_prediction_list: List[TDFPrediction], predict_forecast_statistical_parameter: List[ForecastStatisticalParameter],
                    ensemble_forecast_list: List[float]) -> List[float]:
    total_periods = len(tdf_prediction_list)

    cor_dispersion = []

    for i in range(total_periods):
        tdf_upper_bound = tdf_prediction_list[i].riskBoundary.upperRange
        tdf_lower_bound = tdf_prediction_list[i].riskBoundary.lowerRange
        if tdf_lower_bound < 0:
            tdf_lower_bound = handle_tdf_calibration(tdf_lower_bound, i, interprete_input, predict_forecast_statistical_parameter)
        forecasting_parameter = predict_forecast_statistical_parameter[i]
        tune_boundaries_internal(normal_upper_bounds, normal_lower_bounds, i, total_periods, tdf_upper_bound, tdf_lower_bound, forecasting_parameter, cor_dispersion, ensemble_forecast_list)

    return cor_dispersion

def fix_predicted_value_outside_of_bounds(predictions: List[float], normal_upper_bounds: List[float], normal_lower_bounds: List[float]):
    for i in range(len(predictions)):
        if predictions[i] <= normal_lower_bounds[i] or predictions[i] >= normal_upper_bounds[i]:
            # Brute force way. To do: We need to think for better approach where Predicted Value is shifted in adequate proportion
            planfulNativeStatisticsService = PlanfulNativeStatisticsService()
            predictions[i] = planfulNativeStatisticsService.get_mean([normal_lower_bounds[i], normal_upper_bounds[i]])

def transform_to_predictions(a_forecast: ForecastBundle) -> List[Prediction]:
    # prediction_list = List[Prediction]
    prediction_list = []

    prediction_count = len(a_forecast.predictions)

    for j in range(prediction_count):
        prediction = Prediction(
            predictedValue=a_forecast.predictions[j],
            riskBoundary=RiskBoundary(
                lowerRange=a_forecast.normal_lower_bounds[j],
                upperRange=a_forecast.normal_upper_bounds[j],
                lowLowerBound=a_forecast.low_lower_bounds[j],
                lowUpperBound=a_forecast.low_upper_bounds[j],
                mediumLowerBound=a_forecast.medium_lower_bounds[j],
                mediumUpperBound=a_forecast.medium_upper_bounds[j]
            )
        )
        prediction_list.append(prediction)

    return prediction_list

def get_accuracy_based_algorithm(bucketized_input: List[List[float]], flat_input_series_for_ssa: List[float], prediction_count: int, config: PredictV3EngineConfig) -> str:
    test_data_length = len(bucketized_input)
    data_for_prediction_engine_input = flat_input_series_for_ssa[:-test_data_length]
    test_data_for_accuracy = flat_input_series_for_ssa[-test_data_length:]

    predictCastingHelper = PredictCastingHelper();
    test_bucketized_input = predictCastingHelper.transform_to_bucketize_series(data_for_prediction_engine_input, test_data_length)
    config.DynamicEngineAlgorithm = 0

    projection_upper_bounds: List[float] = []
    projection_lower_bounds: List[float] = []
    cor_dispersion: List[float] = []

    interprete_input: List[List[float]] = []
    bucketize_input_series: List[List[float]] = []
    tdf_prediction_list: List[PredictV3EngineConfig.TDFPrediction] = []
    flat_input_series_for_ssa: List[float] = []
    predictions = []
    
    interpreterService = InterpreterService()
    interprete_input, tdf_prediction_list, bucketize_input_series, flat_input_series_for_ssa = interpreterService.validate_and_interprete_input(bucketized_input, config.IsConsiderTwoYearHistory)

    forecast_response, ssa_series_forecast, holtwinterprediction, predict_v2_prediction, tdf_predictions, is_same_for_all_values = get_forecast(
    bucketized_input, prediction_count, config, projection_upper_bounds, projection_lower_bounds, cor_dispersion,
    interprete_input, tdf_prediction_list, bucketize_input_series, flat_input_series_for_ssa
    )

    predictions = transform_to_predictions(forecast_response)
    
    algorithms_predictions = get_mapped_prediction_value(predictions, ssa_series_forecast.forecast, holtwinterprediction, predict_v2_prediction, data_for_prediction_engine_input, test_data_length)

    mape_v3 = mean_absolute_percentage_error(test_data_for_accuracy, algorithms_predictions.v3)
    mape_ssa = mean_absolute_percentage_error(test_data_for_accuracy, algorithms_predictions.ssa)
    mape_hw = mean_absolute_percentage_error(test_data_for_accuracy, algorithms_predictions.holt_linear)
    mape_v2 = mean_absolute_percentage_error(test_data_for_accuracy, algorithms_predictions.v2)

    accuracy_list = [
        (mape_v3, MLAlgorithm.PredictV3),
        (mape_v2, MLAlgorithm.PredictV2),
        (mape_hw, MLAlgorithm.HoltWinter),
        (mape_ssa, MLAlgorithm.SSA)
    ]
    
    return min(accuracy_list, key=lambda x: x[0])[1]

def get_mapped_prediction_value(user_predictions: List[StatisticalPrediction], ssa_series_forecast: List[float], holtwinter_prediction: List[float], predict_v2_prediction: List[float], data_for_prediction_engine_input: List[Optional[float]], test_data_length: int) -> EnginePredictions:
    extra_element_count = len(data_for_prediction_engine_input) % test_data_length
    output = []
    output_ssa = []
    output_hw = []
    output_v2 = []
    output_counter = 0

    for indexer in range(extra_element_count, len(user_predictions)):
        if output_counter >= test_data_length:
            break
        output.append(float(user_predictions[indexer].predictedValue or 0))
        output_ssa.append(float(ssa_series_forecast[(indexer - extra_element_count) % len(ssa_series_forecast)] or 0))
        output_hw.append(float(holtwinter_prediction[(indexer - extra_element_count) % len(holtwinter_prediction)] or 0))
        output_v2.append(float(predict_v2_prediction[(indexer - extra_element_count) % len(predict_v2_prediction)][0] or 0))
        output_counter += 1

    return EnginePredictions(v3=output, ssa=output_ssa, holt_linear=output_hw, v2=output_v2)

def update_predict_engine_weights(algorithm: str, config: PredictEngineWeightageConfig):
    if algorithm == MLAlgorithm.SSA:
        config.SSA_Weightage = 1.0
        config.V2_Weightage = 0.0
        config.HOLTWINTER_Weightage = 0.0
    elif algorithm == MLAlgorithm.HoltWinter:
        config.SSA_Weightage = 0.0
        config.V2_Weightage = 0.0
        config.HOLTWINTER_Weightage = 1.0
    elif algorithm == MLAlgorithm.PredictV2:
        config.SSA_Weightage = 0.0
        config.V2_Weightage = 1.0
        config.HOLTWINTER_Weightage = 0.0

def calculate_ensemble_forecast_based_on_algorithm(prediction_count: int, period_count: int, ssa_prediction: SSASeriesForecast, holt_winter_predictions: List[float], predict_forecast_list: List[List[float]], predict_engine_weightage_config: PredictEngineWeightageConfig, algorithm: str) -> List[float]:
    update_predict_engine_weights(algorithm, predict_engine_weightage_config)
    ensemble_forecast_list = []

    for j in range(prediction_count):
        for i in range(period_count):
            ensemble_forecast = (
                predict_engine_weightage_config.V2_Weightage * predict_forecast_list[i][j] +
                predict_engine_weightage_config.SSA_Weightage * ssa_prediction.forecast[(j * period_count) + i] +
                predict_engine_weightage_config.HOLTWINTER_Weightage * holt_winter_predictions[(j * period_count) + i]
            )
            ensemble_forecast_list.append(ensemble_forecast)
    
    return ensemble_forecast_list

def get_dynamic_engine_forecast(prediction_count: int, bucketize_input_series: List[List[float]], flat_input_series_for_ssa: List[float], period_count: int, predict_v3_engine_config: PredictV3EngineConfig, ssa_prediction: SSASeriesForecast, holt_winter_predictions: List[float], predict_forecast_list: List[List[float]]) -> List[float]:
    number_of_predictions = period_count * prediction_count
    ensemble_forecast_list = [0.0] * number_of_predictions

    try:
        ml_algorithm = get_accuracy_based_algorithm(bucketize_input_series, flat_input_series_for_ssa, prediction_count, predict_v3_engine_config)
        ensemble_forecast_list = calculate_ensemble_forecast_based_on_algorithm(prediction_count, period_count, ssa_prediction, holt_winter_predictions, predict_forecast_list, predict_v3_engine_config.PredictEngineWeightageConfig, ml_algorithm)
    except Exception as ex:
        # error logging
        print(f"Predict Projection Dynamic Engine selection failed with error: {ex}")

    return ensemble_forecast_list

# input_bucketized = [
#     [3628.735632, 3463.793103, 3463.793103],
#     [3298.850575, 3463.793103, 3463.793103],
#     [3628.735632, 3793.678161, 3793.678161],
#     [3628.735632, 3463.793103, 3463.793103],
#     [3463.793103, 3628.735632, 3628.735632],
#     [3628.735632, 3628.735632, 3628.735632],
#     [3793.678161, 3463.793103, 3463.793103],
#     [3463.793103, 3793.678161, 3793.678161],
#     [3628.735632, 3628.735632, 3628.735632],
#     [3628.735632, 3463.793103, 3463.793103],
#     [3463.793103, 3628.735632, 3628.735632],
#     [3793.678161, 3628.735632, 3628.735632]
# ]
# prediction_count = 1
# prediction_config = PredictV3EngineConfig(False, 6, False, PredictSameMonthOnMonthConfig(0.1, 0.2, 0.3),
#                                           PredictEngineWeightageConfig(0.25, 0.25, 0.5), HoltWinterConfig(0.8, 0.9, 0.01),
#                                           SSAForecastConfig(5, 10, 100), ProjectionViaHoltWinterConfig(0.7, 0.9, 
#                                           StatisticalDistributionConfig(StatisticalDistributionRiskBoundary(0.3, 0.8), StatisticalDistributionRiskBoundary(0.15, 0.95), StatisticalDistributionRiskBoundary(0.1, 0.99), 36, 2.5, 0.05, 0.1, 0.2, 50.0, True), 0.0, True, 0.8, 1.2))

# start_time = time.time()
# predictions = GetPredictions(input_bucketized, prediction_count, prediction_config)
# # predictions = GetPredictions([], prediction_count, prediction_config)
# end_time = time.time()

# for prediction in predictions:
#     print(f"{prediction.value}")
#     print(f"Prediction Value: {prediction.value}")
#     print(f"  Normal Lower Bound: {prediction.riskBoundary.lowerRange}")
#     print(f"  Normal Upper Bound: {prediction.riskBoundary.upperRange}")
#     print(f"  Low Lower Bound: {prediction.riskBoundary.lowLowerBound}")
#     print(f"  Low Upper Bound: {prediction.riskBoundary.lowUpperBound}")
#     print(f"  Medium Lower Bound: {prediction.riskBoundary.mediumLowerBound}")
#     print(f"  Medium Upper Bound: {prediction.riskBoundary.mediumUpperBound}")

 
#region Printing
# print("Normal Low Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.lowerRange)
# print("low Lower Bound")

# print("Normal Upper Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.upperRange)
# print("low Lower Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.lowLowerBound)
# print("Low Upper Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.lowUpperBound)
# print("Medium Lower Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.mediumLowerBound)
# print("Medium Upper Bound")
# for prediction in predictions:
#     # print(f"{prediction.value}")
#     print(prediction.riskBoundary.mediumUpperBound)

#endregion

# execution_time = end_time - start_time
# print(f"Execution Time: {execution_time:.4f} seconds")
