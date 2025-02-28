from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

class LinearExponentialSmoothingModel:
    def __init__(self, Alpha, Beta, Phi):
        self.Alpha = Alpha
        self.Beta = Beta
        self.Phi = Phi

    def forecast(self, data, forecast_periods):
        # Initialize the model with a linear trend and damping
        model = ExponentialSmoothing(
            data,
            trend="add",
            damped_trend=True,
            seasonal=None,            
        )
        
        # Fit the model with specified parameters
        fit = model.fit(smoothing_level=float(self.Alpha), smoothing_trend=float(self.Beta), damping_trend=float(self.Phi))
        
        # Generate in-sample predictions (fitted values, yhat)
        yhat_values = fit.fittedvalues.tolist()
        
        # Generate forecast for future periods
        future_forecast = fit.forecast(steps=forecast_periods).tolist()
        
        # return yhat_values, future_forecast
        return future_forecast,yhat_values 