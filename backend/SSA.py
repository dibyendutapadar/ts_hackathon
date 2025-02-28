import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from backend.PredictV3EngineConfig import SSAForecastConfig, SSASeriesForecast
import xgboost as xgb

class SSAForecastModel:
    def __init__(self):
        pass

    def forecast(self, predictionCount, flatInput, sSAForecastConfig:SSAForecastConfig):
        flatInput = np.array(flatInput)

        trainSize = sSAForecastConfig.TrainSize
        windowSize = sSAForecastConfig.WindowSize
        seriesLength = sSAForecastConfig.SeriesLength

        # Prepare the training data with windows of windowSize
        X_train = []
        y_train = []
        X_train = np.lib.stride_tricks.sliding_window_view(flatInput, windowSize)[:trainSize - windowSize]
        y_train = flatInput[windowSize:trainSize]
       
       # Ensure X_train and y_train have the same number of samples
        if len(X_train) != len(y_train):
            X_train = X_train[:len(y_train)]
        # Create the SSA forecast model
        # model = SVR(kernel='linear', C=0.1, epsilon=0.1)

        # Train the model

        #below code is to Initialize the XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,  # Number of trees
            learning_rate=0.1,  # Learning rate
            max_depth=3,  # Maximum depth of a tree
            subsample=0.8,  # Subsample ratio of the training instances
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
            n_jobs=-1,  # Use all available cores
            random_state=42  # Random seed for reproducibility
        )
        model.fit(X_train, y_train)

        # Make predictions
        predictions = []

        # Ensure trainSize and windowSize are within the bounds of flatInput
        if trainSize > len(flatInput):
            trainSize = len(flatInput)

        if trainSize - windowSize < 0:
            windowSize = trainSize
        testData = flatInput[trainSize - windowSize:trainSize].tolist()

        for i in range(predictionCount):
            # Use the last windowSize values as input for the next prediction
            input = np.array(testData[-windowSize:]).reshape(1, -1)

            # Predict the next value
            prediction = model.predict(input)

            # Append the prediction to the list
            predictions.append(prediction[0])

            # Add the prediction to the test data for the next iteration
            testData.append(prediction[0])

        # # Calculate the root mean squared error
        # rmse = np.sqrt(mean_squared_error(flatInput[trainSize:trainSize + predictionCount], predictions))

        return SSASeriesForecast(predictions)