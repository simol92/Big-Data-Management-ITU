# %% DATA LOADING
########################################################################################################################
# IMPORTS
# You absolutely need these
import warnings
from influxdb import InfluxDBClient
import mlflow

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid

########################################################################################################################

## Step 1: The Data

### Getting the data with InfluxDB

"""
The data is stored in an InfluxDB (https://www.influxdata.com/), which is a non-relational time-series database. InfluxDB can be queried using InfluxQL
(https://docs.influxdata.com/influxdb/v1.8/query_language/spec/), a "SQL-like" query language for time-series data. InfluxDB does not have tables with rows and columns,
instead data is stored in measurements with fields and tags.

NOTE: You don't need to know much about InfluxDB syntax, but if you are interested, feel free to browse around the documentation (https://docs.influxdata.com/).
The data for this assignment is stored in a database, with one table for the weather data and another for the power generation data. To do this, we first need to
create an instance of the InfluxDB Client, that will allow us to query the needed data. Let's see how this is done.

"""

# Set the needed parameters to connect to the database
### THIS SHOULD NOT BE CHANGED ###
settings = {
    'host': 'influxus.itu.dk',
    'port': 8086,
    'username': 'lsda',
    'password': 'icanonlyread'
    }

# Create an InfluxDB Client instance and select the orkney database
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
client = InfluxDBClient(host=settings['host'], port=settings['port'], username=settings['username'], password=settings['password'])
client.switch_database('orkney')

## Function to tranform the InfluxDB resulting set into a Dataframe
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
def set_to_dataframe(resulting_set):
    
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index

    return df

days = 500 # -> You can change this to get any other range of days

power_set = client.query(
    "SELECT * FROM Generation where time > now()-"+str(days)+"d"
    ) # Query written in InfluxQL. We are retrieving all generation data from 90 days back.

wind_set  = client.query(
    "SELECT * FROM MetForecasts where time > now()-"+str(days)+"d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL. We are retrieving all weather forecast data from 90 days back and with 1 lead hour.

power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)
# ########
# wind_df_without_direction = wind_df.drop(columns=['Direction'])
# resampled_wind_df = wind_df_without_direction.resample('30T').mean()
# interpolated_wind_df = resampled_wind_df.interpolate(method='linear')
# final_wind_df = pd.concat([interpolated_wind_df, wind_df['Direction']], axis=1)
# merged_df = pd.merge(power_df, final_wind_df, left_index=True, right_index=True, how='right')
# ########


# %% PREPROCESSING STEPS
joined_dfs = power_df.join(wind_df, how="inner")

final_df = joined_dfs

###################################################
# defining estimators / models and their respective hyperparameter options
estimator_params = {
    "LR": {
        # calculate the intercept for this model
        "fit_intercept": [True, False],
        # normalize the feature variables
        "normalize": [True, False],   
        # copy the input data to ensure integrity
        "copy_X": [True, False]        
    },
    "SVR": {
        #linear for linear data and rbf for nonlinear 
        "kernel": ['linear', 'rbf'],  
        #balances the model’s prioritization between fitting the training data versus wider margin
        "C": [0.1, 1, 10],
        #determines the importance of a single training data sample.
        "gamma": ['scale', 'auto'],
        #specification of tolerance where no penalty is given to errors
        "epsilon": [0.1, 0.2, 0.5], 
        #allows the use of certain heuristics in terms of skipping samples in training iterations
        "shrinking": [True, False],
        #if the convergence criteria is not met within this limit, then the process stops
        "max_iter": [1000, 2000, 3000]       
    }
}

#dropping rows with NaN values
final_df = final_df.dropna()

#setting the input columns from feature variables / independent variables into a set X
X = final_df[["Speed", "Direction"]]  # Independent variables
#setting the output column from the target variable / dependent variable into a set Y
y = final_df["Total"] 

# evaluating differet metrics
def evalMetrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)  
    mae = mean_absolute_error(actual, predicted) 
    r2 = r2_score(actual, predicted)
    #setting the r2 to 100 (penalty) by default
    penalty = 100
    #if the models r2 score is below 0, then give a high penalty of 100 to indicate overfitting!
    if r2 < 0:
        r2 = penalty
    return mse, mae, r2


# %% MLFLOW - TRYING DIFFERENT MODELS AND HYPERPARAMETERS

#autolog each metric and parameter
mlflow.sklearn.autolog()
#setting local mlflow server to track all the different models from local host
mlflow.set_tracking_uri("http://127.0.0.1:5000")
#experiment name
mlflow.set_experiment("with interpolation")

#dictionary to store best models
best_models = {} 

#total runs that have happended in the mlflowproject
totalRuns= 1

#looping through the parameter grids for each estimator
for name, grid in estimator_params.items():
    runNumber = 1
    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, []) 
    ]
    #setting up a dictionary for the pipeline to loop through each model, also to avoid error when inserting estimators as strings
    classes = {
        "LR": LinearRegression,
        "SVR": SVR
    }[name] #get the non-string name for the estimator
    
    #creating the necessary variables to capture the best model and the initial values 
    best_score = 10000000000000.0 #setting it very high as the best model has the lowest
    best_params = None 
    best_model = None
    
    #loop for EACH combination of hyperparameters for each model
    #using parametergrid to create a new grid for the hyperparameters and all its values
    for params in ParameterGrid(grid):
        with mlflow.start_run(run_name=f"{name}_model_run_{runNumber}_total_runs_{totalRuns}"):
            warnings.filterwarnings("ignore")
            
            # first step, create a pipeline with each estimators and its different combinations of hyperparameters
            pipeline = Pipeline([('estimator', classes(**params))])
            
            # number of time our entire dataset will get splitted to try and get different metric values
            number_of_splits = 5
            # list to store metric scores for overall mean value score
            scores = []
            
            # creating training and test set for each split
            for train, test in TimeSeriesSplit(number_of_splits).split(X, y):
                #fit training data and test data on the pipeline
                pipeline.fit(X.iloc[train], y.iloc[train]) 
                #making predictions on test data
                predictions = pipeline.predict(X.iloc[test])
                #actual powergenerated data from the test set
                truth = y.iloc[test]
                
                #for each metric, calculate the score and store it in the scores list
                for metric_name, func, _ in metrics:
                    score = (func(truth, predictions))
                    #if the R2 score is below 0, then give it a penalty as its indicating overfitting!
                    if score < 0 and metric_name == "R2":
                        scores.append(100) 
                    #else if everythings okay, store the score
                    else:
                        scores.append(score)
                    
                
            #take the sum of the list with scores and divide it with the splits
            #this will be the models overall mean score
            mean_score = sum(scores) / number_of_splits
            mlflow.log_metric("Mean_score", mean_score)
            
            #across aaaaaaall the models, which model is the best for further testing?
            if mean_score < best_score:
                best_score = mean_score 
                best_params = params 
                best_model = pipeline
            runNumber += 1
            totalRuns += 1
            
    #storing the absolute best model and its associated score in the dictionary
    best_models[name] = {"best_model": best_model, "best_score": best_score, "best_params": best_params}

#finding the absolute best model across all of the scores and lets use it below for a seperate run on the entire dataset!
best_model_name = min(best_models, key=lambda x: best_models[x]["best_score"])
best_model = best_models[best_model_name]["best_model"]
best_score = best_models[best_model_name]["best_score"]
best_params = best_models[best_model_name]["best_params"]

# %%BEST MODEL TRYOUT

with mlflow.start_run(run_name=f"{best_model_name}_best_model"):
    # logging the best model and its parameters
    mlflow.sklearn.log_model(best_model, f"{best_model_name}_best_model")
    mlflow.log_params(best_params)
    mlflow.log_metric("Mean_score", best_score)
    
    # splitting the data for the final evaluation of the best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)
    
    # using the best estimator and also the eval function to use the class name as a string
    #also, use the best stored parameters for that model
    pipeline = Pipeline([('estimator', eval(best_model_name)(**best_params))])
    pipeline.fit(X_train, y_train)
    
    #ealuating the best model logging the metrics
    mse, mae, r2 = evalMetrics(pipeline.predict(X_test), y_test)
    mlflow.log_metric("Final_run_Mean_Squared_Error", mse)
    mlflow.log_metric("Final_run_Mean_Absolute Error", mae)
    mlflow.log_metric("Final_run_R2_Score", r2)
    #saving the best model
    mlflow.sklearn.save_model(best_model, "best_model_noIntepolation_dataset")
    
