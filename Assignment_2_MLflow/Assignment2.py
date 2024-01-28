# %% DATA LOADING
########################################################################################################################
# IMPORTS
# You absolutely need these
import math
import warnings
from influxdb import InfluxDBClient
import mlflow

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
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

days = 120 # -> You can change this to get any other range of days

power_set = client.query(
    "SELECT * FROM Generation where time > now()-"+str(days)+"d"
    ) # Query written in InfluxQL. We are retrieving all generation data from 90 days back.

wind_set  = client.query(
    "SELECT * FROM MetForecasts where time > now()-"+str(days)+"d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL. We are retrieving all weather forecast data from 90 days back and with 1 lead hour.

power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)

print(power_df)
print(wind_df)

# %% PREPROCESSING STEPS

#finding all unique direction values
distinct_values = wind_df['Direction'].unique().tolist()

#['SSE', 'SE', 'ESE', 'NE', 'NNE', 'NNW', 'NW', 'WNW', 'SW', 'W', 'WSW', 'SSW', 'N', 'E', 'ENE', 'S']

#Creating dictionary to get used on the direction column
conversion = {
    "SSE": 1,
    "SE": 2,
    "ESE": 3,
    "NE": 4,
    "NNE": 5,
    "NNW": 6,
    "NW": 7,
    "WNW": 8,
    "SW": 9,
    "W": 10,
    "WSW": 11,
    "SSW": 12,
    "N": 13,
    "E": 14,
    "ENE": 15,
    "S": 16
    }

def categorize1(string):
    if string in conversion:
        string = conversion[string]
    return string

#applying the defined dictionary on the direction column
wind_df['Direction'] = wind_df['Direction'].apply(
    lambda string: categorize1(string))

# resampling and using interpolation to handle the missing values, resampled to every 10th minute
resampled = wind_df.resample('10T').mean()

#Setting interpolated values to be made from assumed linear relationship between feature variables
interpolated = resampled.interpolate(method='linear')

#merging the interpolated wind df with the power. Now the power df joined on every 10th timestamp
#from the interpolated wind df
merged= pd.merge(power_df, interpolated, left_index=True, right_index=True, how='right')
merged['Direction'] = merged['Direction'].astype(int)
final_df = merged[['Speed', 'Direction', 'Total']]

print(final_df)

# %% MODEL SELECTION AND HYPERPARAMETERS

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
    mlflow.sklearn.save_model(best_model, "best_model_with_Intepolation")
    

# %% CREATING A NEW DF WITH EXISTING POWER GENERATED VALUES 
#
# CREATING a new df to make predictions on with actual power generated values
# afterwards i will create a "label" column with all predicted values
# finally i will calculate and show the prediction residuals as the "error margin" between actual and predicted values
#
joined_dfs = power_df.join(wind_df, how="inner")
joined_dfs = joined_dfs[['Speed','Direction', 'Total']]
joined_dfs['Direction'] = joined_dfs['Direction'].apply(
    lambda string: categorize1(string))
joined_dfs['Direction'] = joined_dfs['Direction'].astype(int)
joined_dfs.rename(columns={'Total' : 'Actual PowerGen'}, inplace=True)

print(joined_dfs)

#%% PREDICTION TRYOUT INTERPOLATION MODEL VS NO INTERPOLATION MODEL

###########################################
# FIRST THE MODEL WITH INTERPOLATED VALUES
###########################################
import mlflow
logged_model = 'best_model_with_Intepolation'

#lets load the best model trained on interpolated values
interpolated_model = mlflow.pyfunc.load_model(logged_model)

#use this model to make predicted power generated values
predictions = interpolated_model.predict(joined_dfs[['Speed', 'Direction']])

withInterpolation_df = joined_dfs
withInterpolation_df ['Predicted PowerGen'] = predictions

#now, lets take the absolute value of substracting actual and predicted values to get the RESIDUAL value
withInterpolation_df['Prediction Residuals'] = abs(withInterpolation_df['Actual PowerGen'] - withInterpolation_df['Predicted PowerGen'])

print(withInterpolation_df)

#storing it as a csv file for better visual representation
#withInterpolation_df.to_csv('Residuals_with_interpolation.csv')

###########################################
# NOW THE MODEL WITHOUT INTERPOLATION
###########################################

logged_model = 'best_model_without_interpolation'

#lets load the best model trained on non-interpolated values
no_interpolation_model = mlflow.pyfunc.load_model(logged_model)

#use this model to make predicted power generated values
predictions = no_interpolation_model.predict(joined_dfs[['Speed', 'Direction']])

noInterpolation_df = joined_dfs

noInterpolation_df['Predicted PowerGen'] = predictions

#now, lets take the absolute value of substracting actual and predicted values to get the RESIDUAL value
noInterpolation_df['Prediction Residuals'] = abs(noInterpolation_df['Actual PowerGen'] - noInterpolation_df['Predicted PowerGen'])

print(noInterpolation_df)

#storing it as a csv file for better visual representation
#noInterpolation_df.to_csv('Residuals_without_interpolation.csv')

# %% Prediction tryout on newest forecasts with interpolated model

#all future forecasts regardless of lead time
forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()")

#transform result set into dataframe
forecasts = set_to_dataframe(forecasts)

#Only get the newest source time -> latest only
newest_forecasts = forecasts.loc[forecasts["Source_time"] == forecasts["Source_time"].max()].copy()

newest_forecasts['Direction'] = newest_forecasts['Direction'].apply(
    lambda string: categorize1(string))
newest_forecasts['Direction'] = newest_forecasts['Direction'].astype(int)
newest_forecasts.head()

#lets load our model with interpolated values
logged_model = 'best_model_with_Intepolation'

model = mlflow.pyfunc.load_model(logged_model)

# using the saved model to predict the power generation
predictions = model.predict(newest_forecasts[['Speed', 'Direction']])

#at last, transform it into a dataframe 
predictions = pd.DataFrame(predictions, index=newest_forecasts.index, columns=["Predicted Power Production"])
predictions.head()

# %% EDA - PowerGeneration meter plot

plt.figure(figsize=(12, 5))

plt.scatter(final_df['Direction'], final_df['Speed'], c=final_df['Total'], cmap='coolwarm')
plt.colorbar(label='Energy Production')
plt.xlabel('Wind Direction')
plt.ylabel('Speed')
plt.title('Wind Direction and Speed (With EnergyProduction Meter)')
plt.show()

######### power generation curve 

plt.figure(figsize=(8, 6))
plt.scatter(joined_dfs['Speed'], joined_dfs['Actual PowerGen'], c=joined_dfs['Direction'],  cmap='coolwarm')
plt.colorbar(label='Direction')
plt.xlabel('Speed')
plt.ylabel('Actual Power Generation')
plt.title('Relationship between Speed and Actual PowerGen')
plt.grid(True)
plt.show()



# %% EDA - CORRELATION MATRIX

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = joined_dfs[["Speed", "Direction", "Actual PowerGen"]].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# %%
