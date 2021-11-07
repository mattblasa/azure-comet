# Databricks notebook source
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

#Comet Experiments 
from comet_ml import Experiment

#preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, LabelEncoder

# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  

# COMMAND ----------

# DBTITLE 1,Load Diamond Dataset from Azure Examples
diamond = pd.read_csv("/dbfs/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")

# COMMAND ----------

diamond

# COMMAND ----------

diamond = diamond.drop(columns='Unnamed: 0')

# COMMAND ----------

diamond.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Experiment Logging

# COMMAND ----------

experiment = Experiment(
    api_key="Insert COMET API Key here",
    project_name="azure-and-comet",
    workspace="mattblasa",
    log_code = False
)
#experiment.add_tag("Data Profile - AZDB")
experiment.add_tag("Azure-Databricks")
experiment.log_html_url("https://github.com/mattblasa/azure-comet", label='Github')

# COMMAND ----------

# MAGIC %md 
# MAGIC # EDA

# COMMAND ----------

# DBTITLE 1,Logging Seaborn Figure
def log_SeaFigure(fig, fig_name):
    '''
    Logs the seaborn figure, first by using depreciated ax.fig, and runs ax.figure if an exception is raised. 
    
    Parameters: 
    fig (object) - seaborn figure
    fig_name (string) - the user defined name for seaborn figure in comet experiment 
    
    Returns: 
    Logs figure to comet experiment log, and prints the method used or an error message. 
    
    '''
    ax = fig
    try:
      experiment.log_figure(fig_name, ax.fig)
      print('Log Figure Successful using ax.fig')
    except: 
      experiment.log_figure(fig_name, ax.figure)
      print('Log Figure Successful using ax.figure')
    
    


# COMMAND ----------

# MAGIC %md 
# MAGIC The log_seaFigure method was added since seaborn plots aren't loaded into experiments as easily as matplotlib generated figures. 

# COMMAND ----------

# MAGIC %md 
# MAGIC # Bivariate Statistics

# COMMAND ----------

ax = sns.heatmap(diamond.corr(),annot=True ,cmap="YlGnBu")

log_SeaFigure(ax, "Diamond Correlation")

# COMMAND ----------

#You will need to sample if the data set is larger. 
pairplt = sns.pairplot(data=diamond, corner=True)
log_SeaFigure(pairplt, "Pairplot Diamond")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Carat 

# COMMAND ----------

ax3 = sns.relplot(x='carat',y='price',hue='cut',data=diamond)
log_SeaFigure(ax3, "carat_price")

# COMMAND ----------


plt.plot(diamond['carat'], diamond['price'], '.')
plt.xlabel('carat')
plt.ylabel('price')
experiment.log_figure(figure=plt)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Cut

# COMMAND ----------

fig=plt.figure(figsize=(18,6))
sns.boxplot(x="cut",y="price",data=diamond)
plt.title("Cut versus Price")
plt.show()


ax3 = sns.boxplot(x="cut",y="price",data=diamond)
experiment.log_figure('cut_price', ax3.figure)

# COMMAND ----------

sns.violinplot(x="cut",y="price",data=diamond)

ax3 = sns.violinplot(x="cut",y="price",data=diamond)
experiment.log_figure('cut_price_violin', ax3.figure)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Feature Engineering Section

# COMMAND ----------

diamond['cut'] = diamond['cut'].replace({'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4})
diamond['color'] = diamond['color'].replace({'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
diamond['clarity'] = diamond['clarity'].replace({'I1':0, 'SI1':1, 'SI2':2, 'VS1':3, 'VS2':4, 'VVS1':5, 'VVS2':6, 'IF':7})


# COMMAND ----------

# DBTITLE 1,Drop Length (x), width (y), height (z)
diamond.drop(['x','y','z'], axis=1, inplace= True)

# COMMAND ----------

diamond

# COMMAND ----------

# DBTITLE 1,Convert Pandas Dataframe to SparkDF
spark_diamond = spark.createDataFrame(diamond)
display(diamond)

# COMMAND ----------

display(spark_diamond)

# COMMAND ----------

# MAGIC %md
# MAGIC # Building Models Using Spark

# COMMAND ----------

# DBTITLE 1,Vectorize Fetures 
from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table'], outputCol = 'features')
vdiamond_df = vectorAssembler.transform(spark_diamond)
vdiamond_df.take(1)



# COMMAND ----------

display(vdiamond_df)

# COMMAND ----------




vdiamond_df.show(3)



# COMMAND ----------

display(vdiamond_df)

# COMMAND ----------



splits = vdiamond_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# COMMAND ----------

# DBTITLE 1,Regression Utilities
def get_SparkMetric(labelCol, predCol, metricName, dfPrediction):
    '''
    Returns the a user-specified statistical metric for non-linear regression
    
    Parameters: 
    labelCol (str) - target column of a Spark Regression 
    predCol (str) - predicted values of regression 
    metricName (str) - metric used for model, such as RMSE, MAE, R2, MSE
    dfPrediction (obj) - transformed dataframe from test data 
    
    Returns: 
    Metric value for regression 
    
    '''
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol=predCol, metricName=metricName)
    metric = evaluator.evaluate(dfPrediction)
    return metric 

# COMMAND ----------

# DBTITLE 1,Linear Regression
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)


# COMMAND ----------

lr_trainingSummary = lr_model.summary
lr_r2 = lr_trainingSummary.r2
lr_mse = lr_trainingSummary.meanSquaredError
lr_rmse = lr_trainingSummary.rootMeanSquaredError
lr_mae = lr_trainingSummary.meanAbsoluteError

# COMMAND ----------

experiment.log_metric("LR_r2", lr_r2, step=0)
experiment.log_metric("LR_mse", lr_mse, step=0)
experiment.log_metric("LR_rmse", lr_rmse, step=0)
experiment.log_metric("LR_mae", lr_mae, step=0)

# COMMAND ----------


print("RMSE: %f" % lr_r2)
print("MSE = %s" % lr_mse)
print("r2: %f" % lr_rmse)
print("MAE = %s" % lr_rmse)

# COMMAND ----------

# DBTITLE 1,Decision Tree Regression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'price')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

# COMMAND ----------

get_SparkMetric("price", "prediction", "r2", dt_predictions)

# COMMAND ----------

dt_r2 = get_SparkMetric("price", "prediction", "r2", dt_predictions)
dt_rmse = get_SparkMetric("price", "prediction", "rmse", dt_predictions)
dt_mae = get_SparkMetric("price", "prediction", "mae", dt_predictions)
dt_mse = get_SparkMetric("price", "prediction", "mse", dt_predictions)

experiment.log_metric("DT_r2", dt_r2, step=0)
experiment.log_metric("DT_rmse", dt_rmse, step=0)
experiment.log_metric("DT_mae", dt_mae, step=0)
experiment.log_metric("DT_mse", dt_mse, step=0)

# COMMAND ----------

# DBTITLE 1,Gradient Boosted Regression
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'price', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)


# COMMAND ----------

gbt_predictions.select('prediction', 'price', 'features').show(5)

# COMMAND ----------

gbt_r2 = get_SparkMetric("price", "prediction", "r2", gbt_predictions )
gbt_rmse = get_SparkMetric("price", "prediction", "rmse", gbt_predictions)
gbt_mae = get_SparkMetric("price", "prediction", "mae", gbt_predictions)
gbt_mse = get_SparkMetric("price", "prediction", "mse", gbt_predictions)

experiment.log_metric("GBT_r2", gbt_r2, step=0)
experiment.log_metric("GBT_rmse", gbt_rmse, step=0)
experiment.log_metric("GBT_mae", gbt_mae, step=0)
experiment.log_metric("GBT_mse", gbt_mse, step=0)

print("r2: %f" % gbt_r2)
print("MSE = %s" % gbt_mse)
print("RMSE: %f" % gbt_rmse)
print("MAE = %s" % gbt_rmse)

# COMMAND ----------

experiment.end()

# COMMAND ----------


