################ Read config.ini and load SQL data ################
from readConfig_loadData import *
# print('s_dbname:',s_dbname)
# print('s_tablename:',s_tablename)
# print('s_features:',s_features)
# print('i_algo:',i_algo)
# print('f_testsize:',f_testsize)
# print('i_seed:',i_seed)

################ Load SQL data ################
import os
import sqlite3

# Get relative path
dir = os.path.dirname(os.getcwd()) # gets path one directory up
fullpath = os.path.join(dir, 'data', s_dbname) # join subdirectorys with ',' instead of '/' as we want to be OS-independant

# Connection object
o_conn = sqlite3.connect(fullpath)

# Cursor object (for executing SQL queries against database)
cur = o_conn.cursor()

# List table names
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cur.fetchall())

################ Load to dataframe ################
import pandas as pd
# s_tablename = 'survive' # change table name accordingly if not 'survive'
s_SQLquery = 'SELECT * FROM ' + s_tablename
df = pd.read_sql_query(s_SQLquery, o_conn) # create dataframe

################ Clean data ################
df = df.dropna() # drop all rows with missing values
# we choose not to drop rows with the same ID, as we've seen from the EDA that they are different individuals
df = df.replace({'Survive' : { '0' : 0, '1' : 1, 'No' : 0, 'Yes' : 1}})
df = df.replace({'Gender' : { 'Male' : 1, 'Female' : 0}})
df = df.replace({'Smoke' : { 'Yes' : 1, 'No' : 0, 'NO' : 0, 'YES' : 1}})
df = df.replace({'Diabetes' : { 'Normal' : 0, 'Pre-diabetes' : 1, 'Diabetes' : 2}})
df['Age'] = df['Age'].abs() # assume that recorded negative values are meant to positive
df = df.replace({'Ejection Fraction' : { 'Low' : 0, 'Normal' : 1, 'High' : 2, 'L' : 0, 'N' : 1}})
df = df[df.Platelets != 263358.03] # this admittedly is rather specific
df = df.replace({'Favorite color' : { 'green' : 0, 'black' : 1, 'white' : 2, 'yellow' : 3, 'blue' : 4, 'red' : 5}})

################ Run pipeline ################
# Pipeline
    # 1. Separate features and labels
    # 2. Split data into training set and test set
    # 3. Select algorithm
    # 4. Generate string list for Categorical Features from p_ls_features
    # 5. Generate string list for Numeric Features from the difference
    # 6. Generate a dynamic dictionary that updates the values when different sets of features are selected
    # 7. Generate integer list for Numeric Features (needed for scaling pre-processing)
    # 8. Generate integer list for Categorical Features (needed for one hot encoding pre-processing)
    # 9. Define preprocessing for numeric columns (make them on the same scale)
    # 10. Define preprocessing for categorical features (encode them)
    # 11. Combine preprocessing steps
    # 12. Create preprocessing and training pipeline
    # 13. Fit the pipeline to train a logistic regression model on the training set
    # 14. Get predictions from test data
    # 15. Calculate ROC curve
    # 16. Format plots arrangement
    # 17. Text display of selected Features shown in 1st subplot
    # 18. Print input parameters in 1st subplot
    # 19. Plot ROC curve in 2nd subplot
    # 20. Plot Confusion Matrix in 3rd subplot
    # 21. Print metrics in 4th subplot

from pipeline_Classifier import *
fig, s_auc = pipeline_classifier(ls_features, df[ls_features], df['Survive'], i_algo, f_testsize, i_seed)

################ Save plot as JPG in output folder ################
import matplotlib.pyplot as plt
s_algo_dict = {
  1: 'Logistic Regression',
  2: 'Random Forest Classifier',
  3: 'Support Vector Machine',
  4: 'K-Nearest Neighbour'
}
from datetime import datetime
import os
dt_string = datetime.now().strftime("%Y-%m-%d %H%M%S") # for prefixing datetime to file name
filename = dt_string+' '+s_algo_dict[i_algo]+', AUC - '+s_auc+'.jpg' # AUC metric tag serves as a quick comparison between runs while seeing in the folder
parentdirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) # gets one directory up
outputdirectory = os.path.join(parentdirectory, 'output') # place JPGs in output folder one directory above where script is
fig.savefig(os.path.join(outputdirectory, filename), bbox_inches='tight', dpi=300)