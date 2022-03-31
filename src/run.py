# Read config.ini
from read_config import *

# Load SQL to dataframe
from read_SQL import df_from_SQL
df = df_from_SQL(s_dbname, s_tablename)

# Clean data
from clean_data import clean_df
df = clean_df(df)

# Run pipeline
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

# Save plot as JPG in output folder
s_algo_dict = {
    1: 'Logistic Regression',
    2: 'Random Forest Classifier',
    3: 'Support Vector Machine',
    4: 'K-Nearest Neighbour'
}
import os
from datetime import datetime
dt_string = datetime.now().strftime("%Y-%m-%d %H%M%S") # for prefixing datetime to file name
filename = dt_string+' '+s_algo_dict[i_algo]+', AUC - '+s_auc+'.jpg' # AUC metric tag serves as a quick comparison between runs while seeing in the folder
parentdirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) # gets one directory up
outputdirectory = os.path.join(parentdirectory, 'output') # place JPGs in output folder one directory above where script is
fig.savefig(os.path.join(outputdirectory, filename), bbox_inches='tight', dpi=300)