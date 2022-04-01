# Read config.ini
from read_config import *

# Load SQL to dataframe
from read_SQL import df_from_SQL
df = df_from_SQL(s_dbname, s_tablename)

# Clean data
from clean_data import clean_df
df = clean_df(df)

# Run pipeline
from pipeline_classifier import *
model = pipeline_classifier(ls_features, df[ls_features], df['Survive'], i_algo, f_testsize, i_seed, True, True)
