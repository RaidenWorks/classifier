from read_config import pop_config_values
from read_SQL import df_from_SQL
from clean_data import clean_df
from pipeline_classifier import pipeline_classifier

# Read config.ini
s_dbname, s_tablename, ls_features, i_algo, f_testsize, i_seed = \
    pop_config_values('config.ini')

# Load SQL to dataframe
df = df_from_SQL(s_dbname, s_tablename)

# Clean data
df = clean_df(df)

# Run pipeline
model = pipeline_classifier(ls_features, df[ls_features], df['Survive'],
                            i_algo, f_testsize, i_seed, True, True)
