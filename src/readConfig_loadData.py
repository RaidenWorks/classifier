# Purpose:   1. Read 'config.ini' and assign variables
#            2. Read database and load to a dataframe
# Inputs:    /src/config.ini
#            /data/*.db
# Return:    Assigns globals - s_dbname, s_tablename, s_features, ls_features, i_algo, f_testsize, i_seed, df
# Examples:  None
# Notes:     Amend the 'config.ini' according to your database 

import configparser
o_config = configparser.ConfigParser()
o_config.read('config.ini')
o_database = o_config['DB']
o_parameters = o_config['PARAM']

# Get database and table name from configuration file
s_dbname = o_database['DBNAME']
s_tablename = o_database['TABLENAME']

# Get value of 'FEATURES' and generate a list
s_features = o_parameters['FEATURES']
ls_features = s_features.split(",")

# Get value of 'ALGO'
i_algo = int(o_parameters['ALGO'])
# i_algo = o_config.getint('PARAM', 'ALGO')

# Get value of 'TESTSIZE'
f_testsize = float(o_parameters['TESTSIZE'])
# f_testsize = o_config.getfloat('PARAM', 'TESTSIZE')

# Get value of 'SEED'
i_seed = int(o_parameters['SEED'])
# i_seed = o_config.getint('PARAM', 'SEED')

# print('s_dbname:', s_dbname)
# print('s_tablename:', s_tablename)
# print('s_features:', ls_features)
# print('i_algo:', i_algo)
# print('f_testsize:', f_testsize)
# print('i_seed:', i_seed)

# Get relative path of database file
import os
dir = os.path.dirname(os.getcwd()) # gets path one directory up
s_fullpath = os.path.join(dir, 'data', s_dbname) # join subdirectorys with ',' instead of '/' as we want to be OS-independant

# Connection object
import sqlite3
o_conn = sqlite3.connect(s_fullpath)

# Create dataframe from database
import pandas as pd
s_SQLquery = 'SELECT * FROM ' + s_tablename
df = pd.read_sql_query(s_SQLquery, o_conn)
