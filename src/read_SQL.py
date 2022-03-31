import os
import sqlite3
import pandas as pd

def df_from_SQL(s_dbname, s_tablename):
    # Get relative path of database file
    dir = os.path.dirname(os.getcwd()) # gets path one directory up
    s_fullpath = os.path.join(dir, 'data', s_dbname) # join subdirectorys with ',' instead of '/' as we want to be OS-independant

    # Connection object
    o_conn = sqlite3.connect(s_fullpath)

    # Create dataframe from database
    s_SQLquery = 'SELECT * FROM ' + s_tablename

    # Cursor object (for executing SQL queries against database)
    # cur = o_conn.cursor()

    # List table names
    # cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cur.fetchall())

    # Generate the dataframe
    df = pd.read_sql_query(s_SQLquery, o_conn)

    return(df)

# import configparser
# o_config = configparser.ConfigParser()
# o_config.read('config.ini')
# o_database = o_config['DB']
# o_parameters = o_config['PARAM']
# s_dbname = o_database['DBNAME']
# s_tablename = o_database['TABLENAME']

# print(df_from_SQL(s_dbname, s_tablename).head)
