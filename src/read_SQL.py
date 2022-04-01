import os
import sqlite3
import pandas as pd


def df_from_SQL(s_dbname, s_tablename, p_ls_features):
    """Load a SQL database into a pandas dataframe

    :param s_dbname: SQL database name
    :param s_tablename: table name within SQL database
    :return df: dataframe from SQL database
    """
    # Get relative path of database file (one directory up)
    dir = os.path.dirname(os.getcwd())

    # join directorys with ',' instead of '/' as we want to be OS-independant
    s_fullpath = os.path.join(dir, 'data', s_dbname)

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

    # Check that FEATURES entered in `config.ini` are present in SQL table
    df_columns_list = [*df.columns]
    for x in p_ls_features:  # iterate over list from `config.ini`
        if x not in df_columns_list:  # if element not present in SQL table
            raise ValueError('FEATURES is entered incorrectly, check again')

    return(df)
