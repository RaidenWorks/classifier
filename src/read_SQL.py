import os
import sqlite3
import pandas as pd


def df_from_SQL(s_dbname, s_tablename):
    """Load a SQL database into a pandas dataframe
    
    :param s_dbname: SQL database name
    :param s_tablename: table name within SQL database
    :return df: dataframe from SQL database
    """
    # Get relative path of database file
    dir = os.path.dirname(os.getcwd())  # gets path one directory up
    s_fullpath = os.path.join(dir, 'data', s_dbname)  # join subdirectorys with ',' instead of '/' as we want to be OS-independant

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