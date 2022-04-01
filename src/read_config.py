import configparser


def pop_config_values(p_ini_configfile):
    """Read config.ini and assign to variables

    :return (s_dbname, s_tablename, ls_features, i_algo, f_testsize, i_seed):
    """
    # Create config instances
    o_config = configparser.ConfigParser()
    o_config.read(p_ini_configfile)
    o_database = o_config['DB']
    o_parameters = o_config['PARAM']

    # Get database and table name from configuration file
    s_dbname = o_database['DBNAME']
    s_tablename = o_database['TABLENAME']

    # Get value of 'FEATURES' and generate a list
    s_features = o_parameters['FEATURES']

    # Get a list of features
    ls_features = s_features.split(",")

    # Get value of 'ALGO'
    i_algo = int(o_parameters['ALGO'])
    if i_algo < 1 or i_algo > 4:
        raise ValueError('ALGO must be integer 1, 2, 3, or 4')

    # Get value of 'TESTSIZE'
    f_testsize = float(o_parameters['TESTSIZE'])
    if f_testsize < 0 or f_testsize > 1:
        raise ValueError('TESTSIZE must be a value between 0 and 1')

    # Get value of 'SEED'
    i_seed = int(o_parameters['SEED'])

    return(s_dbname, s_tablename, ls_features, i_algo, f_testsize, i_seed)

# Notes: alternate way of getting o_config keys
# s_dbname = o_config.get('DB', 'DBNAME')
# s_tablename = o_config.get('DB', 'TABLENAME')
# s_features = o_config.get('PARAM', 'FEATURES')
# i_algo = o_config.getint('PARAM', 'ALGO')
# f_testsize = o_config.getfloat('PARAM', 'TESTSIZE')
# i_seed = o_config.getint('PARAM', 'SEED')
