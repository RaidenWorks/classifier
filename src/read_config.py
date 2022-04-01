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
    # s_dbname = o_config.get('DB', 'DBNAME')
    # s_tablename = o_config.get('DB', 'TABLENAME')

    # Get value of 'FEATURES' and generate a list
    s_features = o_parameters['FEATURES']
    # s_features = o_config.get('PARAM', 'FEATURES')

    # Get a list of features
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

    return(s_dbname, s_tablename, ls_features, i_algo, f_testsize, i_seed)
