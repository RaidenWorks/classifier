def clean_df(p_df):
    # we choose not to drop rows with the same ID, as we've seen from the EDA that they are different individuals

    # drop all rows with missing values
    p_df = p_df.dropna()

    # this specific Platelets value appears to be a data entry error, drop rows with this
    p_df = p_df[p_df.Platelets != 263358.03]

    # we make the assumption that recorded negative Age values are meant to positive
    p_df['Age'] = p_df['Age'].abs()

    # string values to be changed to integers
    p_df = p_df.replace({'Survive' : { '0' : 0, '1' : 1, 'No' : 0, 'Yes' : 1}})
    p_df = p_df.replace({'Gender' : { 'Male' : 1, 'Female' : 0}})
    p_df = p_df.replace({'Smoke' : { 'Yes' : 1, 'No' : 0, 'NO' : 0, 'YES' : 1}})
    p_df = p_df.replace({'Diabetes' : { 'Normal' : 0, 'Pre-diabetes' : 1, 'Diabetes' : 2}})
    p_df = p_df.replace({'Ejection Fraction' : { 'Low' : 0, 'Normal' : 1, 'High' : 2, 'L' : 0, 'N' : 1}})
    p_df = p_df.replace({'Favorite color' : { 'green' : 0, 'black' : 1, 'white' : 2, 'yellow' : 3, 'blue' : 4, 'red' : 5}})

    return(p_df)
