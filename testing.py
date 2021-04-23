def preprocess(df, transformers={}):
    """
    Preprocess the dataframe to better use in the model.

    Args:
        df ([type]): [description]
        transformers (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    
    df_wrangling = df.copy()

    useless_cols = df_wrangling.isna().mean()[df_wrangling.isna().mean() > 0.3].index.tolist()
    useless_cols.append('Id')
    df_wrangling.drop(columns=useless_cols, inplace=True)

    # Dropping NA's
    df_wrangling.dropna(inplace=True)

    # One Hot Enconding
    categorical_variables = df_wrangling.select_dtypes(exclude = ['float64', 'int64'])
    
    try:
        enc = transformers['encoder']
    except:
        enc = OneHotEncoder()
    enc.fit(categorical_variables)
    
    df1 = enc.transform(categorical_variables).toarray()
    df1 = pd.DataFrame(df1, columns=enc.get_feature_names())
    df2 = df_wrangling.drop(columns=categorical_variables.columns)
    df_wrangling = pd.concat([df1, df2], axis=1)

    # Scaling 
    try:
        scaler = transformers['scaler']
    except:
        scaler = MinMaxScaler()
    scaler.fit(df_wrangling)
    df_wrangling = pd.DataFrame(scaler.transform(df_wrangling),
                                columns=df_wrangling.columns)

    return df_wrangling, {'scaler': scaler, 'encoder': enc}