from sklearn.model_selection import GroupShuffleSplit


def train_test_split(data, test_size=0.2, group='patient_id', seed=None):
    """
     Split dataset into random train and test subsets, while keeping all the
     samples of one patient in the same set.

     Parameters:
        data (pd.Dataframe):
            The data that should be splitted.
            Needs to contain a column with the name specified by the group parameter.
        test_size (float): (Default: 0.2)
            A number between 0.0 and 1.0 specifying the relative size of the test set.
        group (string): (Default: 'patient_id')
            The name of the column that the data should be split by.
            Having multiple entries of the same value in this column will result in a split,
            where all of these entries will end up in the same subset.
        seed (int): (Default: None)
            Controlls the shuffling applied to the data before the it is split.
     Returns:
        train_data, test_data (touple of lists):
            Two lists consisting of the train and test split
    """

    if not group in data.columns:
        raise Exception('The column ' + group + ' does not exist')

    # create the group shuffle splitter
    shuffle_split = GroupShuffleSplit(
        test_size=test_size, random_state=seed)

    # define groups as patient ids
    groups = data[group].to_numpy()

    train_idx, test_idx = next(shuffle_split.split(data, groups=groups))

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    return (train_data, test_data)
