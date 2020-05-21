from sklearn.model_selection import GroupShuffleSplit


def train_test_split(data, test_size=0.2, group='patient_id', seed=None):
    """
     Split dataset into random train and test subsets, while keeping all the
     samples of one patient in the same set.

     Parameters:
        data (pd.Dataframe):
          The data that should be splitted
        test_size (float):
          A number between 0.0 and 1.0 specifying the size of the test set.
        seed (int):
          Controlls the shuffling applied to the data before the it is split.
     Returns:
        splits (list):
          A list consisting of the train and test split
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
