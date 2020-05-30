import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from skmultilearn.model_selection import IterativeStratification


def train_test_split(data, test_size=0.2, group='patient_id', labels=None, seed=None):
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
        labels (list):
            A list of labels that are taken in consideration when performing the stratified
            split
        seed (int): (Default: None)
            Controlls the shuffling applied to the data before the it is split.
     Returns:
        train_data, test_data (touple of lists):
            Two lists consisting of the train and test split
    """

    if not group in data.columns:
        raise Exception('The column ' + group + ' does not exist')

    if labels:

        stratifier = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[test_size, 1.0-test_size],
            random_state=seed
        )

        # split into stratified test and train set
        train_idx, test_idx = next(stratifier.split(data, data[labels]))

        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # get group ids that are found in both sets
        splitted_group_ids = np.intersect1d(train_data[group].to_numpy(),
                                            test_data[group].to_numpy())

        train_value_counts = train_data[group].value_counts()
        test_value_counts = test_data[group].value_counts()

        # iterate through groups and move either to test or train based on
        # where the value count of the group is higher
        for group_id in splitted_group_ids:
            if train_value_counts[group_id] > test_value_counts[group_id]:
                rows = test_data[test_data[group] == group_id]
                train_data = pd.concat([train_data, pd.DataFrame(rows)])

                test_data = test_data[test_data[group] != group_id]
            else:
                rows = train_data[train_data[group] == group_id]
                test_data = pd.concat([test_data, pd.DataFrame(rows)])

                train_data = train_data[train_data[group] != group_id]

    else:
        # create the group shuffle splitter
        shuffle_split = GroupShuffleSplit(
            test_size=test_size, random_state=seed)

        # define groups as patient ids
        groups = data[group].to_numpy()

        train_idx, test_idx = next(shuffle_split.split(data, groups=groups))

        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

    return (train_data, test_data)
