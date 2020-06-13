
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from src.datasets.u_encoding import uencode
from src.preprocessing.cropping.template_matching import TemplateMatcher


class ImageDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras image classifier
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, dataset, dataset_folder, label_columns, path_column="Path",
                 path_column_prefix="", shuffle=True, drop_last=False, batch_size=64,
                 dim=(256, 256), n_channels=3, nan_replacement=0, unc_value=-1, u_enc='uzeroes',
                 crop=False, crop_template=None, view_pos_column="Frontal/Lateral",
                 view_pos_frontal="Frontal", view_pos_lateral="Lateral", preprocess_input_fn = None,
                 # TODO: Add support for non-image features (continous and categorical)
                 # conti_feature_columns=[], cat_feature_columns=[],
                 ):
        """
        Returns a data generator for an image classifier model, that provides batch wise access
        to the data.

        Parameters:
            dataset (pd.dataframe):
                A pandas daframe with labels and features (paths to images) for each data sample.
                Must contain a column with the paths to the images (relative to the dataset_folder).
                Must contain all columns defined as labels (in label_columns)

            dataset_folder (Pathlib Path):
                Path to the root of the dataset folder

            label_columns (list):
                Names of columns of pathologies we want to use as labels.
                Valid values in label columns are:
                            0 for mentioned as confidently not present,
                            1 for mentioned as confidently present,
                            nan not mentioned,
                            'unc_value' for mentioned as uncertainly present.

            path_column (str): (default "Path")
                Name of the column that contains the relative path from
                the dataset_folder root directory to each image.

            path_column_prefix (str): (default "")
                Prefix that should be applied to the values of path_column.
                This is intendet to be used for cases, where the paths are
                not relative to the dataset_folder, but to a subfolder
                within the dataset_folder.

            shuffle (bool): (default True)
                Whether to shuffle the data between batches

            drop_last (bool): (default False)
                Wheter to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch
                size, then the last batch will be smaller.

            batch_size (int): (default 64)
                The number of samples per mini-batch.

            dim (int): (default 256x256)
                The dimension that all images will be rescaled to. If cropping is enabled,
                the rescaling is applied only after the image was cropped.

            n_channels (int): (default 3)
                Number of channels the image will be converted to.
                Note: Every image on disk (source) is expected to be a gray scale image.
                This parameter only controls the number of channels of the image
                produced by the generator.

            unc_value (int/str): (default -1)
                Value used to indicate uncertainty of pathologies

            nan_replacement (int): (default 0)
                Value that nan values are replaced with.
                Must be a valid value for label columns.

            u_enc (string): (default 'uzeros')
                style of encoding for uncertainty
                valid values: (uzeros, uones, umulticlass)

            crop (bool): (default False)
                Wether to crop images or not. Optional template may be supplied.
                Default templates are choosen when crop is True and crop_tempalte is None.

            crop_tempalte (dict): (default None)
                A custom template for the crop.
                See src.preprocessing.cropping.template_matching.TemplateMatcher

            view_pos_column (str): (default "Frontal/Lateral")
                If crop is True, the name of the column that provides
                the view position must be supplied.
            view_pos_frontal (str): (default "Frontal")
                If crop is True, the value that mark a sample as
                frontal xray in the view_pos_column must be provided.
            view_pos_lateral (str): (default "Lateral")
                If crop is True, the value that mark a sample as
                lateral xray in the view_pos_column must be provided.

            preprocess_input_fn (function): (default: None)
                When using a pretrained model from tf.keras.application this can be 
                used to provide the corresponding tf.application.*.preprocess_input function
                which will be called for each image input.

        Returns:
            generator (DataGenerator): generator with the given specifications
        """

        if not isinstance(dataset_folder, Path):
            raise ValueError(
                "dataset folder needs to be instance of pathlib.Path")

        # check dataset has path column
        if not path_column in dataset.columns:
            raise ValueError(
                path_column + ' is not a column of the dataset dataframe. Columns are: '
                + str(dataset.columns))

        # check that column for paths only contains strings
        if any([not isinstance(path, str) for path in dataset[path_column]]):
            raise ValueError(
                "Paths to images must be given as string. Dataframe contains non-string values in '"
                + str(path_column) + "' column")

        # check at least one label column is given:
        if len(label_columns) < 1 or label_columns is None:
            raise ValueError(
                "'label_columns' is empty or None, at least one label column must be provided")

        # check all labels are present in dataset dataframe
        for label_column in label_columns:
            if not label_column in dataset.columns:
                raise ValueError(
                    label_column + ' is not a column of the dataset dataframe. Columns are: '
                    + str(dataset.columns))

        # check that all label columns contain valid values {0,1, nan, unc_value}
        valid_values = [0, 1, unc_value]
        for label_column in label_columns:
            if not all(dataset[label_column].isin(valid_values) | dataset[label_column].isna()):
                raise ValueError(
                    label_column + ' contains values which are not valid or NaN. Valid values are '
                    + str(valid_values))

        self.dataset = dataset
        self.dataset_folder = dataset_folder
        self.label_columns = label_columns
        self.path_column = path_column
        self.path_column_prefix = path_column_prefix
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.nan_replacement = nan_replacement
        self.unc_value = unc_value
        self.u_enc = u_enc
        self.drop_last = drop_last
        self.view_pos_column = view_pos_column
        self.view_pos_frontal = view_pos_frontal
        self.view_pos_lateral = view_pos_lateral
        self.preprocess_input_fn = preprocess_input_fn
        self.template_matcher = None
        if crop:
            self.template_matcher = TemplateMatcher(
                template_conf=crop_template, size=dim)

        self.on_epoch_end()

    def get_new_index(self):
        """
            Returns a list of ids for the data generation.
        """
        if self.shuffle:
            return np.random.permutation(len(self.dataset))

        return range(len(self.dataset))

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        num_batches = len(self.dataset) / self.batch_size

        if self.drop_last:
            return int(np.floor(num_batches))

        return int(np.ceil(num_batches))

    def label_generation(self, sample_ids):
        """
        Loads one batch of encoded and cleaned labels.

        Parameters:
            sample_ids (integer list): the ids of the samples for which the
                                       labels should be retrieved.

        Returns:
            list (numpy.array) of the encoded labels
        """
        labels = self.dataset.iloc[sample_ids][self.label_columns].to_numpy()

        # replace nan values
        labels[np.isnan(labels)] = self.nan_replacement

        # enforce uncertainty encoding strategy
        labels = uencode(self.u_enc, labels, unc_value=self.unc_value)
        return np.array(labels, dtype=float)

    def data_generation(self, sample_ids):
        """
        Loads one batch of data.

        Parameters:
            sample_ids (integer list): the ids of the samples that should be loaded
                                       as part of the batch.

        Returns:
            Tupel with list of images and list of labels
        """
        img_paths = self.dataset.iloc[sample_ids][self.path_column].to_numpy()

        img_paths = str(self.dataset_folder) + "/" + \
            self.path_column_prefix + img_paths

        img_template_type = (self.dataset.iloc[sample_ids][self.view_pos_column]
                             .replace({self.view_pos_frontal: "frontal",
                                       self.view_pos_lateral: "lateral"})
                             .to_numpy())

        images = np.array([self.load_image(*args)
                  for args in zip(img_paths, img_template_type)])

        if self.preprocess_input_fn is not None:
            images = self.preprocess_input_fn(images)
            
        return images, self.label_generation(sample_ids)

    def load_image(self, path, template_type):
        """
        Paramter:
            path: the path to the image. Either absolut or relative to repositories root directory.

        Returns a numpy array of the gray scale image

        """
        img = Image.open(str(Path(path)))

        if img.mode != "L":
            img = img.convert(mode="L")

        if self.template_matcher is not None:
            # resize image to dim + 10%
            size = (int(self.dim[0] * 1.1), int(self.dim[1] * 1.1))
            img = img.resize(size)
            img = np.array(img).astype(np.float32)

            # crop image to correct dim
            img = self.template_matcher.crop(img, template_type)
            img = Image.fromarray(img)
        else:
            img = img.resize(self.dim)

        if self.n_channels == 3:
            img = img.convert(mode="RGB")
        elif self.n_channels != 1:
            raise NotImplementedError(
                "n_channels is only supported for values from set {1,3}")
        img_array = np.asarray(img)
        return img_array

    def __getitem__(self, batch_index):
        """
        Loads specific batch of data

        Paramters:
            batch_index (int): the id of the batch that should be loaded

        Returns:
            Tupel with list of images and list of labels of length batch_size

        """

        start_index = batch_index * self.batch_size
        end_index = (batch_index+1) * self.batch_size

        if ((self.drop_last and end_index > len(self.dataset))
                or (start_index > len(self.dataset))):
            raise ValueError("Index out of range! Number of batches exceeded."
                             " Only {max_batches} batches available, not {num_batches}.".format(
                                 max_batches=len(self), num_batches=batch_index))

        return self.data_generation(self.index[start_index:end_index])

    def _iter_(self):
        """
        An iterable function that samples batches from the dataset.
        """

        for i in iter(range(len(self))):
            yield self.data_generation(self[i])

    def get_label_batch(self, batch_index):
        """
        Loads a batch of labels (without loading the images)

        Paramters:
            batch_index (int): the id of the batch that should be loaded

        Returns:
            list of labels of length batch_size

        """
        start_index = batch_index * self.batch_size
        end_index = (batch_index+1) * self.batch_size

        if ((self.drop_last and end_index > len(self.dataset))
                or (start_index > len(self.dataset))):
            raise ValueError("Index out of range! Number of batches exceeded."
                             " Only {max_batches} batches available, not {num_batches}.".format(
                                 max_batches=len(self), num_batches=batch_index))

        return self.label_generation(self.index[start_index:end_index])

    def get_labels(self):
        """
        Returns all labels encoded and cleaned
        """
        if self.drop_last:
            return self.label_generation(self.index[:len(self)*self.batch_size])
        return self.label_generation(self.index)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.index = self.get_new_index()
