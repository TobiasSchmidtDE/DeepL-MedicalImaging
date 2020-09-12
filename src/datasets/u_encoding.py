import numpy as np
import tensorflow as tf

# open: extend encoding to differ between pathologies


def uencode(enc_type, labels, unc_value=-1):
    # Author: Johanna & Tobias
    """
    uncertainty encoding as described in
    https://arxiv.org/pdf/1901.07031.pdf

    Parameters:
        enc_type (string): One of 'uzeroes', 'uones' or 'umulticlass'.
            Case 'uzeroes':
                All uncertainty values are replaced with 0
            Case 'uones':
                All uncertainty values are replaced with 1
            Case 'umulticlass':
                Each pathologie 'P' get's associated with tree separat classes:
                    - positiv (confidently present)
                    - negativ (confidently not present)
                    - uncertain (uncertainly present)
                Resulting in new label vector with 3 times the size of the label vector.
                So for example a label vector [1,-1, 0] gets translated to [1,0,0, 0,0,1, 0,1,0]
        labels (list): A list of label vectors having 0, 1 or unc_value as values.
        unc_value (int/str): Value used to indicate uncertainty of pathologies (default -1)

    Returns:
        A list of label vectors without any occurances of <unc_value> values.
    """

    uncertainty_mask = labels == unc_value

    if enc_type == 'uzeroes':
        labels[uncertainty_mask] = 0
    elif enc_type == 'uones':
        labels[uncertainty_mask] = 1
    elif enc_type == 'umulticlass':
        onehot_matrix = tf.keras.utils.to_categorical(labels)
        flattend_shape = labels.shape[:1] + (np.prod(labels.shape[1:]),)
        labels = onehot_matrix.reshape(flattend_shape)
    else:
        raise NotImplementedError(
            "Encodings of type {} are not supported".format(enc_type))

    return labels
