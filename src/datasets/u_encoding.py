import pandas as pd


# open: extend encoding to differ between pathologies
def uencode(enc_type, labels, unc_value=-1, nan_value=0):
    """
    uncertainty encoding as described in
    https://arxiv.org/pdf/1901.07031.pdf
    """
    for key in labels:
        if isinstance(labels[key], list):
            for i, value in enumerate(labels[key]):
                if pd.isna(value):
                    labels[key][i] = nan_value
                if value == unc_value:
                    if enc_type == 'uzeroes':
                        if isinstance(labels[key], list):
                            labels[key][i] = 0
                    elif enc_type == 'uones':
                        labels[key][i] = 1
                    elif enc_type == 'umulticlass':
                        labels[key][i] = 2
        else:
            if pd.isna(labels[key]):
                labels[key] = nan_value
            if labels[key] == unc_value:
                if enc_type == 'uzeroes':
                    labels[key] = 0
                elif enc_type == 'uones':
                    labels[key] = 1
                elif enc_type == 'umulticlass':
                    labels[key] = 2
    if enc_type == 'umulticlass':
        num_classes = 3
    else:
        num_classes = 2
    return labels, num_classes
