import pandas as pd


# open: extend encoding to differ between pathologies
def uencode(enc_type, labels):
    for key in labels:
        for i, value in enumerate(labels[key]):
            if pd.isna(value):
                labels[key][i] = 0
            if value == -1:
                if enc_type == 'uzeroes':
                    labels[key][i] = 0
                elif enc_type == 'uones':
                    labels[key][i] = 1
                elif enc_type == 'umulticlass':
                    labels[key][i] = 2
    if enc_type == 'umulticlass':
        num_classes = 3
    else:
        num_classes = 2
    return labels, num_classes


def uencode_single(enc_type, labels):
    for key in labels:
        if pd.isna(labels[key]):
            labels[key] = 0
        if labels[key] == -1:
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
