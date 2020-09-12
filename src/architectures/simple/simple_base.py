from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def SimpleBaseArchitecture(base_model_fn, num_classes, name=None, train_last_layer_only=False):
    # Author: Johanna
    """
    Instantiates a customized version of a predifined keras architecture that uses
    the specified keras architecture and adds a dense layer as the last layer.

    Parameters:
        base_model_fn :
            Any function from tf.keras.applications that returns a valid model
            and can be intitialized with the weights of imagenet.
        num_classes (int):
            number of classes predicted by the model
        name (str): (default None)
            name of the model for logging purposes
        train_last_layer_only (bool): (default False)
            Set to true if only the very last layer should be trained
    """
    try:
        base_model = base_model_fn(include_top=False, weights='imagenet')
    except Exception as err:
        print(err)
        ValueError(
            "You must provide a function from tf.keras.applications " +
            "that instantiates a model architecure")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    if train_last_layer_only:
        for layer in base_model.layers:
            layer.trainable = False

    prediction_layer = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=prediction_layer)
    model.simple_name = name if name is not None else base_model_fn.__name__
    model.simple_name += "_small" if train_last_layer_only else ""
    return model
