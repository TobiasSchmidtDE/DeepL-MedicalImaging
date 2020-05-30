from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def SimpleBaseArchitecture(base_model_fn, num_classes):
    """
    Instantiates a customized version of a predifined keras architecture that uses
    the specified keras architecture and adds a dense layer as the last layer.

    Parameters:
        base_model_fn :
            Any function from tf.keras.applications that returns a valid model
            and can be intitialized with the weights of imagenet.
        num_classes (int): number of classes predicted by the model
    """
    try:
        base_model = base_model_fn(include_top=False, weights='imagenet')
    except TypeError:
        ValueError(
            "You must provide a function from tf.keras.applications " +
            "that instantiates a model architecure")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    for layer in base_model.layers:
        layer.trainable = False

    prediction_layer = Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=prediction_layer)
