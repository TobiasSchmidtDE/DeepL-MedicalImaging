from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class SimpleBase:
    """
        class to construct baseline models
    """

    def __init__(self, base_model_fn, num_classes):
        """
        instantiates model

        Parameters:
            base_model_fn : ...
            num_classes (int): number of classes predicted by the model
        """
        try:
            base_model = base_model_fn(include_top=False, weights='imagenet')
        except TypeError:
            ValueError(
                "You must provide a function from tf.keras.applications " +
                "that instantiates a model architecure")

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)

        prediction_layer = Dense(num_classes, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=prediction_layer)

    def get_model(self):
        """ returns model """
        return self.model
