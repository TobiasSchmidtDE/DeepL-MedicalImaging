from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class SimpleBase:
    """
        class to construct baseline models
    """
    def __init__(self, model_name, num_classes):
        """
        instantiates model

        Parameters:
            model_name (string): name of the model to be constructed
            num_classes (int): number of classes predicted by the model
        """
        # TODO: implement ResNext, other densenet
        if model_name == 'inceptionv3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        elif model_name == 'densenet121':
            base_model = DenseNet121(include_top=False, weights='imagenet')
        elif model_name == 'resnet152v2':
            base_model = ResNet152V2(include_top=False, weights='imagenet')
        else:
            raise NotImplementedError()
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
