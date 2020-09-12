# Author: Johanna
import tensorflow as tf


def dense_block(x, blocks, name):
    """ Creates a dense block

        Parameters:
            x (layer): previous tensor
            blocks (int): number of blocks
            name (string): name for dense layer

        Returns:
            output tensor for blocks
        """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """ Creates a transition block with BN, Relu and Conv
        # Arguments
            x (layer): input tensor
            reduction (float): compression rate at transition layers
            name (string): block label
        # Returns
            output tensor for the block
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                           name=name + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Conv2D(int(tf.keras.backend.int_shape(x)[bn_axis] * reduction), 1,
                               use_bias=False,
                               name=name + '_conv')(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """ Creates a dense block
       Arguments
           x (layer): input tensor
           growth_rate (float): growth rate at dense layers
           name (string): block label
       Returns
           Output tensor for the block
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                            epsilon=1.001e-5,
                                            name=name + '_0_bn')(x)
    x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1,
                                use_bias=False,
                                name=name + '_1_conv')(x1)
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                            name=name + '_1_bn')(x1)
    x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, 3,
                                padding='same',
                                use_bias=False,
                                name=name + '_2_conv')(x1)
    x = tf.keras.layers.Concatenate(
        axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def densenet(classes=14):
    """ instantiates implementation as described in
        https://arxiv.org/abs/1905.06362
        code taken from
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
        and modified according to specifications in the paper
        Arguments
            classes (int): number of classes
        Returns
            densenet architecture model
    """
    blocks = [6, 12, 24, 16]
    img_input = tf.keras.layers.Input(shape=(256, 256, 3))
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = tf.keras.layers.Conv2D(
        64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='sigmoid', name='fc')(x)

    model = tf.keras.models.Model(img_input, x, name='densenet')
    model.simple_name = "guendel"
    return model
