from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs.update({'learning rate': K.eval(self.model.optimizer.lr)})
        super().on_train_batch_end(batch, logs)
