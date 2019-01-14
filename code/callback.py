import os
from tensorflow import keras


class ModelCheckpointAfter(keras.callbacks.ModelCheckpoint):
    """ModelCheckPoint after specified epoch instead of Every epoch.
    
        Attributes:
            id_epoch: Int.
                Check after id_epoch.
            filepath: Str.
                Path of Checkpoint.
            monitor: Str.
            verbose: 0 or 1.
            save_best_only: Bool.
            save_weights_only: Bool.
            mode: "max" or "min" or "auto".
            period: Int
    """

    def __init__(self,
                 id_epoch,
                 filepath,
                 monitor='val_loss',
                 verbose=1,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super().__init__(filepath, monitor, verbose, save_best_only,
                         save_weights_only, mode, period)
        self.after_epoch = id_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(epoch, logs)


def model_checkpoint_after(epoch, path, monitor, save_best_only, mode):
    pattern = os.path.join(path,
                           'epoch-{epoch:03d}-psnr-{' + monitor + ':.4f}.h5')
    return ModelCheckpointAfter(
        epoch,
        filepath=pattern,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode)


def learning_rate(step_size, decay, verbose=1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr

    return keras.callbacks.LearningRateScheduler(schedule, verbose=verbose)


def tensor_board(path, histogram_freq, **kwargs):
    """write_graph, write_grads, write_images"""
    return keras.callbacks.TensorBoard(
        log_dir=os.path.join(path, 'log'),
        histogram_freq=histogram_freq,
        **kwargs)
