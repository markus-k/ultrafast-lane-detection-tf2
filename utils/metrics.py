import tensorflow as tf


def ultrafast_accuracy(y_true, y_pred):
    return 1 - tf.math.reduce_mean(
        tf.math.pow(
            (tf.math.argmax(y_true, axis=1) - tf.math.argmax(y_pred, axis=1)) / y_true.shape[1],
            2
        )
    ) 

