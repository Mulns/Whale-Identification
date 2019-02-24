from tensorflow import keras
import tensorflow as tf
import math

weight_decay = 5e-4


def norm_embedding(embedding):

    embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
    return embedding / (embedding_norm + keras.backend.epsilon())
    

def arcface_loss(inputs, out_num, m=0.5, s=64.):
    embedding, labels, cos_t = inputs
    # labels = keras.backend.squeeze(labels, axis=1)
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(
        tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

    # this condition controls the theta+m should in range [0, pi]
    #      0<=theta+m<=pi
    #     -m<=theta<=pi-m
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

    keep_val = s * (cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)

    mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    # mask = tf.squeeze(mask, 1)
    inv_mask = tf.subtract(1., mask, name='inverse_mask')

    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

    output = tf.add(
        tf.multiply(s_cos_t, inv_mask),
        tf.multiply(cos_mt_temp, mask),
        name='arcface_loss_output')
    return output


class ArcfaceLoss(keras.Model):
    def __init__(self, out_num, m=0.5, s=64., *args, **kwargs):
        super(ArcfaceLoss, self).__init__(*args, **kwargs)
        self.embd_norm = keras.layers.Lambda(
            norm_embedding, name="embedding_norm")
        self.dense = keras.layers.Dense(
            out_num,
            kernel_regularizer=keras.regularizers.l2(l=weight_decay),
            kernel_constraint=keras.constraints.UnitNorm(axis=0))
        self.arcface_loss = keras.layers.Lambda(
            arcface_loss,
            name="arcface_loss",
            arguments=dict(out_num=out_num, m=m, s=s))

    def call(self, inputs):
        embedding, labels = inputs
        embedding = self.embd_norm(embedding)
        # print("Embedding shape: ", keras.backend.int_shape(embedding))
        # print("Labels shape: ", keras.backend.int_shape(labels))
        cos_t = self.dense(embedding)
        # print("Cos_t shape: ", keras.backend.int_shape(cos_t))
        arc_loss = self.arcface_loss([embedding, labels, cos_t])
        # print("arc_loss shape: ", keras.backend.int_shape(arc_loss))
        return arc_loss


def inference_loss(inputs):
    logit, labels = inputs
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=labels))


def identity_loss(y_true, y_pred):
    return y_pred

