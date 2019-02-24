import tensorflow as tf
import numpy as np
from tensorflow import keras
import math
i  = tf.cast(np.random.random_integers(0,5,(4,3)),tf.float32)
matrix = tf.constant([[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]],tf.float32)
norm = tf.norm(i,axis=0)
norm1 = keras.backend.sqrt(
            tf.reduce_sum(
                tf.square(i), axis=0, keepdims=True))
matrix_norm = tf.norm(matrix,ord='fro',axis=[-2,-1])
matrix_first_element_norm = tf.norm(matrix[0],ord='fro',axis=[-2,-1])
#等价于计算矩阵F范数
same_matrix_ele_norm = tf.sqrt(tf.reduce_sum(tf.square(matrix[0])))
with tf.Session() as sess:
    print(sess.run(i))
    print(sess.run(norm))
    print(sess.run(norm1))
    print("matrix_norm:",sess.run(matrix_norm))
    print("matrix_first_element_norm:",sess.run(matrix_first_element_norm))
    print("same_matrix_ele_norm:",sess.run(same_matrix_ele_norm))

embedding = tf.cast(np.random.random_integers(0,5,(4,3)),tf.float32)
labels = tf.cast(np.random.random_integers(0,10, (4,1)), tf.int32)
labels = keras.backend.squeeze(labels, axis=1)
cos_t = tf.cast(np.random.random_integers(0,5,(4,10)),tf.float32)
def arcface_loss(inputs, out_num, m=0.5, s=64.):
    embedding, labels, cos_t = inputs
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
out = arcface_loss([embedding, labels, cos_t], 10)
with tf.Session() as sess:
    print(sess.run(out))