# -*- coding: utf-8 -*-
"""
@File  : tensors.py
@Author: Yulong He
@Date  : 2023-03-22 5:05 p.m.
@Desc  : 
"""
import numpy as np
import tensorflow as tf

print(tf.__version__)
print(f"Num GPUs Available: {tf.config.list_physical_devices('GPU')}")
exit(0)


def create_const_tensor(data):
    tensor = tf.constant(data)
    print(tensor)
    print(tensor.ndim)


if __name__ == '__main__':
    # create
    create_const_tensor(1)
    # tf.Tensor(1, shape=(), dtype=int32)
    # 0

    create_const_tensor([1, 2])
    # tf.Tensor([1 2], shape=(2,), dtype=int32)
    # 1

    create_const_tensor([[1, 2], [3, 4]])
    # tf.Tensor(
    # [[1 2]
    #  [3 4]], shape=(2, 2), dtype=int32)
    # 2

    # dtype
    t1 = tf.constant([1, 2], dtype=tf.bfloat16)
    t2 = tf.constant([1, 2], dtype=tf.qint8)
    t3 = tf.constant([True, False], dtype=tf.bool)
    t4 = tf.constant(['hello', 'world'], dtype=tf.string)
    print(t1)  # tf.Tensor([1 2], shape=(2,), dtype=bfloat16)
    print(t1.shape)  # (2,)

    # casting
    t5 = tf.cast(t1, dtype=tf.float32)
    print(t5)  # tf.Tensor([1. 2.], shape=(2,), dtype=float32)

    # convert
    x = np.array([1, 2, 3])
    t6 = tf.convert_to_tensor(x, dtype=tf.float16)
    print(t6)  # tf.Tensor([1. 2. 3.], shape=(3,), dtype=float16)
    t7 = tf.constant(x, shape=(1, 3), dtype=tf.float16)
    print(t7)  # tf.Tensor([[1. 2. 3.]], shape=(1, 3), dtype=float16)
    print(t7.numpy())  # [[1. 2. 3.]]

    # matrix
    print(tf.eye(2))
    # tf.Tensor(
    # [[1. 0.]
    #  [0. 1.]], shape=(2, 2), dtype=float32)

    print(tf.eye(2, 3))
    # tf.Tensor(
    # [[1. 0. 0.]
    #  [0. 1. 0.]], shape=(2, 3), dtype=float32)

    # fill
    print(tf.fill((2, 3), 9))
    # tf.Tensor(
    # [[9 9 9]
    #  [9 9 9]], shape=(2, 3), dtype=int32)

    # ones / zeros
    print(tf.ones((2, 3)))
    # tf.Tensor(
    # [[1. 1. 1.]
    #  [1. 1. 1.]], shape=(2, 3), dtype=float32)
    tf.zeros((2, 3))

    # ones_like / zeros_like
    tf.ones_like(x)
    tf.zeros_like(x)

    # random_normal_initializer
    x = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
    print(x)
    # <tensorflow.python.ops.init_ops_v2.RandomNormal object at 0x00000264409884F0>

    # range
    x = tf.range(start=2, limit=10, delta=2)
    print(x)
    # tf.Tensor([2 4 6 8], shape=(4,), dtype=int32)

    # tf.rank == np.ndim
    print(tf.rank(
        tf.zeros((3, 2, 1, 1))
    ))  # tf.Tensor(4, shape=(), dtype=int32)

    # shape
    print(tf.shape(x))  # tf.Tensor([4], shape=(1,), dtype=int32)
    print(x.shape)  # (4,)

    # size
    print(tf.size(
        tf.zeros((3, 2, 1, 1))
    ))  # tf.Tensor(6, shape=(), dtype=int32)

    # random tensor
    g = tf.random.Generator.from_seed(1234)
    # g = tf.random.Generator.from_non_deterministic_state()
    print(g.normal(shape=(2, 3)))
    # tf.Tensor(
    # [[ 0.9356609   1.0854306  -0.93788373]
    #  [-0.5061547   1.3169702   0.7137579 ]], shape=(2, 3), dtype=float32)
    print(g.uniform(shape=(3,)))  # tf.Tensor([0.6512613  0.9295906  0.50873387], shape=(3,), dtype=float32)

    tf.random.normal(shape=(2, 3))
    tf.random.uniform(shape=(3,))

    # variable
    v = tf.Variable(1.)
    print(v)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
    v.assign(2.)
    print(v)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    v.assign_add(0.5)
    print(v)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.5>

    # shuffle
    x = tf.constant([1, 2, 3, 4, 5, 6])
    print(tf.random.shuffle(x))  # tf.Tensor([6 3 2 4 5 1], shape=(6,), dtype=int32)

    # ------------------------------------------------------
    # math
    x = tf.constant([-1.0, 2.0])
    print(tf.abs(x))  # tf.Tensor([1. 2.], shape=(2,), dtype=float32)
    print(tf.abs(3.0 ** 2 + 4.0 ** 2))  # tf.Tensor(25.0, shape=(), dtype=float32)

    x1 = tf.constant([1, 2])
    x2 = tf.constant([3, 4])
    print(tf.add(x1, x2))  # tf.Tensor([4 6], shape=(2,), dtype=int32)
    print(tf.multiply(x1, x2))  # tf.Tensor([3 8], shape=(2,), dtype=int32)

    x1 = tf.constant(10.0)
    x2 = tf.constant(0.0)
    print(tf.math.divide_no_nan(x1, x2))  # tf.Tensor(0.0, shape=(), dtype=float32)

    # broadcasting
    x1 = tf.constant([1, 2, 3])
    x2 = tf.constant(2)
    print(tf.multiply(x1, x2))  # tf.Tensor([2 4 6], shape=(3,), dtype=int32)

    x = tf.constant([10, 11, 12, 13, 14])
    print(tf.math.argmax(x))  # tf.Tensor(4, shape=(), dtype=int64)
    print(x[tf.math.argmax(x)])  # tf.Tensor(14, shape=(), dtype=int32)

    print(tf.math.reduce_sum(x))  # tf.Tensor(60, shape=(), dtype=int32)
    print(tf.math.reduce_max(x))  # tf.Tensor(14, shape=(), dtype=int32)
    print(tf.math.reduce_mean(x))  # tf.Tensor(12, shape=(), dtype=int32)
    x = tf.cast(x, dtype=tf.float16)
    print(tf.math.reduce_std(x))  # tf.Tensor(1.414, shape=(), dtype=float16)

    x = tf.constant([[1, 2, 3],
                     [4, 5, 6]])
    print(tf.math.top_k(x, k=1))
    # TopKV2(values=<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    # array([[3],
    #        [6]])>, indices=<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    # array([[2],
    #        [2]])>)

    # ------------------------------------------------------
    # matrix
    x1 = tf.constant([[1, 2], [3, 4]])
    x2 = tf.constant([[1, 0], [0, 1]])
    print(tf.linalg.matmul(x1, x2))
    print(x1 @ x2)
    # tf.Tensor(
    # [[1 2]
    #  [3 4]], shape=(2, 2), dtype=int32)
    print(tf.transpose(x1))
    # [[1 2]
    #  [3 4]], shape=(2, 2), dtype=int32)
    print(tf.linalg.band_part(x1, 0, -1))
    # tf.Tensor(
    # [[1 2]
    #  [0 4]], shape=(2, 2), dtype=int32)
    print(tf.linalg.band_part(x1, -1, 0))
    # tf.Tensor(
    # [[1 0]
    #  [3 4]], shape=(2, 2), dtype=int32)
    print(tf.linalg.band_part(x1, 0, 0))
    # tf.Tensor(
    # [[1 0]
    #  [0 4]], shape=(2, 2), dtype=int32)
    x1 = tf.cast(x1, dtype=tf.float32)
    print(tf.linalg.inv(x1))
    # tf.Tensor(
    # [[-2.0000002   1.0000001 ]
    #  [ 1.5000001  -0.50000006]], shape=(2, 2), dtype=float32)
    s, u, v = tf.linalg.svd(x1)

    x1 = tf.constant([[1, 2, 3], [4, 5, 6]])
    x2 = tf.constant([[1, 2], [3, 4], [5, 6]])
    print(tf.einsum('ij,jk->ik', x1, x2))
    # tf.Tensor(
    # [[22 28]
    #  [49 64]], shape=(2, 2), dtype=int32)
    # ------------------------------------------------------
    # new axis / squeeze
    x = tf.constant([1, 2, 3])
    print(x[tf.newaxis, ...])  # tf.Tensor([[1 2 3]], shape=(1, 3), dtype=int32)
    print(tf.expand_dims(x, axis=0))  # tf.Tensor([[1 2 3]], shape=(1, 3), dtype=int32)
    print(tf.reshape(x, shape=(1, 3)))  # tf.Tensor([[1 2 3]], shape=(1, 3), dtype=int32)

    print(tf.squeeze(x[tf.newaxis, ...], axis=0))  # tf.Tensor([1 2 3], shape=(3,), dtype=int32)

    t1 = [[1, 2], [3, 4]]
    t2 = [[5, 6], [7, 8]]
    print(tf.concat((t1, t2), axis=1))
    # tf.Tensor(
    # [[1 2 5 6]
    #  [3 4 7 8]], shape=(2, 4), dtype=int32)
    print(tf.stack((t1, t2), axis=1))
    # tf.Tensor(
    # [[[1 2]
    #   [5 6]]
    #
    #  [[3 4]
    #   [7 8]]], shape=(2, 2, 2), dtype=int32)
    t = tf.constant([[1, 2, 3], [4, 5, 6]])
    paddings = tf.constant([[1, 1], [2, 2]])
    print(tf.pad(t, paddings, 'CONSTANT'))
    # tf.Tensor(
    # [[0 0 0 0 0 0 0]
    #  [0 0 1 2 3 0 0]
    #  [0 0 4 5 6 0 0]
    #  [0 0 0 0 0 0 0]], shape=(4, 7), dtype=int32)

    params = tf.constant(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'])
    print(tf.gather(params, tf.range(1, 4)))  # tf.Tensor([b'p1' b'p2' b'p3'], shape=(3,), dtype=string)

    params = tf.constant([['a', 'b'], ['c', 'd']])
    print(tf.gather_nd(params, [[0], [1]]))
    # tf.Tensor(
    # [[b'a' b'b']
    #  [b'c' b'd']], shape=(2, 2), dtype=string)

    # ragged
    x = tf.ragged.constant([[1, 2], [3, 4, 5]])
    print(x)
    # <tf.RaggedTensor [[1, 2], [3, 4, 5]]>

    T, F = True, False
    print(tf.ragged.boolean_mask(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        mask=[[T, F, T], [F, F, F], [T, F, F]]
    ))  # <tf.RaggedTensor [[1, 3], [], [7]]>

    print(tf.RaggedTensor.from_row_lengths(
        values=[1, 2, 3, 4, 5, 6, 7, 8],
        row_lengths=[4, 0, 3, 1, 0]
    ))  # <tf.RaggedTensor [[1, 2, 3, 4], [], [5, 6, 7], [8], []]>

    # sparse tensor
    x = tf.sparse.SparseTensor(
        indices=[[0, 0], [2, 2]],
        values=[11, 56],
        dense_shape=[3, 3]
    )
    print(tf.sparse.to_dense(x))
    # tf.Tensor(
    # [[11  0  0]
    #  [ 0  0  0]
    #  [ 0  0 56]], shape=(3, 3), dtype=int32)

    # string
    print(tf.strings.join(['abc', 'def']).numpy())  # b'abcdef'

    with tf.device('GPU:0'):
        x_var = tf.Variable(0.2)
        x_con = tf.constant(0.2)
    print(x_var)
    print(x_var.device)
    print(x_con)
    print(x_con.device)
