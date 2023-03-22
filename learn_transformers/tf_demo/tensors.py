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
