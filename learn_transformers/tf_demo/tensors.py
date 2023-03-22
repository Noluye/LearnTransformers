# -*- coding: utf-8 -*-
"""
@File  : tensors.py
@Author: Yulong He
@Date  : 2023-03-22 5:05 p.m.
@Desc  : 
"""
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
