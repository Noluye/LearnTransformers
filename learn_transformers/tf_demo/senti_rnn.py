# -*- coding: utf-8 -*-
"""
@File  : senti_rnn.py
@Author: Yulong He
@Date  : 2023-03-27 11:00 a.m.
@Desc  : 
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Simple RNN1

# Simple RNN2: all states，并没有提升

# LSTM

# batch, sequence, vocabulary
inputs = np.random.random([32, 10, 8]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences=True)
output = simple_rnn(inputs)
print(output.shape)
