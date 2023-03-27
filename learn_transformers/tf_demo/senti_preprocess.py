# -*- coding: utf-8 -*-
"""
@File  : senti_preprocess.py
@Author: Yulong He
@Date  : 2023-03-24 5:30 p.m.
@Desc  : 
"""
import string
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from nltk.stem.porter import PorterStemmer


def standardization(input_data):
    """
    Text Standardization
    1. converting to lower case
    2. take off HTML tags
    3. take off punctuations
    4. remove special characters
    5. remove accented characters
    6. [stemming]
    7. [lemmatization]
    :param input_data: raw reviews
    :return: standardized reviews
    """
    lowercase = tf.strings.lower(input_data)
    no_tag = tf.strings.regex_replace(lowercase, '<[^>]+>', '')
    output = tf.strings.regex_replace(no_tag, "[%s]" % re.escape(string.punctuation), '')
    return output


if __name__ == '__main__':
    train_ds, val_ds, test_ds = tfds.load('imdb_reviews',
                                          split=['train', 'test[:50%]', 'test[50%:]'],
                                          as_supervised=True)
    print(len(train_ds))
    print(len(val_ds))
    print(len(test_ds))
    for review, label in train_ds.take(5):
        print(review)
        print(label)
        print('-' * 20)

    raw_review = tf.constant(
        "<u>In the movie?, </u>man called Tévèz, went to a friend’s pl**ce and they had a tensed discussion. I don’t love this movie! would you?<br> <br /><br />T")
    print(raw_review)
    output = standardization(raw_review)
    print(output)

    # Tokenization
    # character tokenization
    # word tokenization
    # subword tokenization
    # n-gram

    # Numericalization
    # one-hot
    # bag of words
    # tf-idf
    # embeddings

    VOCAB_SIZE = 10000
    SEQUENCE_LENGTH = 200
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH
    )

    training_data = train_ds.map(lambda x, y: x)  # input x and y and outputx
    vectorize_layer.adapt(training_data)  # adapt the vectorize_layer to the training data


    # print(len(vectorize_layer.get_vocabulary())) # 10000

    def vectorizer(review, label):
        return vectorize_layer(review), label


    training_dataset = train_ds.map(vectorizer)
    val_dataset = val_ds.map(vectorizer)

    for review, label in training_dataset.take(1):
        print(review)
        print(label)
