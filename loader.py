
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow._api.v2.compat.v1 as tf
from abc import abstractmethod
from typing import Iterator, Optional
import numpy as np



# network logger initialization
import logging

loader_logger = logging.getLogger('main.loader')


class BaseReader(object):
    def __init__(self, batch_size, shuffle, raw_size, infinite, name=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.name = name
        self.raw_size = raw_size
        self.infinite = infinite

    @abstractmethod
    def next(self): pass


class AeTFReader(BaseReader):
    def __init__(self, data_dir, batch_size, shuffle, raw_size, infinite, prefix, name=None):
        super().__init__(batch_size, shuffle, raw_size, infinite, name)

        self.record_dir = data_dir
        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir)
                        if f.find(prefix) != -1 and  f.endswith('tfrecord')]
        loader_logger.info('Load TFRecords: {}'.format(record_files))


        self.raw_size = raw_size
        dataset = tf.data.TFRecordDataset(record_files)
        dataset = dataset.map(self.preprocess, tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(self.batch_size * 50)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = dataset
        self.iterator: Optional[Iterator] = iter(self.dataset)

    def preprocess(self, example_proto):
        features = tf.io.parse_single_example(example_proto,
                                              features={
                                                  'img_raw': tf.io.VarLenFeature(tf.float32),
                                                  'edis_raw': tf.io.VarLenFeature(tf.float32),
                                              })

        img_raw = tf.sparse.to_dense(features['img_raw'])
        input_raw = tf.reshape(img_raw, [self.raw_size[0], self.raw_size[1], 1])
        input_raw = tf.cast(input_raw, tf.float32)

        input_dis = tf.sparse.to_dense(features['edis_raw'])
        input_dis = tf.reshape(input_dis, [self.raw_size[0], self.raw_size[1], 1])
        input_dis = tf.cast(input_dis, tf.float32)

        return input_raw,input_dis

    def next(self):
        while True:
            try:
                next_elem = next(self.iterator)
                return next_elem
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                else:
                    raise StopIteration


class GPRegTFReader(BaseReader):
    def __init__(self, data_dir, batch_size, shuffle, raw_size, infinite, prefix, compress='ZLIB', name=None):

        super().__init__(batch_size, shuffle, raw_size, infinite, name)

        self.record_dir = data_dir
        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir)
                        if f.find(prefix) != -1 and f.endswith('tfrecord')]
        loader_logger.info('Load TFRecords: {}'.format(record_files))
        self.raw_size = raw_size
        self.stroke_padding = -2.0  # do not matter, we will padding again when preparing data
        self.label_padding = tf.cast(-1, tf.int64)  # same

        dataset = tf.data.TFRecordDataset(record_files)
        dataset = dataset.map(self.preprocess, tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(self.batch_size * 20)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([ raw_size[0], raw_size[1],None]),  # [256, 256, max]
                                           tf.TensorShape([None, None]),  # [max, max]
                                           tf.TensorShape([]),  # [max]
                                           tf.TensorShape([])
                                       ),
                                       padding_values=(self.stroke_padding,
                                                       self.label_padding,
                                                       0,
                                                       0),
                                       drop_remainder=True)


        self.dataset = dataset
        self.iterator: Optional[Iterator] = iter(self.dataset)

    def preprocess(self, example_proto):
        features = tf.io.parse_single_example(example_proto,
                                              features={
                                                  'img_raw': tf.io.VarLenFeature(tf.float32),
                                                  'glabel_raw': tf.io.VarLenFeature(tf.int64),
                                                  'input_shape': tf.io.FixedLenFeature([3], tf.int64),
                                                  'glabel_shape': tf.io.FixedLenFeature([2], tf.int64)


                                              })

        img_raw = tf.sparse.to_dense(features['img_raw'])

        glabel_raw = tf.sparse.to_dense(features['glabel_raw'])

        input_shape = features['input_shape']
        glabel_shape = features['glabel_shape']

        input_raw = tf.reshape(img_raw, input_shape)
        glabel_raw = tf.reshape(glabel_raw, glabel_shape)


        nb_gp = tf.shape(glabel_raw)[0]
        nb_stroke = tf.shape(glabel_raw)[1]

        return input_raw,glabel_raw,nb_stroke,nb_gp

    def next(self):
        while True:
            try:
                next_elem = next(self.iterator)
                return next_elem
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                else:
                    raise StopIteration

