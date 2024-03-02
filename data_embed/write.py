import numpy as np
import tensorflow as tf

img_size = 156

def split_list(lst):
    total_length = len(lst)
    first_end = int(total_length * 0.7)
    second_end = int(total_length * 0.9)

    first_part = lst[:first_end]
    second_part = lst[first_end:second_end]
    third_part = lst[second_end:]

    return first_part, second_part, third_part


input_raw_list= np.load('input_raw.npy')
dis_raw_list = np.load('dis_raw.npy')



train_list ,test_list,valid_list = split_list(input_raw_list)
train_dis_list ,test_dis_list,valid_dis_list = split_list(dis_raw_list)


writer = tf.compat.v1.python_io.TFRecordWriter('%s.tfrecord' %'train')
for idx in range(len(train_list)):
    input_raw = np.array(train_list[idx], dtype=np.float32)
    input_dis = np.array(train_dis_list[idx], dtype=np.float32)
    print(idx)
    features = {}
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_raw.flatten().tolist())),
        'edis_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_dis.flatten().tolist())),
    }))
    writer.write(example.SerializeToString())
writer.close()

writer = tf.compat.v1.python_io.TFRecordWriter('%s.tfrecord' %'test')
for idx in range(len(test_list)):
    input_raw = np.array(test_list[idx], dtype=np.float32)
    input_dis = np.array(test_dis_list[idx], dtype=np.float32)
    print(idx)
    features = {}
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_raw.flatten().tolist())),
        'edis_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_dis .flatten().tolist())),
    }))
    writer.write(example.SerializeToString())
writer.close()

writer = tf.compat.v1.python_io.TFRecordWriter('%s.tfrecord' %'valid')
for idx in range(len(valid_list)):
    input_raw = np.array(valid_list[idx], dtype=np.float32)
    input_dis = np.array(valid_dis_list[idx], dtype=np.float32)
    print(idx)
    features = {}
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_raw.flatten().tolist())),
        'edis_raw': tf.train.Feature(float_list=tf.train.FloatList(value=input_dis.flatten().tolist())),
    }))
    writer.write(example.SerializeToString())
writer.close()
