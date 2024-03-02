import numpy as np
import tensorflow as tf

img_size = 156
numb_group = 5

def get_bounds(data, factor=1):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i][0]) / factor
    y = float(data[i][1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def scale_bound(stroke, average_dimension=img_size):
  """Scale an entire image to be less than a certain size."""

  # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
  # modifies stroke directly.

  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke = np.array(stroke)
  scale = (max_dimension / average_dimension)
  stroke[:,0:2] = stroke[:,0:2]/scale
  return stroke

def find_duplicate_indices(lst):
    result = [[] for _ in range(numb_group)]

    for i, num in enumerate(lst):
        result[int(num)].append(i)

    group_lengths = [len(group) for group in result]

    return result,group_lengths


def strokes_to_lines(strokes):
  strokes = scale_bound(strokes)

  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  group_id = []
  cur_group_ip = -1
  for i in range(len(strokes)):
    if strokes[i][2] == 1:
      x += float(strokes[i][0])
      y += float(strokes[i][1])
      line.append([x, y])
      group_id.append(cur_group_ip)
      lines.append(line)
      line = []
    else:
      x += float(strokes[i][0])
      y += float(strokes[i][1])
      line.append([x, y])
      cur_group_ip = strokes[i][3]
  return lines, group_id


input_raw_list = np.load('input_raw.npy')
glabel_raw_list = np.load('glabel_raw.npy')

test_input_list = np.load('test_input.npy')
test_label_list = np.load('test_label.npy')



writer = tf.compat.v1.python_io.TFRecordWriter('%s.tfrecord' %'former_train')
for idx in range(len(input_raw_list)):

    input_raw = np.array(input_raw_list[idx],   dtype=np.float32).transpose(1, 2, 0)
    glabel_raw = np.array(glabel_raw_list[idx] , dtype=np.int64)

    print(idx)


    features = {}

    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(float_list = tf.train.FloatList(value=input_raw.flatten().tolist())),
        'glabel_raw': tf.train.Feature(int64_list=tf.train.Int64List(value=glabel_raw.flatten().tolist())),
        'input_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=input_raw.shape)),
        'glabel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=glabel_raw.shape))

    }))
    writer.write(example.SerializeToString())


writer.close()



writer = tf.compat.v1.python_io.TFRecordWriter('%s.tfrecord' %'former_test')
for idx in range(len(test_input_list)):

    input_raw = np.array(test_input_list[idx],   dtype=np.float32).transpose(1, 2, 0)
    glabel_raw = np.array(test_label_list[idx] , dtype=np.int64)

    print(idx)


    features = {}

    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(float_list = tf.train.FloatList(value=input_raw.flatten().tolist())),
        'glabel_raw': tf.train.Feature(int64_list=tf.train.Int64List(value=glabel_raw.flatten().tolist())),
        'input_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=input_raw.shape)),
        'glabel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=glabel_raw.shape))

    }))
    writer.write(example.SerializeToString())

writer.close()


