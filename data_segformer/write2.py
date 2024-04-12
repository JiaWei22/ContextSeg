import os.path
import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import random

random.shuffle (lst )


save_path = r"../SketchX-PRIS-Dataset-master/sg/for"
os.makedirs(save_path, exist_ok=True)
with open(r'../SketchX-PRIS-Dataset-master/Perceptual Grouping/airplane.ndjson', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)

data_list = json_data["train_data"]
random.shuffle (data_list )
air_plane_train = data_list[:700]
air_plane_test = data_list[700:]
img_size = 156
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

  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke = np.array(stroke)
  scale = (max_dimension / average_dimension)
  stroke[:,0:2] = stroke[:,0:2]/scale
  return stroke

def find_duplicate_indices(lst):
    result = [[] for _ in range(4)]

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

input_raw_list = []
glabel_raw_list = []

for inx, line_list in enumerate(air_plane_train):
    lines, group_id = strokes_to_lines(line_list)
    nb_stroke = len(group_id)
    nb_group = 4
    index_group, _ = find_duplicate_indices(group_id)
    glabel = np.zeros((nb_group, nb_stroke))
    for row, row_indices in enumerate(index_group):
        for col in row_indices:
            glabel[row][col] = 1
    glabel_raw_list.append(glabel)
    temp_for_input_raw = []
    for inxx, line in enumerate(lines):
        img = Image.new('1', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        pixels = [(int(x), int(y)) for x, y in line]
        draw.line(pixels, fill=1, width=2)
        arr = np.array(img)
        arr_with_pad = np.zeros((256, 256))
        start_row = (256 - img_size) // 2
        start_col = (256 - img_size) // 2
        arr_with_pad[start_row:start_row + img_size, start_col:start_col + img_size] = arr
        temp_for_input_raw.append(arr_with_pad)
    input_raw_list.append(temp_for_input_raw)


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




