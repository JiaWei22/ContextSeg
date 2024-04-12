import os.path
import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import random
import GeodisTK
# save_path = r"../SketchX-PRIS-Dataset-master/sg/for"
# os.makedirs(save_path, exist_ok=True)

folder_path = r'../SketchX-PRIS-Dataset-master/Perceptual Grouping'
merged_data = []

img_size = 156


for filename in os.listdir(folder_path):

    with open(os.path.join(folder_path, filename), "r") as json_file:
        json_data = json.load(json_file)

        merged_data.extend(json_data["train_data"])

random.shuffle(merged_data)
# merged_data = merged_data[:10000]
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

def geodesic_distance_2d(I, S, lamb, iter):
    '''
    get 2d geodesic disntance by raser scanning.
    I: input image, can have multiple channels. Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic2d_raster_scan(I, S, lamb, iter)

def group_lines_by_category(lines, categories):
  category_dict = {}
  for line, category in zip(lines, categories):
    if category in category_dict:
        category_dict[category].append(line)
    else:
        category_dict[category] = [line]
  grouped_lines = list(category_dict.values())
  return grouped_lines


stroke_list = []
dis_list = []
group_list = []


for inx, line_list in enumerate(merged_data):
    print(inx)
    lines, group_id = strokes_to_lines(line_list)



    for inxx, line in enumerate(lines):
      img = Image.new('1', (img_size, img_size), 0)
      draw = ImageDraw.Draw(img)
      pixels = [(int(x), int(y)) for x, y in line]

      draw.line(pixels, fill=1, width=4)

      arr = np.array(img)

      arr_with_pad = np.zeros((256, 256))
      start_row = (256 - img_size) // 2  
      start_col = (256 - img_size) // 2 
      arr_with_pad[start_row:start_row + img_size, start_col:start_col + img_size] = arr

      stroke_list.append(arr_with_pad)

      arr_int = np.copy(arr_with_pad).astype(np.uint8)

      distance = GeodisTK.geodesic2d_raster_scan(arr_int.astype(np.float32), arr_int, 0, 5)
      sd = 1 / (1 + 0.01 * np.exp(distance))
      dis_list.append(sd)

for inx, line_list in enumerate(merged_data):
    print(inx)
    lines, group_id = strokes_to_lines(line_list)
    grouped_lines = group_lines_by_category(lines, group_id)
    # 定义曲线上的点
    for group_id, line_for_a_group in enumerate(grouped_lines):
      if len(line_for_a_group) == 1:
        continue
      img = Image.new('1', (img_size, img_size), 0)
      draw = ImageDraw.Draw(img)
      for line in line_for_a_group:
        pixels = [(int(x), int(y)) for x, y in line]
        draw.line(pixels, fill=1, width=4)
      img = img.transpose(Image.FLIP_TOP_BOTTOM)
      arr = np.array(img)
      arr_with_pad = np.zeros((256, 256))
      start_row = (256 - img_size) // 2 
      start_col = (256 - img_size) // 2 
      arr_with_pad[start_row:start_row + img_size, start_col:start_col + img_size] = arr
      group_list.append(arr_with_pad)

      arr_int = np.copy(arr_with_pad).astype(np.uint8)

      distance = GeodisTK.geodesic2d_raster_scan(arr_int.astype(np.float32), arr_int, 0, 5)
      sd = 1 / (1 + 0.01 * np.exp(distance))
      dis_list.append(sd)


def split_list(lst):
    total_length = len(lst)
    first_end = int(total_length * 0.8)
    second_end = int(total_length * 0.9)

    first_part = lst[:first_end]
    second_part = lst[first_end:second_end]
    third_part = lst[second_end:]

    return first_part, second_part, third_part


input_raw_list = group_list +stroke_list
random.shuffle(input_raw_list)


train_list ,test_list,valid_list = split_list(input_raw_list)
train_dis_list ,test_dis_list,valid_dis_list = split_list(dis_list)


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
