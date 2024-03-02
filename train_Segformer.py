
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import argparse
import time
import datetime
import tensorflow as tf
import numpy as np
from loader import GPRegTFReader
from network import GpTransformer, AutoencoderEmbed
from keras.preprocessing import image
from keras.utils import image_utils as image
import cv2
import time
# Hyper Parameters
hyper_params = {
    'maxIter': 1500000,
    'batchSize': 128,
    'dbDir': '',
    'outDir': '',
    'device': '0',
    'dispLossStep': 100,
    'exeValStep': 1000,
    'saveModelStep': 1000,
    'ckpt': '',
    'cnt': False,
    'rootSize': 64,
    'status:': 'train',
    'embed_ckpt': '',
    'nb_layers':2,
    'd_model': 256,
    'd_ff': 2048,
    'nb_heads': 2,
    'drop_rate': 0.4,
    'nb_stroke_max': 512,
    'nb_gp_max': 64,
    'imgSize': 256,
    'maxS': 3,
}

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', help='checkpoint path', type=str, default='')
parser.add_argument('--cnt', help='continue training flag', type=bool, default=False)
parser.add_argument('--status', help='training or testing flag', type=str, default='train')
parser.add_argument('--d_model', help='codeSize and bottleneck size', type=int, default=256)
parser.add_argument('--bSize', help='batch size', type=int, default=8)
parser.add_argument('--maxS', help='maximum step size', type=int, default=3)

args = parser.parse_args()
hyper_params['dbDir'] = r"data_airplane/"
hyper_params['num_of_group'] = 4
hyper_params['outDir'] = r"result2/"

hyper_params['cnt'] = True
hyper_params['ckpt'] = "/data1/sketch2/result2/_20230908_104826/checkpoint"
hyper_params['status'] = 'test'
hyper_params['en_embed_ckpt'] = r"result/_20230810_100818/checkpoint"
hyper_params['de_embed_ckpt'] = r"result/_20230810_100818/checkpoint"
hyper_params['d_model'] = 256
hyper_params['batchSize'] = 32
hyper_params['maxS'] = args.maxS

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def create_padding_mask(seq, token):
    seq = tf.reduce_sum(seq, axis=2)
    seq = tf.cast(tf.math.equal(seq, token * hyper_params['d_model']), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, -2.0)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp, -2.0)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, -2.0)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def loss_fn(real, pred):

    # create mask
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, -1.0)), tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(real, pred)  # remove sigmoid from the last network layer
    loss *= mask

    nb_elem = tf.reduce_sum(mask)
    loss_val = tf.reduce_sum(loss) / nb_elem

    pred_sigmoid_masked = tf.round(tf.math.sigmoid(pred)) * mask

    real_masked = real * mask
    acc_val = tf.reduce_sum(tf.math.abs(pred_sigmoid_masked - real_masked)) / nb_elem

    return loss_val, 1.0 - acc_val



def get_predicted_gp(inp, tar, gt_label, label_pred, allStroke_map):

    # mask
    mask = tf.cast(tf.math.logical_not(tf.math.equal(gt_label, -1.0)), tf.float32)

    # calculate the real nb_g, nb_s
    # inp: [N, nb_s, 256]
    inp_sum = tf.reduce_sum(inp, axis=2)
    inp_sum_mask = tf.cast(tf.logical_not(tf.math.equal(inp_sum, -2.0 * 256)), tf.int32)  # [N, nb_s]
    nb_strokes = tf.reduce_sum(inp_sum_mask, axis=1)  # [N]

    # tar: [N, nb_g, 256]
    tar_sum = tf.reduce_sum(tar, axis=2)
    tar_sum_mask = tf.cast(tf.logical_not(tf.math.equal(tar_sum, -2.0 * 256)), tf.int32)  # [N, nb_g]
    nb_groups = tf.reduce_sum(tar_sum_mask, axis=1)  # [N]

    target_gp_nb = tf.shape(tar)[1]

    # sigmoid, round
    label_pred = tf.sigmoid(label_pred) * mask  # [N, nb_gp, nb_s]

    # assemble strokes
    nb_batch = tf.shape(gt_label)[0]
    gp_stroke_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for itr in range(nb_batch):
        cur_nb_s = nb_strokes[itr]
        cur_nb_g = nb_groups[itr]
        cur_strokes = tf.slice(allStroke_map, [itr, 0, 0, 0], [1, -1, -1, cur_nb_s])  # [1, 256, 256, nb_s]
        cur_strokes = tf.transpose(cur_strokes, [3, 1, 2, 0])  # [nb_s, 256, 256, 1]
        cur_strokes = tf.subtract(1.0, cur_strokes)  # inverse stroke value: background-0.0, strokes-1.0
        cur_gp_labels = tf.slice(label_pred, [itr, 0, 0], [1, cur_nb_g, cur_nb_s])  # [1, nb_g, nb_s]
        cur_gp_labels = tf.reshape(cur_gp_labels, [cur_nb_g, cur_nb_s])  # [nb_g, nb_s]

        cur_stroke_rep = tf.repeat(tf.expand_dims(cur_strokes, axis=0), cur_nb_g, axis=0)  # [nb_g, nb_s, 256, 256, 1]
        cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_labels, axis=2), 256, axis=2)  # [nb_g, nb_s, 256]
        cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_label_rep, axis=3), 256,
                                     axis=3)  # [nb_g, nb_s, 256, 256]
        cur_gp_label_rep = tf.reshape(cur_gp_label_rep,
                                      [cur_nb_g, cur_nb_s, 256, 256, 1])  # [nb_g, nb_s, 256, 256, 1]
        gp_strokes_sel = cur_gp_label_rep * cur_stroke_rep  # [nb_g, nb_s, 256, 256, 1]
        gp_stroke_max = tf.reduce_max(gp_strokes_sel, axis=1)  # [nb_g, 256, 256, 1]
        gp_strokes = tf.subtract(1.0, gp_stroke_max)  # convert stroke value back

        # padding
        gp_strokes_shape = tf.shape(gp_strokes)
        gp_strokes_pad = tf.pad(gp_strokes, [[0, target_gp_nb - gp_strokes_shape[0]],
                                             [0, 0], [0, 0], [0, 0]], constant_values=0.0)
        gp_stroke_list = gp_stroke_list.write(itr, gp_strokes_pad)

    pred_gp_stroke = gp_stroke_list.stack()  # [N, nb_g, 256, 256, 1]

    return pred_gp_stroke


def sacc(label_pred, gt_label,nb_stroke):


    mask = tf.cast(tf.math.logical_not(tf.math.equal(gt_label, -1.0)), tf.int64)
    mask_idx = tf.reduce_max(mask, 1)
    gt_label_idx = tf.argmax(gt_label, 1)
    gt_label_idx_maked = gt_label_idx*mask_idx

    label_pred = tf.round(tf.sigmoid(label_pred))
    label_pred_idx = tf.argmax(label_pred, 1)
    label_pred_idx_maked = label_pred_idx*mask_idx

    not_equal = tf.math.not_equal(gt_label_idx_maked, label_pred_idx_maked)
    not_equal_count = tf.reduce_sum(tf.cast(not_equal, tf.int32),1)
    sacc_val = tf.reduce_mean(not_equal_count/nb_stroke)

    return 1 - sacc_val


def cacc(label_pred, gt_label):

    gt_label_idx = tf.argmax(gt_label, 1)

    label_pred = tf.round(tf.sigmoid(label_pred))
    label_pred_idx = tf.argmax(label_pred, 1)



    category_list = gt_label_idx.numpy().tolist()[0]
    prediction_list = label_pred_idx.numpy().tolist()[0]

    group_count = {}

    for category in category_list:
        if category in group_count:
            group_count[category] += 1
        else:
            group_count[category] = 1

    correct_predictions = {}
    for category, prediction in zip(category_list, prediction_list):
        if category == prediction:
            if category in correct_predictions:
                correct_predictions[category] += 1
            else:
                correct_predictions[category] = 1

    accuracy = {}

    total_groups = len(group_count.keys())
    correct_groups = sum(1 for acc in accuracy.values() if acc >=0.8)

    overall_accuracy = correct_groups / total_groups

    return overall_accuracy


def modify_matrix(matrix_tensor):
    matrix_tensor = matrix_tensor[0]
    num_rows = len(matrix_tensor)
    num_cols = len(matrix_tensor[0])
    for j in range(num_cols):
        column = matrix_tensor[:, j]
        has_one = tf.reduce_any(tf.equal(column, 1))
        last_one_row = tf.reduce_max(tf.where(tf.equal(column, 1)))
        if has_one:
            matrix_tensor = tf.tensor_scatter_nd_update(matrix_tensor, [(i, j) for i in range(num_rows) if i != last_one_row], tf.zeros(num_rows - 1))  # 将除最后一个1所在的行以外的元素改为0
        else:
            matrix_tensor = tf.tensor_scatter_nd_update(matrix_tensor, [(0, j)], [1])
    return matrix_tensor


def combine_images(images):

    images_gray = tf.tile(tf.expand_dims(images, -1), [1, 1, 1, 3])
    colors = tf.constant([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype = tf.float32)

    colors_broadcasted = tf.expand_dims(tf.expand_dims(colors, 1), 1)
    colors_broadcasted = tf.tile(colors_broadcasted, [1, 256, 256, 1])
    combined_image = tf.multiply(images_gray, colors_broadcasted)
    image = tf.reduce_sum(combined_image, axis=0)

    return image

# define model
en_modelAESSG = AutoencoderEmbed(code_size=hyper_params['d_model'], x_dim=hyper_params['imgSize'],
                              y_dim=hyper_params['imgSize'], root_feature=32)
en_embS_ckpt = tf.train.Checkpoint(modelAESSG=en_modelAESSG)
en_embS_ckpt_manager = tf.train.CheckpointManager(en_modelAESSG, hyper_params['en_embed_ckpt'], max_to_keep=30)
if en_embS_ckpt_manager.latest_checkpoint:
    en_embS_ckpt.restore(en_embS_ckpt_manager.latest_checkpoint)
    print('restore stroke en_network from the checkpoint {}'.format(en_embS_ckpt_manager.latest_checkpoint))

de_modelAESSG = AutoencoderEmbed(code_size=hyper_params['d_model'], x_dim=hyper_params['imgSize'],
                              y_dim=hyper_params['imgSize'], root_feature=32)
de_embS_ckpt = tf.train.Checkpoint(modelAESSG=de_modelAESSG)
de_embS_ckpt_manager = tf.train.CheckpointManager(de_modelAESSG, hyper_params['de_embed_ckpt'], max_to_keep=30)
if de_embS_ckpt_manager.latest_checkpoint:
    de_embS_ckpt.restore(de_embS_ckpt_manager.latest_checkpoint)
    print('restore stroke de_network from the checkpoint {}'.format(de_embS_ckpt_manager.latest_checkpoint))


transformer = GpTransformer(num_layers=hyper_params['nb_layers'], d_model=hyper_params['d_model'],
                            num_heads=hyper_params['nb_heads'], dff=hyper_params['d_ff'],
                            pe_input=hyper_params['nb_stroke_max'], pe_target=hyper_params['nb_gp_max'],
                            rate=hyper_params['drop_rate'])

# define reader
readerTrain = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=True,
                          raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=True, prefix='train')
readerEval = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=1, shuffle=False,
                         raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='valid')
readerTest = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=1, shuffle=False,
                         raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='test')



optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

val_loss_metric = tf.keras.metrics.Mean(name='Validate_loss', dtype=tf.float32)
val_acc_metric = tf.keras.metrics.Mean(name='Validate_acc', dtype=tf.float32)
val_cacc_metric = tf.keras.metrics.Mean(name='Validate_cacc', dtype=tf.float32)
val_sacc_metric = tf.keras.metrics.Mean(name='Validate_sacc', dtype=tf.float32)


train_loss_metric = tf.keras.metrics.Mean(name='Train_loss', dtype=tf.float32)
train_acc_metric = tf.keras.metrics.Mean(name='Train_acc', dtype=tf.float32)
train_cacc_metric = tf.keras.metrics.Mean(name='Train_cacc', dtype=tf.float32)
train_sacc_metric = tf.keras.metrics.Mean(name='Train_sacc', dtype=tf.float32)


test_loss_metric = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_acc_metric = tf.keras.metrics.Mean(name='test_acc', dtype=tf.float32)
test_cacc_metric = tf.keras.metrics.Mean(name='test_cacc', dtype=tf.float32)
test_sacc_metric = tf.keras.metrics.Mean(name='test_sacc', dtype=tf.float32)


def train_step(inp, tar, label, allStroke_map,nb_stroke):
    tar_inp = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss_val, acc_val = loss_fn(label, predictions)
        pred_gp_strokes = get_predicted_gp(inp, tar_inp, label, predictions, allStroke_map)
        sacc_val = sacc(predictions, label,nb_stroke)
        cacc_val = cacc(predictions, label)


    gradients = tape.gradient(loss_val, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss_metric.update_state(loss_val)
    train_acc_metric.update_state(acc_val)
    train_cacc_metric.update_state(cacc_val)
    train_sacc_metric.update_state(sacc_val)

    return loss_val, acc_val, pred_gp_strokes, sacc_val, cacc_val


def eval_step(inp, tar, label, allStroke_map,nb_stroke):
    tar_inp = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss_val, acc_val = loss_fn(label, predictions)
    val_pred_gp_stroke = get_predicted_gp(inp, tar_inp, label, predictions, allStroke_map)
    val_cacc_val = sacc(predictions, label,nb_stroke)

    val_loss_metric.update_state(loss_val)
    val_acc_metric.update_state(acc_val)
    val_cacc_metric.update_state(val_cacc_val)

    cacc_val = sacc(predictions, label,nb_stroke)

    return loss_val, acc_val, tf.sigmoid(predictions), val_pred_gp_stroke, cacc_val


def assemble_gp(pred_label, full_stroke_input):
    # pred_label [1, 1, nb_stroke]
    # full_stroke_input [1, 256, 256, nb_stroke]

    nb_stroke = tf.shape(pred_label)[2]
    cur_strokes = tf.transpose(full_stroke_input, [3, 1, 2, 0])  # [nb_s, 256, 256, 1]
    cur_gp_labels = tf.reshape(pred_label, [1, nb_stroke])  # [1, nb_stroke]
    cur_stroke_rep = tf.reshape(cur_strokes, [1, nb_stroke, 256, 256, 1])  # [1, nb_stroke, 256, 256, 1]
    cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_labels, axis=2), 256, axis=2)  # [1, nb_stroke, 256]
    cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_label_rep, axis=3), 256, axis=3)  # [1, nb_stroke, 256, 256]
    cur_gp_label_rep = tf.reshape(cur_gp_label_rep, [1, nb_stroke, 256, 256, 1])  # [1, nb_stroke, 256, 256, 1]

    gp_strokes_sel = cur_gp_label_rep * cur_stroke_rep  # [1, nb_s, 256, 256, 1]
    gp_stroke_sum = tf.reduce_sum(gp_strokes_sel, axis=1)  # [1, 256, 256, 1]
    label_rep_sum = tf.reduce_sum(cur_gp_label_rep, axis=1)  # [1, 256, 256, 1]
    gp_strokes = tf.where(tf.math.equal(gp_stroke_sum, label_rep_sum), 1.0, 0.0)  # [1, 256, 256, 1]

    return gp_strokes


def generate_grouped_image(grouped_images):
    nb_g = grouped_images.shape[0]
    image_height = grouped_images.shape[1]
    image_width = grouped_images.shape[2]
    colormap = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    grouped_images_float = tf.cast(grouped_images, dtype=tf.float32)
    canvas = tf.ones(shape=(image_height, image_width, 3))
    for i in range(nb_g):
        current_image = grouped_images_float[i]
        colored_image = tf.tile(tf.expand_dims(tf.expand_dims(colormap[i], axis=0),axis=1), [image_height, image_width, 1])
        colored_image = tf.multiply(colored_image, tf.expand_dims(current_image, axis=2))
        canvas = tf.subtract(canvas, colored_image)

    canvas = tf.clip_by_value(canvas, 0.0, 1.0)
    canvas = tf.cast(canvas * 255, dtype=tf.float32)


    return canvas


# without k=maxS, true ground truth labels
def cook_raw(en_net, de_net, input_raw, glabel_raw, nb_gps, nb_strokes, gp_embSize):
    gp_start_token = tf.fill([1, gp_embSize], -1.0)
    nb_batch = tf.shape(input_raw)[0]

    input_cook_list = []
    gp_token_cook_list = []
    label_cook_list = []
    gp_stroke_list = []


    for itr in range(nb_batch):
        # get group and stroke numbers
        nb_gp = nb_gps[itr]
        nb_stroke = nb_strokes[itr]

        # get slice data
        cur_input = tf.slice(input_raw, [itr, 0, 0, 0], [1, -1, -1, nb_stroke])
        cur_label = tf.slice(glabel_raw, [itr, 0, 0], [1, nb_gp, nb_stroke])
        cur_label = tf.cast(tf.reshape(cur_label, [nb_gp, nb_stroke]),tf.float32)  # [nb_gp, nb_stroke]

        # stroke embedding
        input_trans = tf.transpose(cur_input, [3, 1, 2, 0]) # [nb_stroke, 256, 256, 1]
        input_cook = en_net.encoder(input_trans, training=False)  # [nb_stroke, 256]

        gp_stroke_imgs = group_images(input_raw, glabel_raw)
        gp_stroke_imgs = tf.expand_dims(gp_stroke_imgs,axis=3)

        # [nb_g, 256, 256, 1]
        gp_embed = de_net.encoder(gp_stroke_imgs, training=False)[:-1]  # [nb_g-1, 256]

        # add start and end token
        gp_cook = tf.concat([gp_start_token, gp_embed], axis=0)

        # label: add end group label (all zeros)
        label_cook = cur_label

        # padding
        target_stroke_nb = tf.shape(input_raw)[3]
        target_gp_nb = tf.shape(glabel_raw)[1]
        assert (target_stroke_nb == tf.shape(glabel_raw)[2])

        input_cook_shape = tf.shape(input_cook)
        input_cook_pad = tf.pad(input_cook, [[0, target_stroke_nb - input_cook_shape[0]], [0, 0]], constant_values=-2.0)
        input_cook_pad = tf.reshape(input_cook_pad, [1, -1, input_cook_shape[1]])

        gp_cook_shape = tf.shape(gp_cook)
        gp_cook_pad = tf.pad(gp_cook, [[0, target_gp_nb - gp_cook_shape[0] ], [0, 0]], constant_values=-2.0)
        gp_cook_pad = tf.reshape(gp_cook_pad, [1, -1, gp_cook_shape[1]])

        label_cook_shape = tf.shape(label_cook)
        label_cook_pad = tf.pad(label_cook,
                                [[0, target_gp_nb - label_cook_shape[0] ],
                                 [0, target_stroke_nb - label_cook_shape[1]]], constant_values=-1.0)
        label_cook_pad = tf.reshape(label_cook_pad, [1, -1, target_stroke_nb])

        gp_stroke_pad = tf.pad(gp_stroke_imgs, [[0, target_gp_nb - nb_gp + 1],
                                            [0, 0], [0, 0], [0, 0]], constant_values=0.0)
        gp_stroke_pad = tf.reshape(gp_stroke_pad, [1, -1, 256, 256, 1])

        gp_stroke_list.append(gp_stroke_pad)
        input_cook_list.append(input_cook_pad)
        gp_token_cook_list.append(gp_cook_pad)
        label_cook_list.append(label_cook_pad)

    gp_img = tf.concat(input_cook_list, axis=0)
    gp_label = tf.concat(label_cook_list, axis=0)
    gp_token = tf.concat(gp_token_cook_list, axis=0)
    gp_stroke_imgs = tf.concat(gp_stroke_list, axis=0)

    return gp_img, gp_label, gp_token, gp_stroke_imgs


count_index = 0
def test_autoregre_step(inp, label, allStroke_map):
    global count_index
    # inp:      [1, nb_s, 256]
    # label:    [1, nb_g, nb_s]
    # fullsmap: [1, 256, 256, nb_s]


    # assemble start group token
    nb_stroke = tf.shape(inp)[1]
    stroke_idx = np.arange(0, nb_stroke)
    cur_inp = inp
    cur_nb_s = nb_stroke
    cur_stroke_idx = np.arange(0, cur_nb_s)
    full_prediction = []
    gp_token = tf.fill([1, 1, hyper_params['d_model']], -1.0)

    predicts = []
    nb_max_try = hyper_params['num_of_group']
    for itr in range(nb_max_try):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(cur_inp, gp_token)
        predictions, _ = transformer(cur_inp, gp_token, False, enc_padding_mask,
                                     combined_mask, dec_padding_mask)  # [batch, nb_group, nb_stroke]

        predicts.append(predictions[:,-1,:])

        # check the last predictions
        pred_sigmoid_round = tf.round(tf.math.sigmoid(predictions))
        pred_last_labels = pred_sigmoid_round[:, -1:, :]  # [batch, 1, nb_stroke]

        # add to the global prediction
        cur_global_label = np.zeros(nb_stroke)
        pred_last_labels_np = pred_last_labels.numpy()
        cur_global_label[stroke_idx[cur_stroke_idx[np.where(pred_last_labels_np == 1)[2]]]] = 1.0
        cur_global_label_tf = tf.reshape(tf.convert_to_tensor(cur_global_label, dtype=tf.float32),
                                         [1, 1, -1])  # [1, 1, nb_stroke]
        full_prediction.append(cur_global_label_tf)

        # calculate the new gp token and add to gp_token
        last_gp_strokes_sel = assemble_gp(cur_global_label_tf, allStroke_map)  # [1, 256, 256, 1]
        last_gp_strokes = tf.where(last_gp_strokes_sel > 0.5, 1.0, 0.0)  # [1, 256, 256, 1]

        last_gp_embed = en_modelAESSG.encoder(last_gp_strokes, training=False)  # [1, 256]
        last_gp_embed = tf.reshape(last_gp_embed, [1, 1, hyper_params['d_model']])  # [1, 1, 256]

        gp_token = tf.concat([gp_token, last_gp_embed], axis=1)

    full_prediction =  tf.concat(full_prediction, axis=1)

    predicts = tf.concat( predicts, axis=0)
    predicts = tf.expand_dims(predicts, 0)



    loss_val, acc_val = loss_fn(label,predicts)


    cacc_val = cacc(full_prediction, label)
    val_pred_gp_stroke = get_predicted_gp(inp, label, label,  full_prediction, allStroke_map)
    sacc_val  = sacc(full_prediction, label,nb_stroke)

    val_loss_metric.update_state(loss_val)
    val_acc_metric.update_state(acc_val)
    val_cacc_metric.update_state(cacc_val)
    val_sacc_metric.update_state(sacc_val)


    return full_prediction,loss_val,acc_val,sacc_val,val_pred_gp_stroke,cacc_val


def train_net():
    # Set logging

    best_result = 0
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_folder + '/summary/train_' + current_time
    test_log_dir = output_folder + '/summary/test_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Checkpoint
    if not hyper_params['ckpt']:
        hyper_params['ckpt'] = output_folder + '/checkpoint'
    ckpt = tf.train.Checkpoint(en_modelAESSG=en_modelAESSG, de_modelAESSG=de_modelAESSG, transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=100)

    if hyper_params['cnt']:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            train_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    # Training process
    for step in range(hyper_params['maxIter']):

        # train step
        input_raw,glabel_raw,nb_stroke,nb_gps = readerTrain.next()

        gp_img, gp_label, gp_token, gp_stroke_imgs = cook_raw(en_modelAESSG,de_modelAESSG,
                                                      input_raw,
                                                      glabel_raw,
                                                      nb_gps,
                                                      nb_stroke,
                                                      hyper_params['d_model'])

        train_loss_val, train_acc_val, pred_gp_strokes, train_sacc_val, train_cacc_val = train_step(gp_img, gp_token, gp_label, input_raw,nb_stroke)

        # display training loss
        if step % hyper_params['dispLossStep'] == 0:
            train_logger.info('Training loss at step {} is: {}, acc is: {}, sacc is: {}, cacc is: {}'.
                              format(step, train_loss_val, train_acc_val, train_sacc_val, train_cacc_val))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss_metric.result(), step=step)
                tf.summary.scalar('train_acc', train_acc_metric.result(), step=step)
                tf.summary.scalar('train_cacc', train_cacc_metric.result(), step=step)
                tf.summary.scalar('train_sacc', train_sacc_metric.result(), step=step)

                train_loss_metric.reset_states()
                train_acc_metric.reset_states()
                train_cacc_metric.reset_states()
                train_sacc_metric.reset_states()


                train_gt_strokes = tf.slice(input_raw, [0, 0, 0, 0], [1, -1, -1, -1])
                train_gt_strokes = tf.transpose(train_gt_strokes, [3, 1, 2, 0])
                tf.summary.image('train_gt_stroke', train_gt_strokes, step=step, max_outputs=50)
        # eval step
        if step % hyper_params['exeValStep'] == 0:
            val_loss_metric.reset_states()
            val_acc_metric.reset_states()
            val_cacc_metric.reset_states()
            val_sacc_metric.reset_states()
            try:
                while True:
                    input_raw, glabel_raw, nb_stroke, nb_gps = readerTest.next()
                    gp_img, gp_label, gp_token, gp_stroke_imgs = cook_raw(en_modelAESSG,de_modelAESSG,
                                                                           input_raw,
                                                                           glabel_raw,
                                                                           nb_gps,
                                                                           nb_stroke,
                                                                           hyper_params['d_model'])


                    full_prediction ,loss_val, acc_val, sacc_val , val_pred_gp_stroke, cacc_val = test_autoregre_step(gp_img, gp_label, input_raw)


            except StopIteration:
                train_logger.info('Validating loss at step {} is: {}, acc is: {}, cacc is: {}, sacc is: {}'.
                                  format(step, val_loss_metric.result(), val_acc_metric.result(), val_sacc_metric.result(), val_cacc_metric.result()))
                with test_summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss_metric.result(), step=step)
                    tf.summary.scalar('val_acc', val_acc_metric.result(), step=step)
                    tf.summary.scalar('val_cacc', val_cacc_metric.result(), step=step)
                    tf.summary.scalar('val_sacc', val_sacc_metric.result(), step=step)


                    eval_gt_strokes = tf.slice(input_raw, [0, 0, 0, 0], [1, -1, -1, -1])
                    eval_gt_strokes = tf.transpose(eval_gt_strokes, [3, 1, 2, 0])
                    tf.summary.image('val_gt_stroke', eval_gt_strokes, step=step, max_outputs=50)
        # save model
        if step % hyper_params['saveModelStep'] == 0 and step > 0:
            ckpt_save_path = ckpt_manager.save()
            train_logger.info('Save model at step: {:d} to file: {}'.format(step, ckpt_save_path))

            if val_sacc_metric.result() > best_result:
                best_result = val_sacc_metric.result()
            train_logger.info('Best result: sacc {}  cacc{}'.format(best_result, val_cacc_metric.result()))



def group_images(images, labels):
    images = tf.transpose(images[0], [2, 0, 1])
    labels = labels[0]
    nb_g = labels.shape[0]
    images_tensor = tf.convert_to_tensor(images)
    labels_tensor = tf.convert_to_tensor(labels)
    grouped_images = []
    for i in range(nb_g):
        indices = tf.where(tf.equal(labels_tensor[i], 1))
        indices = tf.reshape(indices, [-1])
        group_images_tensor = tf.gather(images_tensor, indices)
        summed_image = tf.reduce_sum(group_images_tensor, axis=0)
        grouped_images.append(summed_image)

    grouped_images_tensor = tf.convert_to_tensor(grouped_images)

    return grouped_images_tensor

def test_net():
    start_time = time.time()
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')
    # reader
    readerTest = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=1, shuffle=False,
                             raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='test')

    # save model
    tf.saved_model.save(transformer, os.path.join(output_folder, 'gpTFNet'))
    test_logger.info('Write model to gpTFNet')

    tf.saved_model.save(en_embS_ckpt, os.path.join(output_folder, 'en_embedNet'))
    test_logger.info('Write model to en_embedNet')

    tf.saved_model.save(de_modelAESSG, os.path.join(output_folder, 'de_embedNet'))
    test_logger.info('Write model to de_embedNet')

    # Checkpoint
    if not hyper_params['ckpt']:
        hyper_params['ckpt'] = output_folder + '/checkpoint'
    ckpt = tf.train.Checkpoint(en_modelAESSG=en_modelAESSG, de_modelAESSG=de_modelAESSG, transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=100)

    if hyper_params['cnt']:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('restore Grouper from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))


    val_loss_metric.reset_states()
    val_acc_metric.reset_states()
    val_cacc_metric.reset_states()
    val_sacc_metric.reset_states()


    try:
        img_folder = os.path.join(output_folder, 'imgs')
        if tf.io.gfile.exists(img_folder):
            tf.io.gfile.rmtree(img_folder)
        tf.io.gfile.makedirs(img_folder)

        test_itr = 1
        while True:

            input_raw, glabel_raw, nb_stroke, nb_gps = readerTest.next()
            gp_img, gp_label, gp_token, gp_stroke_imgs = cook_raw(en_modelAESSG, de_modelAESSG,
                                                                   input_raw,
                                                                   glabel_raw,
                                                                   nb_gps,
                                                                   nb_stroke,
                                                                   hyper_params['d_model'])

            # loss_val, acc_val, full_prediction, val_pred_gp_stroke, cacc_val = eval_step(gp_img, gp_token, gp_label, input_raw,nb_stroke)
            full_prediction, loss_val, acc_val, sacc_val,val_pred_gp_stroke,cacc_val = test_autoregre_step(gp_img, gp_label,
                                                                                        input_raw)
            full_prediction = full_prediction[:, :-1, :]
            full_prediction = tf.expand_dims(modify_matrix(full_prediction),axis=0)

            test_logger.info('Testing step {}, acc is: {}, sacc is: {},cacc:{}'.format(test_itr,acc_val,sacc_val,cacc_val))


            gp_stroke_imgs  = group_images(input_raw,glabel_raw)
            label_vis_img = generate_grouped_image(gp_stroke_imgs)
            fn_logits1 = os.path.join(img_folder, '{}_g_label.jpeg'.format(test_itr))
            image.save_img(fn_logits1, label_vis_img)

            pred_gp_stroke_imgs = group_images(input_raw,full_prediction)
            pred_vis_img = generate_grouped_image(pred_gp_stroke_imgs)
            fn_logits1 = os.path.join(img_folder, '{}_g_pred.jpeg'.format(test_itr))
            image.save_img(fn_logits1, pred_vis_img)

            test_itr += 1
    except StopIteration:

        test_logger.info('Testing average loss is: {}, acc is: {},  sacc is: {},cacc is: {}'.
                         format(val_loss_metric.result(), val_acc_metric.result(), val_sacc_metric.result(),val_cacc_metric.result()))

    end_time = time.time()
    execution_time = end_time - start_time


if __name__ == '__main__':

    # Set output folder
    timeSufix = time.strftime(r'%Y%m%d_%H%M%S')
    output_folder = hyper_params['outDir'] + '_{}'.format(timeSufix)
    if tf.io.gfile.exists(output_folder):
        tf.io.gfile.rmtree(output_folder)
    tf.io.gfile.makedirs(output_folder)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Begin training
    if hyper_params['status'] == 'train':
        train_net()
    elif hyper_params['status'] == 'test':
        test_net()
