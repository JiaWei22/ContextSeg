
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import time
import datetime
import numpy as np
import tensorflow as tf
from loader import AeTFReader
from network import AutoencoderEmbed
from keras.preprocessing import image
from keras.utils import image_utils
from PIL import Image,ImageOps
# Hyper Parameters
hyper_params = {
    'maxIter': 1500000,
    'batchSize': 64,
    'dbDir': 'data',
    'outDir': 'result',
    'device': '0',
    'rootFt': 32,
    'dispLossStep': 200,
    'exeValStep': 5000,
    'saveModelStep': 5000,
    'nbDispImg': 4,
    'ckpt': '',
    'cnt': False,
    'status:': 'train',
    'codeSize': 256,
    'imgSize': 256,
    'alpha':0.8
}

def loss_fn_for_eval(logits, labels):

    # sigmoid CE with logits
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)  # Remove sigmoid activation
    loss = tf.reduce_mean(loss)

    # sigmoid CE acc
    sigmoid_logits = tf.sigmoid(logits)
    train_acc = tf.keras.metrics.binary_accuracy(labels, sigmoid_logits)
    train_acc = tf.reduce_mean(train_acc)

    return loss, train_acc

def loss_fn_for_train(logits, labels,dis_map,dis_map_labels,alpha=0.8):

    # sigmoid CE with logits
    cons_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)  # Remove sigmoid activation
    cons_loss  = tf.reduce_mean(cons_loss)
    pre_loss = tf.losses.mean_squared_error(dis_map_labels, dis_map)
    pre_loss = tf.reduce_mean(pre_loss)

    loss = alpha*cons_loss+(1-alpha)*pre_loss

    # sigmoid CE acc
    sigmoid_logits = tf.sigmoid(logits)
    train_cons_acc = tf.keras.metrics.binary_accuracy(labels, sigmoid_logits)
    train_cons_acc = tf.reduce_mean(train_cons_acc)

    train_pre_acc = tf.keras.losses.mean_squared_error(dis_map_labels, dis_map)
    train_pre_acc = tf.reduce_mean(train_pre_acc)

    return loss,cons_loss,pre_loss, train_cons_acc,train_pre_acc


# @tf.function
def train_step(net, opt, inputs, labels, dis_map_labels,train_loss_metric, train_cons_acc_metric,train_pre_acc_metric):
    with tf.GradientTape() as tape:
        logits, dis_map = net(inputs, training=True)
        loss,cons_loss,pre_loss, train_cons_acc,train_pre_acc = loss_fn_for_train(logits, labels,dis_map,dis_map_labels,hyper_params['alpha'])
    grads = tape.gradient(loss, net.trainable_weights)
    opt.apply_gradients(zip(grads, net.trainable_weights))
    train_loss_metric.update_state(loss)
    train_cons_acc_metric.update_state( train_cons_acc)
    train_pre_acc_metric.update_state( train_pre_acc )

    return loss,cons_loss,pre_loss, train_cons_acc,train_pre_acc,logits,labels


# @tf.function
def eval_step(net, inputs, labels, val_loss_metric, val_acc_metric):
    val_logits,_ = net(inputs, training=False)
    val_loss, val_acc = loss_fn_for_eval(val_logits, labels)
    val_loss_metric.update_state(val_loss)
    val_acc_metric.update_state(val_acc)

    return val_loss, val_acc, val_logits, labels


# @tf.function
def test_step(net, inputs, labels, test_loss_metric, test_acc_metric):
    test_logits = net(inputs, training=False)
    test_loss, _ = loss_fn_for_eval(test_logits, labels)
    test_loss_metric.update_state(test_loss)

    # sigmoid CE
    sigmoid_logits = tf.sigmoid(test_logits)
    test_acc = tf.keras.metrics.binary_accuracy(labels, sigmoid_logits)
    test_acc = tf.reduce_mean(test_acc)
    test_acc_metric.update_state(test_acc)

    return test_loss, test_acc

def vis_step(net, inputs, labels):
    codes = net.encoder(inputs, training=False)
    logits = net.decoder_for_cons(codes, training=False)

    return labels, tf.sigmoid(logits)

def gaussian_noise(inputs, std):
    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=std, dtype=tf.float32)
    return inputs + noise


# @tf.function
def interpolation_step(net, inputs, labels):
    codes = net.encoder(inputs, training=False)
    logits = net.decoder_for_cons(codes, training=False)

    perturb_codes = gaussian_noise(codes, 0.1)
    perturb_logits = net.decoder_for_cons(perturb_codes)

    itp_codes = []
    for itr in range(11):
        ratio = itr / 10.0
        cur_codes = codes[0] * (1.0 - ratio) + ratio * codes[1]
        itp_codes.append(tf.reshape(cur_codes, [1, -1]))
    itp_codes = tf.concat(itp_codes, axis=0)
    interplated_logits = net.decoder_for_cons(itp_codes, training=False)

    # Sigmoid CE
    return labels, tf.sigmoid(logits), tf.sigmoid(perturb_logits), tf.sigmoid(interplated_logits)

# @tf.function
def interpolate3_step(net, inputs, labels):
    codes = net.encoder(inputs, training=False)
    logits = net.decoder_for_cons(codes, training=False)

    itp_code = tf.reduce_mean(codes, axis=0)  # [1, 256]
    itp_code = tf.reshape(itp_code, [1, 256])
    interpolated_logits = net.decoder_for_cons(itp_code, training=False)

    return labels, tf.sigmoid(logits), tf.sigmoid(interpolated_logits)


def train_net():
    # Set logging
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])
    modelAESSG.build(input_shape=(None, hyper_params['imgSize'], hyper_params['imgSize'], 1))
    modelAESSG.encoder.summary()

    # define reader
    readerAETrain = AeTFReader(hyper_params['dbDir'], hyper_params['batchSize'], True,
                               [hyper_params['imgSize'], hyper_params['imgSize']], True, 'train')
    readerAEEval = AeTFReader(hyper_params['dbDir'], hyper_params['batchSize'], False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'valid')

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_folder + '/summary/train_' + current_time
    valid_log_dir = output_folder + '/summary/valid_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    val_loss_metric = tf.keras.metrics.Mean(name='Validate_loss', dtype=tf.float32)
    val_acc_metric = tf.keras.metrics.Mean(name='Validate_acc', dtype=tf.float32)
    train_loss_metric = tf.keras.metrics.Mean(name='Train_loss', dtype=tf.float32)
    train_cons_acc_metric = tf.keras.metrics.Mean(name='Train_acc', dtype=tf.float32)
    train_pre_acc_metric = tf.keras.metrics.Mean(name='Train_acc', dtype=tf.float32)

    # Checkpoint
    if not hyper_params['ckpt']:
        hyper_params['ckpt'] = output_folder + '/checkpoint'
    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=50)

    if hyper_params['cnt']:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            train_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    tf.io.gfile.makedirs(hyper_params['midDir'])

    # Training process
    # step-by-step training, not epoch-based
    for step in range(hyper_params['maxIter']):

        # train step
        train_inputs ,dis_map_labels= readerAETrain.next()
        train_loss_val, cons_loss,pre_loss,train_cons_acc,train_pre_acc,\
                    train_logits, train_labels   =             train_step(modelAESSG,
                                                                           optimizer,
                                                                           train_inputs,
                                                                           train_inputs,
                                                                           dis_map_labels,
                                                                           train_loss_metric,
                                                                           train_cons_acc_metric,
                                                                           train_pre_acc_metric)


        # display training loss
        if step % hyper_params['dispLossStep'] == 0:
            train_logger.info('Training loss at step {} is: {}, cacc: {},pacc: {},cl:{},pl:{}'.format(step, train_loss_val, train_cons_acc,train_pre_acc,cons_loss,pre_loss))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss_metric.result(), step=step)
                tf.summary.scalar('train_cons_acc', train_cons_acc_metric.result(), step=step)
                tf.summary.scalar('train_pre_acc', train_pre_acc_metric.result(), step=step)
                train_loss_metric.reset_states()
                train_cons_acc_metric.reset_states()
                train_pre_acc_metric.reset_states()
                tf.summary.image('train_logits', train_logits, step=step, max_outputs=4)
                tf.summary.image('train_labels', train_labels, step=step, max_outputs=4)

        # eval step

        if step % hyper_params['exeValStep'] == 0 and step > 0:
            index = 0
            val_loss_metric.reset_states()
            try:
                while True:
                    index+=1
                    val_inputs,_ = readerAEEval.next()
                    _, _, val_logits, val_labels = eval_step(modelAESSG,
                                                             val_inputs,
                                                             val_inputs,
                                                             val_loss_metric,
                                                             val_acc_metric)
                    val_logits_vis = tf.sigmoid(val_logits)

                    if step == 5000:

                        fn_logits1 = os.path.join(hyper_params['midDir'],
                                                  '{}_{}_AE_logit_{}.jpeg'.format(index,step,  1))
                        logits_img1 = tf.slice(val_labels, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
                        image_utils.save_img(fn_logits1, logits_img1)

                    fn_logits0 = os.path.join(hyper_params['midDir'], '{}_{}_AE_logit_{}.jpeg'.format(index, step, 0))

                    logits_img0 = tf.slice(val_logits_vis, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
                    image_utils.save_img(fn_logits0, logits_img0)



            except StopIteration:
                train_logger.info('Validation loss at step {} is: {}, acc is: {}'.format(step,
                                                                                         val_loss_metric.result(),
                                                                                         val_acc_metric.result()))
                with valid_summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss_metric.result(), step=step)
                    tf.summary.scalar('val_acc', val_acc_metric.result(), step=step)

        # save model
        if step % hyper_params['saveModelStep'] == 0 and step > 0:
            # save model
            ckpt_save_path = ckpt_manager.save()
            train_logger.info('Save model at step: {:d} to file: {}'.format(step, ckpt_save_path))


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])

    # define reader
    readerAETest = AeTFReader(hyper_params['dbDir'], 1, False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'test')

    # Testing process
    test_loss_metric = tf.keras.metrics.Mean(name='Test_loss', dtype=tf.float32)
    test_acc_metric = tf.keras.metrics.Mean(name="Test_acc", dtype=tf.float32)

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        test_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    # save model
    tf.saved_model.save(modelAESSG, os.path.join(output_folder, 'embedNet'))
    test_logger.info('Write mode to embedNet')

    try:
        test_itr = 1
        while True:
            test_inputs = readerAETest.next()
            test_loss, test_acc = test_step(modelAESSG, test_inputs, test_inputs, test_loss_metric, test_acc_metric)

            test_logger.info('Testing step {}, loss is {}, acc is {}'.format(test_itr, test_loss, test_acc))

            test_itr += 1
    except StopIteration:
        test_logger.info('Testing average loss is: {}, acc is {}'.format(test_loss_metric.result(),
                                                                         test_acc_metric.result()))


def test_code():
    # Set logging
    interpolation_logger = logging.getLogger('main.interpolating')
    interpolation_logger.info('---Begin interpolating: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])

    # define reader - for coding interpolation, we use batch size=2
    readerAETest = AeTFReader(hyper_params['dbDir'], 2, False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'test')

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        interpolation_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    try:
        # create folder
        img_folder = os.path.join(output_folder, 'imgs')
        if tf.io.gfile.exists(img_folder):
            tf.io.gfile.rmtree(img_folder)
        tf.io.gfile.makedirs(img_folder)

        for test_itr in range(500):
            test_inputs = readerAETest.next()
            labels, logits, perturb_logits, interplated_logits = interpolation_step(modelAESSG,
                                                                                    test_inputs,
                                                                                    test_inputs)
            # write image out
            fn_logits0 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 0))
            fn_logits1 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 1))
            logits_img0 = tf.slice(logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img1 = tf.slice(logits, [1, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_logits0, logits_img0)
            image_utils.save_img(fn_logits1, logits_img1)

            fn_gt0 = os.path.join(img_folder, '{}_AE_gt_{}.jpeg'.format(test_itr, 0))
            fn_gt1 = os.path.join(img_folder, '{}_AE_gt_{}.jpeg'.format(test_itr, 1))
            gt_img0 = tf.slice(labels, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            gt_img1 = tf.slice(labels, [1, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_gt0, gt_img0)
            image_utils.save_img(fn_gt1, gt_img1)

            fn_pt0 = os.path.join(img_folder, '{}_AE_pt_{}.jpeg'.format(test_itr, 0))
            fn_pt1 = os.path.join(img_folder, '{}_AE_pt_{}.jpeg'.format(test_itr, 1))
            pt_img0 = tf.slice(perturb_logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            pt_img1 = tf.slice(perturb_logits, [1, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_pt0, pt_img0)
            image_utils.save_img(fn_pt1, pt_img1)

            for itr in range(11):
                fn_itp = os.path.join(img_folder, '{}_AE_ipt_{}.jpeg'.format(test_itr, itr / 10.0))
                itp_img = tf.slice(interplated_logits, [itr, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
                image_utils.save_img(fn_itp, itp_img)

            interpolation_logger.info("Iteation {}".format(test_itr))

    except StopIteration:
        interpolation_logger.info('End of the interpolation!')


def test_code3():
    # Set logging
    interpolation_logger = logging.getLogger('main.interpolating')
    interpolation_logger.info('---Begin interpolating: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])

    # define reader - for coding interpolation, we use batch size=2
    readerAETest = AeTFReader(hyper_params['dbDir'], 3, False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'test')

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        interpolation_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    try:
        # create folder
        img_folder = os.path.join(output_folder, 'imgs')
        if tf.io.gfile.exists(img_folder):
            tf.io.gfile.rmtree(img_folder)
        tf.io.gfile.makedirs(img_folder)

        for test_itr in range(50):
            test_inputs = readerAETest.next()
            labels, logits, interplated_logits = interpolate3_step(modelAESSG,
                                                                   test_inputs,
                                                                   test_inputs)
            # write image out
            fn_logits0 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 0))
            fn_logits1 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 1))
            fn_logits2 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 2))
            logits_img0 = tf.slice(logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img1 = tf.slice(logits, [1, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img2 = tf.slice(logits, [2, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_logits0, logits_img0)
            image_utils.save_img(fn_logits1, logits_img1)
            image_utils.save_img(fn_logits2, logits_img2)

            fn_gt0 = os.path.join(img_folder, '{}_AE_gt_{}.jpeg'.format(test_itr, 0))
            fn_gt1 = os.path.join(img_folder, '{}_AE_gt_{}.jpeg'.format(test_itr, 1))
            fn_gt2 = os.path.join(img_folder, '{}_AE_gt_{}.jpeg'.format(test_itr, 2))
            gt_img0 = tf.slice(labels, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            gt_img1 = tf.slice(labels, [1, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            gt_img2 = tf.slice(labels, [2, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_gt0, gt_img0)
            image_utils.save_img(fn_gt1, gt_img1)
            image_utils.save_img(fn_gt2, gt_img2)

            fn_pt0 = os.path.join(img_folder, '{}_AE_ipt_{}.jpeg'.format(test_itr, 0))
            pt_img0 = tf.slice(interplated_logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            image_utils.save_img(fn_pt0, pt_img0)

            interpolation_logger.info("Iteation {}".format(test_itr))

    except StopIteration:
        interpolation_logger.info('End of the interpolation!')




def test_white_image():
    # Set logging
    interpolation_logger = logging.getLogger('main.interpolating')
    interpolation_logger.info('---Begin interpolating: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])

    # define reader - for coding interpolation, we use batch size=2
    readerAETest = AeTFReader(hyper_params['dbDir'], 1, False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'test')

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        interpolation_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    try:
        # create folder
        img_folder = os.path.join(output_folder, 'imgs')
        if tf.io.gfile.exists(img_folder):
            tf.io.gfile.rmtree(img_folder)
        tf.io.gfile.makedirs(img_folder)

        for test_itr in range(3000):
            test_inputs ,_= readerAETest.next()
            labels, logits = vis_step(modelAESSG,test_inputs,test_inputs)
            # write image out
            fn_logits0 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 0))
            fn_logits1 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 1))

            logits_img0 = tf.slice(labels, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img1 = tf.slice(logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)

            logits_img0 = 1- logits_img0

            logits_img1 = 1 - logits_img1

            image_utils.save_img(fn_logits0, logits_img0)
            image_utils.save_img(fn_logits1, logits_img1)

            interpolation_logger.info("Iteation {}".format(test_itr))

    except StopIteration:
        interpolation_logger.info('End of the interpolation!')


def output_vis():
    # Set logging
    interpolation_logger = logging.getLogger('main.interpolating')
    interpolation_logger.info('---Begin interpolating: ---')

    # define model
    modelAESSG = AutoencoderEmbed(hyper_params['codeSize'], hyper_params['imgSize'],
                               hyper_params['imgSize'], hyper_params['rootFt'])

    # define reader - for coding interpolation, we use batch size=2
    readerAETest = AeTFReader(hyper_params['dbDir'], 1, False,
                              [hyper_params['imgSize'], hyper_params['imgSize']], False, 'test')

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        interpolation_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    try:
        # create folder
        img_folder = os.path.join(output_folder, 'imgs')
        if tf.io.gfile.exists(img_folder):
            tf.io.gfile.rmtree(img_folder)
        tf.io.gfile.makedirs(img_folder)

        for test_itr in range(3000):
            test_inputs ,_= readerAETest.next()
            labels, logits = vis_step(modelAESSG,test_inputs,test_inputs)
            # write image out
            fn_logits0 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 0))
            fn_logits1 = os.path.join(img_folder, '{}_AE_logit_{}.jpeg'.format(test_itr, 1))

            logits_img0 = tf.slice(labels, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img1 = tf.slice(logits, [0, 0, 0, 0], [1, -1, -1, -1]).numpy().reshape(256, 256, 1)
            logits_img0 = 1- logits_img0

            logits_img1 = 1 - logits_img1


            image_utils.save_img(fn_logits0, logits_img0)
            image_utils.save_img(fn_logits1, logits_img1)

            interpolation_logger.info("Iteation {}".format(test_itr))

    except StopIteration:
        interpolation_logger.info('End of the interpolation!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', help='GPU device indices', type=str, default='0')
    parser.add_argument('--ckpt', help='checkpoint path', type=str, default='')
    parser.add_argument('--cnt', help='continue training flag', type=bool, default=False)
    parser.add_argument('--rootFt', help='root feature size', type=int, default=32)
    parser.add_argument('--status', help='training or testing flag', type=str, default='train')

    args = parser.parse_args()
    hyper_params['dbDir'] = r"/data1/sketch2/data_all2/"
    hyper_params['outDir'] = r"result/"
    hyper_params['device'] = args.devices
    hyper_params['ckpt'] = r"/data1/sketch2/result/_20231028_145322/checkpoint"
    # hyper_params['ckpt'] = ""
    hyper_params['cnt'] = False
    hyper_params['rootFt'] = args.rootFt
    hyper_params['status'] = "vis"
    hyper_params['codeSize'] = 256
    hyper_params['midDir'] = r"mid_1/"

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
    elif hyper_params['status'] == 'codeItp':
        test_code()
    elif hyper_params['status'] == 'tripleItp':
        test_code3()
    elif hyper_params['status'] == 'white_image':
        test_white_image()
    elif hyper_params['status'] == 'vis':
        output_vis()
