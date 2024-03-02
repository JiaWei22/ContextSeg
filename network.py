
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import math
from keras import layers
from keras import Model
import numpy as np
import logging



# network logger initialization
net_logger = logging.getLogger('main.network')


class AddCoords(layers.Layer):
    """Add coords to a tensor"""

    def __init__(self, x_dim=256, y_dim=256, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)
        yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = tf.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
            ret = tf.concat([ret, rr], axis=-1)

        return ret


class CoordConv(layers.Layer):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, root_feature, with_r=False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        self.convModule = tf.keras.Sequential([
            layers.Conv2D(root_feature, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU()])

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.convModule(ret)

        return ret


def positionalencoding2d(d_model, height, width, batch_size):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))

    pe = tf.zeros([batch_size, d_model, height, width])

    d_model = d_model // 2
    div_term = tf.exp(tf.range(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = tf.reshape(tf.range(0., width), [1, 1, 1, width])
    pos_h = tf.reshape(tf.range(0., height), [1, 1, height, 1])

    pe[:, 0:d_model:2, :, :] = tf.sin(pos_w * div_term[:, tf.newaxis]).numpy().transpose(0, 1, 3, 2)
    pe[:, 1:d_model:2, :, :] = tf.cos(pos_w * div_term[:, tf.newaxis]).numpy().transpose(0, 1, 3, 2)
    pe[:, d_model::2, :, :] = tf.sin(pos_h * div_term[:, tf.newaxis]).numpy().transpose(0, 1, 3, 2)
    pe[:, d_model + 1::2, :, :] = tf.cos(pos_h * div_term[:, tf.newaxis]).numpy().transpose(0, 1, 3, 2)

    return pe

class AddPE(layers.Layer):
    def __init__(self):
        super(AddPE, self).__init__()


    def call(self, x):
        print(x.shape)
        batchsize =  32
        d_model = x.shape[3]
        width = x.shape[1]
        height = x.shape[2]

        pe = positionalencoding2d(d_model, height, width,batchsize)
        pe = tf.transpose(pe, perm=[0,2, 3, 1])
        print(pe)
        x = x+pe


        return x


class AutoencoderEmbed(Model):
    def __init__(self, code_size, x_dim, y_dim, root_feature):
        super(AutoencoderEmbed, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(x_dim, y_dim, 1), name='AE_input'),
            CoordConv(x_dim, y_dim, root_feature),  # Conv, BN, Relu
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(root_feature * 2, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2D(root_feature * 2, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(root_feature * 4, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2D(root_feature * 4, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(root_feature * 8, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),

            layers.Conv2D(root_feature * 8, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(root_feature * 16, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2D(root_feature * 16, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),

            layers.MaxPool2D((2, 2)),
            #
            layers.Flatten(),
            layers.Dense(4096, activation='sigmoid'),
            layers.Dense(code_size, activation='sigmoid')  # [Batch, codeSize]
        ])
        self.decoder_for_cons = tf.keras.Sequential([
            layers.Dense(4096),
            layers.Dense(32768),
            layers.Reshape((8, 8, 512)),

            layers.Conv2D(root_feature * 16, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),

            layers.Conv2DTranspose(root_feature * 16, (2, 2), 2, padding='same'),

            layers.Conv2D(root_feature * 8, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),

            layers.Conv2DTranspose(root_feature * 8, (2, 2), 2, padding='same'),

            layers.Conv2D(root_feature * 4, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2DTranspose(root_feature * 4, (2, 2), 2, padding='same'),

            layers.Conv2D(root_feature * 2, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2DTranspose(root_feature * 2, (2, 2), 2, padding='same'),

            layers.Conv2D(root_feature, (3, 3), padding='same'),
            layers.BatchNormalization(momentum=0.95),
            layers.ReLU(),
            layers.Conv2DTranspose(root_feature, (2, 2), 2, padding='same'),

            layers.Conv2D(1, (1, 1), padding='same')
        ])


    def call(self, input_tensor, training):
        code = self.encoder(input_tensor, training=training)
        decoded = self.decoder_for_cons(code, training=training)

        return decoded
    @tf.function(input_signature=[tf.TensorSpec([None, 256, 256, 1], tf.float32)])
    def forward(self, input_tensor):
        code = self.encoder(input_tensor, training=False)
        recons = self.decoder_for_cons(code, training=False)
        return {'output_code': code, 'output_recons': recons}


# Transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Pre-LN
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, mask)
        out1 = x + attn_output

        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        out2 = out1 + ffn_output

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # Pre-LN
        x_norm = self.layernorm1(x)
        attn1, attn_weights_block1 = self.mha1(x_norm, x_norm, x_norm,
                                               look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        out1 = x + attn1

        out1_norm = self.layernorm2(out1)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1_norm, padding_mask)
        out2 = out1 + attn2

        out2_norm = self.layernorm3(out2)
        ffn_output = self.ffn(out2_norm)
        out3 = out2 + ffn_output

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x += self.pos_encoding[:, :seq_len, :]
        # x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x += self.pos_encoding[:, :seq_len, :]

        # x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, d_model, rate=0.1):
        super(Discriminator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.dense2 = tf.keras.layers.Dense(512)
        self.dense3 = tf.keras.layers.Dense(d_model)

    def call(self, enc_output, dec_output, training):
        # dec_output: [N, nb_g, d_model]
        x = self.dense1(dec_output)  # [N, nb_g, 1024]
        x = self.dense2(x)  # [N, nb_g, 512]
        x = self.dense3(x)  # [N, nb_g, d_model]

        # Matrix dot-product: [N, nb_s, 256] * [N, nb_g, 256]
        x = tf.einsum('imk, ink->inm', enc_output, x)  # (batch_size, nb_group, nb_stroke)

        return x


class GpTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(GpTransformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)
        self.disc_layer = Discriminator(d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # final layer norm as in Pre-LN transformer paper
        dec_output = self.layernorm(dec_output)

        final_output = self.disc_layer(enc_output, dec_output, training)  # (batch_size, tar_seq_len, inp_seq_len)

        return final_output, attention_weights

    @tf.function(input_signature=[tf.TensorSpec([1, None, 256], tf.float32),  # batch * nb_s * 256
                                  tf.TensorSpec([1, None, 256], tf.float32),  # batch * nb_g * 256
                                  tf.TensorSpec([1, 1, 1, None], tf.float32),  # batch * 1 * 1 * nb_s
                                  tf.TensorSpec([1, 1, None, None], tf.float32),  # batch * 1 * nb_g * nb_g
                                  tf.TensorSpec([1, 1, 1, None], tf.float32)])  # batch * 1 * 1 * nb_g
    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, False, enc_padding_mask)
        dec_output, _ = self.decoder(tar, enc_output, False, look_ahead_mask, dec_padding_mask)
        dec_output = self.layernorm(dec_output)
        final_output = self.disc_layer(enc_output, dec_output, False)

        return {'res_group_label': final_output}

