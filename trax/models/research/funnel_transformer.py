# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
Language Processing https://arxiv.org/abs/2006.03236 """
import numpy as np

from trax import layers as tl
from trax.layers.relattention import RelativeAttentionLayer
from trax.layers import core
from trax.layers import initializers as init
from trax.layers import combinators as cb
from trax.layers.assert_shape import assert_shape
from trax.models.transformer import _FeedForwardBlock


def _InternalMaxPool(arr):
  shape = arr.shape
  arr = arr.reshape((*shape[:-1], int(shape[-1] / 2), 2))
  arr = arr.max(axis=-1, keepdims=False)
  return arr


@assert_shape('bld->bSd')
def PoolLayer(pool_layer=tl.AvgPool,
              pool_size=(2,),
              strides=(2,),
              separate_cls=True):
  return tl.Serial(
      tl.Branch(
          tl.Fn('select_cls_token', lambda x: x[:, :1, :]),
          tl.Serial(
              tl.Fn('rest_tokens', lambda x: x[:, 1:, :]),
              pool_layer(pool_size, strides)
          )
      ),
      tl.Concatenate(axis=1)
  ) if separate_cls else pool_layer(pool_size, strides)


def _upsample(short, masks, long):
  factor = -(-long.shape[1] // short.shape[1])  # ceil division
  new_vecs = long + short.repeat(factor, axis=1)[:, :long.shape[1], :]
  new_masks = masks.repeat(factor, axis=-1)[:, :, :, :long.shape[1]]
  return new_vecs, new_masks


def _Upsampler():
  return tl.Fn('Upsampler', _upsample, n_out=2)


def _FunnelBlock(d_model, d_ff, n_heads,
                 dropout, dropout_shared_axes, mode, ff_activation,
                 pool_layer, pool_size, strides, separate_cls):
  """Internal funnel block. On input it takes (activations, masks).

  Args:
      d_model: Final dimension of tensors at most points in the model, including
          the initial embedding output.
      d_ff: Size of special dense layer in the feed-forward part of each block.
      n_heads: Number of attention heads.
      dropout: Stochastic rate (probability) for dropping an activation value
          when applying dropout within a block.
      dropout_shared_axes: Tensor axes on which to share a dropout mask.
          Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
          a useful way to save memory and apply consistent masks to activation
          vectors at different sequence positions.
      mode: If `'train'`, each block will include dropout; else, it will
          pass all values through unaltered.
      ff_activation: Type of activation function at the end of each block; must
          be an activation-type subclass of `Layer`.
      pool_size: Shape of window that gets reduced to a single vector value.
          If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
          must be a tuple of length :math:`n-2`.
      strides: Offsets from the location of one window to the locations of
          neighboring windows along each axis. If specified, must be a tuple of
          the same length as `pool_size`. If None, then offsets of 1 along each
          window axis, :math:`(1, ..., 1)`, will be used.
  Returns:
      A list of layers that maps (activations, mask) to (activations', mask).
  """
  attention = RelativeAttentionLayer(
      d_feature=d_model, n_heads=n_heads, dropout=dropout, mode=mode)
  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)
  pooling = PoolLayer(pool_layer, pool_size, strides, separate_cls)

  return tl.Serial(  # vecs, pos_emb, biases, masks
      tl.Branch(pooling, None, None),  # h', h, h, pos_emb, biases, masks
      tl.Dup(),  # h', h', h, h, pos_emb, biases, masks
      tl.Parallel(
          None,
          attention
      ),  # h', attention(...), pos_emb, biases, masks
      tl.Add(),  # h'+attention(...), pos_emb, biases, masks
      tl.LayerNorm(),  # funnel_activations, pos_emb, biases, masks
      tl.Parallel(
          None,
          None,
          None,
          tl.Fn('max pool experiment',
                _InternalMaxPool),
      ),  # funnel_activations, pos_emb, biases, mask'
      feed_forward
  )


def _RelativeEncoderBlock(d_model, d_ff, n_heads,
                          dropout, dropout_shared_axes, mode, ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the block is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.

  Returns:
    A list of layers that maps (activations, att_vecs, mask) to
                               (activations, att_vecs, mask).
  """
  attention = RelativeAttentionLayer(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  dropout_ = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual( # vecs, pos_emb, biases, masks
          tl.LayerNorm(),
          cb.Select([0, 0, 0]),
          attention,
          dropout_,
      ), # vecs, pos_emb, biases, masks
      tl.Residual(
          feed_forward
      ), # vecs, pos_emb, biases, masks
  ]


def FunnelTransformerEncoder(vocab_size,
                             n_classes=10,
                             d_model=512,
                             d_ff=2048,
                             encoder_segment_lengths=(2, 2, 2),
                             n_heads=8,
                             max_len=2048,
                             dropout=0.1,
                             dropout_shared_axes=None,
                             mode='train',
                             ff_activation=tl.Relu,
                             pool_layer=tl.AvgPool,
                             pool_size=(2,),
                             strides=(2,),
                             separate_cls=True,
                             bias_initializer=init.RandomNormalInitializer(
                                 1e-6)):
  """Returns a Funnel Encoder.
  """
  embedding_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)]

  encoder_blocks = []
  n_encoder_segments = len(encoder_segment_lengths)

  for i in range(n_encoder_segments):
    # Building i'th segment
    for _ in range(encoder_segment_lengths[i]):
      # segment_size encoder blocks
      encoder_blocks.append(
          _RelativeEncoderBlock(d_model, d_ff, n_heads, dropout,
                                dropout_shared_axes,
                                mode, ff_activation))

    # if not last segment, add funnel block
    if i != n_encoder_segments - 1:
      encoder_blocks.append(_FunnelBlock(d_model, d_ff, n_heads, dropout,
                                         dropout_shared_axes, mode,
                                         ff_activation, pool_layer, pool_size,
                                         strides, separate_cls))

  # Global relative attentions bias initialization, layer sharing
  assert d_model % n_heads == 0 and d_model % 2 == 0
  d_head = d_model // n_heads

  def PrepareAttentionInputs():
    """Layer that prepares additional global biases for relative attention."""

    def F(input_tokens):
      seq_len = input_tokens.shape[-1]
      inv_freq = 1 / (10000 ** (np.arange(0.0, d_model, 2.0) / d_model))
      positions = np.arange(-seq_len + 1, seq_len, 1.0)
      sinusoid_freq = np.einsum('i,j->ij', positions, inv_freq)
      pos_emb = np.concatenate([np.sin(sinusoid_freq),
                                np.cos(sinusoid_freq)], axis=1)
      pos_emb = pos_emb.reshape((seq_len * 2 - 1, n_heads, d_head))
      return pos_emb

    def G(a, b, c):
      return c, (a, b)

    return tl.Serial(
        tl.Fn('Absolute positional embeddings', F, n_out=1),
        core.Weights(bias_initializer, shape=(1, n_heads, 1, d_head)),
        core.Weights(bias_initializer, shape=(1, n_heads, 1, d_head)),
        tl.Fn('To tuple', G, n_out=2)  # pos_emb, (v_bias, u_bias)
    )

  # Assemble and return the model.
  return tl.Serial(  # toks
      # Encode.
      tl.Branch(
          embedding_encoder,
          PrepareAttentionInputs(),
          tl.PaddingMask()),  # vecs, pos_emb, biases, masks
      encoder_blocks,  # vecs, pos_emb, biases, masks
      tl.Select([0], n_in=4),  # vecs
      tl.LayerNorm(),  # vecs

      # Map to output categories.
      tl.Mean(axis=1),  # vecs
      tl.Dense(n_classes),  # vecs
      tl.LogSoftmax(),  # vecs
  )


def FunnelTransformer(vocab_size,
                      d_model=512,  # start
                      d_ff=2048,
                      encoder_segment_lengths=(2, 2, 2),
                      n_decoder_blocks=2,
                      n_heads=8,
                      max_len=2048,
                      dropout=0.1,
                      dropout_shared_axes=None,
                      mode='train',
                      ff_activation=tl.Relu,
                      pool_layer=tl.AvgPool,
                      pool_size=(2,),
                      strides=(2,),
                      separate_cls=True,
                      bias_initializer=init.RandomNormalInitializer(1e-6)):
  """Returns a Full Funnel Transformer.
  """
  segments = len(encoder_segment_lengths)
  funnels = segments - 1
  assert (funnels >= 0)

  embedding_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)]

  n_encoder_segments = len(encoder_segment_lengths)

  encoder_blocks_before_first_pooling = [
      _RelativeEncoderBlock(d_model, d_ff, n_heads, dropout,
                            dropout_shared_axes, mode, ff_activation)
      for _ in range(encoder_segment_lengths[0])]
  encoder_blocks_from_first_pooling = []

  for i in range(1, n_encoder_segments):
    # Building i'th segment

    # add funnel block between segments
    encoder_blocks_from_first_pooling.append(
        _FunnelBlock(d_model, d_ff, n_heads, dropout,
                     dropout_shared_axes, mode,
                     ff_activation, pool_layer, pool_size,
                     strides, separate_cls))

    for _ in range(encoder_segment_lengths[i]):
      # segment_size encoder blocks
      encoder_blocks_from_first_pooling.append(
          _RelativeEncoderBlock(d_model, d_ff, n_heads, dropout,
                                dropout_shared_axes, mode, ff_activation))

  decoder_blocks = [_RelativeEncoderBlock(d_model, d_ff, n_heads, dropout,
                                          dropout_shared_axes, mode,
                                          ff_activation)
                    for _ in range(n_decoder_blocks)]

  # Assemble and return the model.
  return tl.Serial(  # toks
      tl.Branch(
          embedding_encoder, tl.PaddingMask()),  # vecs masks
      encoder_blocks_before_first_pooling,  # vecs masks
      tl.Select([0, 1, 0]),  # vecs masks residual = vecs
      encoder_blocks_from_first_pooling,  # vecs masks residual
      tl.Parallel(
          # residual from first segment is taken before
          # normalization, so apply it now
          None, None, tl.LayerNorm()),  # vecs masks norm(residual)
      _Upsampler(),  # vecs masks
      decoder_blocks,
      tl.Select([0], n_in=2),  # vecs
      tl.LayerNorm(),
  )
