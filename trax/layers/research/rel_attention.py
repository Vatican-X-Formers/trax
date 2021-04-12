# coding=utf-8
# Copyright 2021 The Trax Authors.
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
"""Relative attention related layers.

Implementation of Relative Attention mechanism first exposed in Transformer XL
paper: https://arxiv.org/pdf/1901.02860.pdf.
This particular implementation however focus on compatibility with
Funnel Transformer model from:
- Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
  Language Processing https://arxiv.org/abs/2006.03236
"""

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.assert_shape import assert_shape
from trax.layers.attention import MergeHeads
from trax.layers.attention import SplitIntoHeads

# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


@assert_shape('bSq,blk,blv,b1xp->bSd,b1xp')
def RelativeAttentionLayer(d_feature,
                           context_bias_layer,
                           location_bias_layer,
                           total_kv_pooling,
                           separate_cls,
                           n_heads=1,
                           dropout=0.0,
                           n_raw_tokens_generated=1,
                           max_inference_length=3072,
                           chunk_len=None,
                           chunk_offset=None,
                           mode='train'):
  """Returns a layer that maps (q, k, v, masks) to (activations, masks).

  When number of keys is smaller than number of queries layer works in O(q^2*d).
  Otherwise it is O(q*k*d). That is because we need to shift relative distances
  by current_pooling. When we upsample this is current pooling is a fraction < 1
  Visual explanation:
  [01][23][45][67] -> [0][1][2][3][4][5][6][7]
  For token [0] we calculate relative distances as follows:
  * 0 2 4 6
  However for token [1] we need relative distances changed by 1, specifically:
  * -1 1 3 5
  So we not only need to calculate the distances that corresponds to spacing
  between the keys but also for the ones in between because there are more than
  one query tokens (on different positions which means different relative
  distances) for single key token.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    context_bias_layer: Global context bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    location_bias_layer: Global location bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer.
    separate_cls: True/False if we separate_cls in calculations.
    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
      activations (based on query-key pairs) before dotting them with values.
    n_raw_tokens_generated: Number of tokens generated in a single pass through
      this layer. Used only in 'predict' non-training mode.
    max_inference_length: Maximum sequence length allowed in non-training
        modes.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  return cb.Serial(
      cb.Branch(
          PositionalEmbeddings(
              d_feature,
              separate_cls,
              total_kv_pooling,
              n_raw_tokens_generated=n_raw_tokens_generated,
              max_inference_length=max_inference_length,
              mode=mode), cb.Select([0]), cb.Select([1])),
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      context_bias_layer,
      location_bias_layer,
      RelativeAttention(  # pylint: disable=no-value-for-parameter
          total_kv_pooling=total_kv_pooling,
          separate_cls=separate_cls,
          n_heads=n_heads,
          dropout=dropout,
          n_raw_tokens_generated=n_raw_tokens_generated,
          max_inference_length=max_inference_length,
          chunk_len=chunk_len,
          chunk_offset=chunk_offset,
          mode=mode),
      core.Dense(d_feature),
  )


@assert_shape('bSq,blk,blv->bSd')
def RelativeAttentionLMLayer(d_feature,
                             context_bias_layer,
                             location_bias_layer,
                             total_kv_pooling,
                             separate_cls=False,
                             n_heads=1,
                             dropout=0.0,
                             n_raw_tokens_generated=1,
                             max_inference_length=3072,
                             chunk_len=None,
                             chunk_offset=None,
                             mode='train'):
  """Returns a layer that maps (q, k, v) to (activations).

  Same as standard Relative attention layer but additionally based on sizes
  of queries and keys prepares a mask that masks out the future.
  Masking the future is the concept primarily used for Language Modelling.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    context_bias_layer: Global context bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    location_bias_layer: Global location bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer.
    separate_cls: True/False if we separate_cls in calculations.
    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
      activations (based on query-key pairs) before dotting them with values.
    n_raw_tokens_generated: Number of tokens generated in a single pass through
      this layer. Used only in 'predict' non-training mode.
    max_inference_length: Maximum sequence length allowed in non-training
        modes.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  attention = RelativeAttentionLayer(
      d_feature,
      context_bias_layer,
      location_bias_layer,
      total_kv_pooling,
      separate_cls,
      n_heads=n_heads,
      dropout=dropout,
      n_raw_tokens_generated=n_raw_tokens_generated,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      mode=mode)

  return cb.Serial(
      AttentionMaskLayer(
          total_kv_pooling=total_kv_pooling,
          n_raw_tokens_generated=n_raw_tokens_generated,
          max_inference_length=max_inference_length,
          mode=mode),  # q, k, v, mask
      attention,  # vecs, mask
      cb.Select([0], n_in=2),  # vecs
  )


class RelativeAttention(base.Layer):
  """Relative attention.

  A layer that maps (location_bias, context_bias, pos_emb, q, k, v, mask)
  to (activations, mask).
  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:
    - splits queries, keys, and values into multiple 'heads',
    - splits positional embeddings into multiple 'heads',
    - computes per-head attention weights from per-head (queries, keys),
    - applies mask to screen out positions that come from padding tokens,
    - [in `'train'` mode] applies dropout to attention weights,
    - uses attention weights to combine per-head values vectors, and
    - merges per-head results into outgoing activations matching original input
      activation vector shapes.
  """

  def __init__(self,
               total_kv_pooling,
               separate_cls,
               n_heads=1,
               dropout=0.0,
               n_raw_tokens_generated=1,
               max_inference_length=3072,
               chunk_len=None,
               chunk_offset=None,
               mode='train'):
    """Returns a new PureAttention instance.

    Args:
      total_kv_pooling: Total shorten factor used in the model
      separate_cls: True/False if we separate_cls in calculations.
      n_heads: Number of attention heads.
      dropout: Probabilistic rate for dropout applied to attention strengths
        (based on query-key pairs) before applying them to values.
      n_raw_tokens_generated: Number of tokens generated in a single pass
        through this layer. Used only in 'predict' non-training mode.
      max_inference_length: Maximum sequence length allowed in non-training
        modes.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=7, n_out=2)
    self._total_kv_pooling = total_kv_pooling
    self._separate_cls = separate_cls
    self._n_heads = n_heads
    self._dropout = dropout
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._max_len = max_inference_length
    self._chunk_len = chunk_len
    self._chunk_offset = chunk_offset
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A (location_bias, context_bias, pos_emb, q, k, v, mask) tuple.
    """
    location_bias, context_bias, pos_emb, q, k, v, mask = inputs

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    if self._mode == 'predict':
      self._fast_inference_update_state((k, v), self.state)
      (k, v, _) = self.state

    per_head_results, dots = DotProductAttention(
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(q),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(k),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(v),
        pos_emb.reshape((-1, n_heads, d_feature // n_heads)),
        context_bias,
        location_bias,
        mask,
        separate_cls=self._separate_cls,
        dropout=self._dropout,
        mode=self._mode,
        rng=self.rng,
        chunk_len=self._chunk_len,
        chunk_offset=self._chunk_offset
    )
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(
        n_heads, merged_batch_and_head=False).forward(per_head_results)
    return merged_results, mask

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      cache_signature = input_signature[4:6]
      self.state = self._fast_inference_init_state(cache_signature)

  def _fast_inference_init_state(self, input_signature):
    """Returns an initial state for causal attention layer fast inference."""

    def zeros_for(bs, shape_dtype):
      shape, dtype = shape_dtype.as_tuple()
      d_feature = shape[-1]
      return jnp.zeros((bs, self._max_len, d_feature), dtype=dtype)

    batch_size = input_signature[0].shape[0]
    k = zeros_for(batch_size, input_signature[0])
    v = zeros_for(batch_size, input_signature[1])
    return k, v, jnp.array(0)

  def _fast_inference_update_state(self, inputs, state):
    """Updates state of a causal attention layer for fast inference.

    The layer state stores arrays with cached values of keys and values,
    as well as an index. To make shapes static, keys and values in the state are
    long, and the index indicates where the new keys and values from inputs need
    to be appended.

    During update, we append new_keys and new_values to keys and values at
    position given by index. And we increment index by length of new keys.
    We also create a mask to be 1 at appropriate positions (causal mask).

    Args:
      inputs: a double (new_keys, new_values)
      state: layer state with (keys, values, index)
    """
    # Fast inference: run step-by-step, storing the sequence
    # of keys and values calculated so far in state.
    new_k, new_v = inputs
    length = new_k.shape[1]
    (ks, vs, idx) = state

    assert length == 1

    # Keys and values are of shape [batch_size, length, d_kv].
    ks = fastmath.dynamic_update_slice_in_dim(
        ks, new_k, idx // self._total_kv_pooling, axis=1)
    vs = fastmath.dynamic_update_slice_in_dim(
        vs, new_v, idx // self._total_kv_pooling, axis=1)

    self.state = ks, vs, idx + self._n_raw_tokens_generated


def DotProductAttention(queries, keys, values, pos_emb, context_bias,
                        location_bias, mask, separate_cls, dropout, mode, rng,
                        chunk_len, chunk_offset=0):
  """Computes new activations via masked attention-weighted sum of values.

  This function is the core of the attention mechanism. It:
    - computes per-head attention weights from per-head `queries` and `keys`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights, and
    - uses attention weights to combine per-head `values` vectors.

  Args:
    queries: Per-head activations representing attention queries.
    keys: Per-head activations representing attention keys.
    values: Per-head activations to be combined by computed attention weights.
    pos_emb: Per-head activations representing positional embeddings.
    context_bias: Global context bias from Transformer XL's attention.
    location_bias: Global location bias from Transformer XL's attention.
    mask: Mask that distinguishes positions with real content vs. padding.
    separate_cls: True/False if we separate_cls in calculations.
    dropout: Probabilistic rate for dropout applied to attention strengths
      (based on query-key pairs) before applying them to values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Per-head activations resulting from masked per-head attention-weighted
    sum of per-head values.
  """
  bs, nh, original_l, d_feature = queries.shape

  def _calc_attn_scores(q, k):
    ac = jnp.einsum('bnid,bnjd->bnij', q + context_bias, k)
    bd = jnp.einsum('bnid,jnd->bnij', q + location_bias,
                    pos_emb[:2 * k.shape[2] - 1, ...])

    if mode != 'predict':
      bd = _fast_matrix_shift(bd)

    if separate_cls:
      # Masking out location part of attention for cls token
      bd = bd.at[:, :, :, 0].set(0)
      bd = bd.at[:, :, 0, :].set(0)

    dots = (ac + bd) / jnp.sqrt(d_feature)

    seq_len = q.shape[2]
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))

    # Softmax.
    dots = jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True))
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if dropout is not None and dropout > 0.0 and mode == 'train':
      keep = fastmath.random.bernoulli(rng, 1.0 - dropout, dots.shape)
      dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))

    return dots

  if chunk_len is None or mode == 'predict':
    dots = _calc_attn_scores(queries, keys)
    out = jnp.matmul(dots, values)
  else:
    n_chunks = original_l // chunk_len

    def chunk_split(v):
      chunked_shape = (bs, nh, n_chunks, chunk_len, d_feature)
      v = jnp.reshape(v, chunked_shape)
      v = v.swapaxes(1, 2)
      return jnp.reshape(v, (bs * n_chunks, nh, chunk_len, d_feature))

    def chunk_join(v):
      swapped_shape = (bs, n_chunks, nh, chunk_len, d_feature)
      v = jnp.reshape(v, swapped_shape)
      v = v.swapaxes(1, 2)
      return jnp.reshape(v, (bs, nh, original_l, d_feature))

    if chunk_offset == 0:
      queries, keys, values = map(chunk_split, [queries, keys, values])
      chunked_dots = _calc_attn_scores(queries, keys)
      chunked_result = jnp.matmul(chunked_dots, values)
      out = chunk_join(chunked_result)
    else:
      pre_len, post_len = chunk_offset, chunk_len - chunk_offset
      total_len = queries.shape[2]

      def split_along_l(v, mid_start, mid_end, end):
        pre = jnp.take(v, indices=range(mid_start), axis=2)
        mid = jnp.take(v, indices=range(mid_start, mid_end), axis=2)
        post = jnp.take(v, indices=range(mid_end, end), axis=2)
        return pre, mid, post

      def permute(v):
        pre, mid, post = split_along_l(v, pre_len, total_len - post_len,
                                       total_len)
        return jnp.concatenate([mid, pre, post], axis=2)

      def unpermute(v):
        mid, pre, post = split_along_l(v, total_len - chunk_len,
                                       total_len - post_len, total_len)
        return jnp.concatenate([pre, mid, post], axis=2)

      queries, keys, values = map(lambda x: chunk_split(permute(x)),
                                  [queries, keys, values])
      permuted_dots = _calc_attn_scores(queries, keys)
      permuted_out = chunk_join(jnp.matmul(permuted_dots, values))

      out = unpermute(permuted_out)

  out = out.astype(jnp.float32)
  return out, None  # We don't store full dots matrix


class PositionalEmbeddings(base.Layer):
  """Positional embedding for relative attention.

  Returns a layer that based on queries, keys and accumulated pool size of
  keys/values until this layer calculates sinusoidal positional embeddings
  for relative attention calculations.

  """

  def __init__(self,
               d_feature,
               separate_cls,
               total_kv_pooling,
               n_raw_tokens_generated=1,
               max_inference_length=3072,
               mode='train'):
    """The init method of positional embeddings.

    Args:
      d_feature: Depth/dimensionality of feature embedding.
      separate_cls: True/False if we separate_cls in calculations.
      total_kv_pooling: Accumulated pool size of keys/values until this layer.
      n_raw_tokens_generated: Number of tokens generated in a single pass
        through this layer. Used only in 'predict' non-training mode.
      max_inference_length: Maximum sequence length allowed in non-training
        modes.
      mode: One of `'train'`, `'eval'`, or `'predict'`.

    Returns:
      Positional embedding.
    """
    super().__init__(n_in=2, n_out=1)
    self._d_feature = d_feature
    self._separate_cls = separate_cls
    self._total_kv_pooling = total_kv_pooling
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._max_len = max_inference_length
    self._mode = mode

  def forward(self, inputs):
    queries, keys = inputs
    positions = self.PositionsVectors(queries, keys)
    pos_emb = Sinusoidal_Embeddings(positions, self._d_feature)
    return pos_emb

  def PositionsVectors(self, queries, keys):
    if self._mode == 'predict':
      assert queries.shape[1] == 1
      cur_token = self.state // self._total_kv_pooling
      positions = jnp.arange(0, self._max_len, 1.0) - cur_token
      positions = positions * self._total_kv_pooling
      self.state += self._n_raw_tokens_generated
      return positions

    is_funnel_layer = queries.shape != keys.shape
    keys_len, queries_len = keys.shape[1], queries.shape[1]
    current_pooling_ratio = keys_len / queries_len

    # Special case of upsampling
    if is_funnel_layer and current_pooling_ratio < 1:
      # We should not be doing standard upsampling when we use separate_cls
      # Cls token is being used for classification
      assert not self._separate_cls
      assert (self._total_kv_pooling * keys_len) % queries_len == 0
      multiplier = ((self._total_kv_pooling * keys_len) // queries_len)
      positions = jnp.arange(-queries_len + 1, queries_len, 1.0) * multiplier
    else:
      positions = jnp.arange(-keys_len + 1, keys_len, 1.0) * \
                  self._total_kv_pooling

    if is_funnel_layer and self._separate_cls:
      # For pool_size 2 without separating cls we have got
      # [0][1][2][3][4][5][6][7] -> [01][23][45][67]
      # With separating cls we have got
      # [0][1][2][3][4][5][6][7] -> [0][12][34][56]

      # First group always will always consist of one token after pooling
      # instead of (pool_size) tokens. We need to add proper offset so
      # that our shift later on in calculating attention works properly
      cls_offset = (current_pooling_ratio - 1) * self._total_kv_pooling
      positions = positions + cls_offset

    return positions

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      self.state = jnp.array(0)


def Sinusoidal_Embeddings(positions, d_feature):
  """Sinusoidal Embeddings.

    Computes out of 1-D integer absolute position vector the sinusoidal
  embeddings defined like in paper Attention is all you need (2017).
  Embeddings are shaped (positions, d_feature).

  Args:
    positions: a one-dimensional array of positions.
    d_feature: the number of sin-cos features.

  Returns:
    Positional embeddings.
  """
  inv_freq = 1 / (10000**(jnp.arange(0.0, d_feature, 2.0) / d_feature))
  sinusoid_freq = jnp.einsum('i,j->ij', positions, inv_freq)
  pos_emb = jnp.concatenate(
      [jnp.sin(sinusoid_freq), jnp.cos(sinusoid_freq)], axis=1)
  return pos_emb


def calc_funnel_ratio(keys_len, queries_len):
  """Calculates funnel ratio."""
  if queries_len > keys_len:  # Upsampling
    assert queries_len % keys_len == 0
    funnel_factor = queries_len // keys_len
    is_upsampling = True
  else:  # Downsampling
    assert keys_len % queries_len == 0
    funnel_factor = keys_len // queries_len
    is_upsampling = False

  return funnel_factor, is_upsampling


def _fast_matrix_shift(x):
  """Fast matrix shift.

  Implements necessary shift for relative positional attention calculations.
  Based on funnel_factor and information whether we perform upsampling
  or downsampling it calculates necessary shift and interval at which
  we pick correct values for attention.

  Args:
    x: matrix.
    funnel_factor: factor to be used for shift.
    is_upsampling: determines whether perform upsampling.

  Returns:
    Shifted matrix x.
  """
  #  shift: i-th row is shifted by i * shift elements to the left
  #  k: after shift, we pick every kth element

  k, shift = 1, 1

  bsz, n_head = x.shape[0], x.shape[1]
  qlen, klen = x.shape[2], (x.shape[3] + 1) // 2

  zero_pad = jnp.zeros((bsz, n_head, qlen, shift))
  x = jnp.concatenate([zero_pad, x], axis=3)
  x = x.reshape(bsz, n_head, 2 * klen - 1 + shift, qlen)
  x = x[:, :, shift:, :]
  x = x.reshape(bsz, n_head, qlen, klen * 2 - 1)
  x = x[:, :, :, shift - 1:shift - 1 + klen:k]
  return x


class AttentionMaskLayer(base.Layer):
  """Creates attention mask layer.

  Returns:
    Returns a layer that based on queries, keys and accumulated pool size of
    keys/values until this layer calculates positional embeddings for
    causal relative attention calculations.

    Takes as input q, k, v and appends proper mask in the end.

    Causal attention uses masking to prevent a given sequence position from
    attending to positions greater than / following it. This is used, for
    example, when training autoregressive sequence models, or when decoding a
    sequence symbol by symbol.
  """

  def __init__(self,
               total_kv_pooling=1,
               n_raw_tokens_generated=1,
               max_inference_length=3072,
               mode='train'):
    super().__init__(n_in=3, n_out=4)
    self._total_kv_pooling = total_kv_pooling
    self._max_len = max_inference_length
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._mode = mode

  def forward(self, inputs):
    q, k, v = inputs
    return q, k, v, self.calculate_mask(q, k)

  def calculate_mask(self, queries, keys):
    batch_size = queries.shape[0]
    keys_len, queries_len = keys.shape[-2], queries.shape[-2]
    funnel_factor, is_upsampling = calc_funnel_ratio(keys_len, queries_len)

    return self._funnel_mask(batch_size, keys_len, queries_len, funnel_factor,
                             is_upsampling)

  def _funnel_mask(self, batch_size, keys_len, queries_len, funnel_factor,
                   is_upsampling):
    """Creates a funnel mask.

    This function based on keys/queries lengths creates a triangle mask
    that prevents tokens from attending to positions following it.

    If funnel_factor is not equal to 1 due to funnel upsampling or
    downsampling it adjusts created mask for funnel attention
    by repeating each element funnel_factor times.

    This is because after funnel layer one token attends to funnel_factor
    different tokens in downsampling. During upsampling on the other hand
    funnel_factor tokens are attending to single token before upsampling.

    Args:
      batch_size: batch size.
      keys_len: keys length.
      queries_len: queries length.
      funnel_factor: funnel factor.
      is_upsampling: upsampling if set to True.

    Returns:
      Funnel mask.
    """

    if self._mode == 'predict':
      # We cannot generate more than one token because it contradicts
      # all autoregressive properties
      assert queries_len == 1
      mask = jnp.arange(self._max_len) <= (self.state // self._total_kv_pooling)
      mask = jnp.reshape(mask, (1, 1, 1, self._max_len))
      mask = jnp.repeat(mask, batch_size, axis=0)
      self.state += self._n_raw_tokens_generated
      return mask

    if funnel_factor != 1:
      if not is_upsampling:
        mask = jnp.tril(jnp.ones((queries_len, queries_len), dtype=jnp.bool_))
        mask = jnp.repeat(mask, funnel_factor, axis=-1)
      else:
        mask = jnp.tril(jnp.ones((keys_len, keys_len), dtype=jnp.bool_))
        mask = jnp.repeat(mask, funnel_factor, axis=-2)
    else:
      mask = jnp.tril(jnp.ones((queries_len, queries_len), dtype=jnp.bool_))

    return jnp.repeat(mask[None, None, :, :], batch_size, axis=0)

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      self.state = jnp.array(0)


@assert_shape('...d->...d')
def ShiftRightCls(cls_id):
  """Shifts right.

  Returns a layer that shifts input tokens to the right by one
  and inserts an cls token to the beginning like in BERT paper.

  Args:
    cls_id: id of the cls token in embedding dictionary.
  Returns:
    shift_right layer.
  """

  def shift_right(x):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(cls_id))
    return padded[:, :-1]

  return cb.Fn('ShiftRightCls()', shift_right)
