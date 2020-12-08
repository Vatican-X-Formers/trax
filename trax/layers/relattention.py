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
"""Attention-related layers.

Attention is a powerful extension of basic neural network ideas.
In a classic neural network:

    - node activations are floating point values (one float per node), and
    - inter-node connections are trainable weights (one float per connection).

Attention assembles networks of *vectors* and uses vector calculations to
derive connection strength; in other words:

    - node activations are floating point vectors, and
    - inter-node connections come from trainable vector computations.

Attention thus involves extra concepts/mechanisms -- queries, keys, values,
masks, attention heads -- that factor heavily into this module's API. See
specific classes and functions for details.

NOTE: Attention layers in this module include `mode`-dependent behavior.
The possible modes are:

    - `'train'`: in training -- dropouts and position shifts active
    - `'eval'`:  in evals -- dropouts inactive, position shifts active
    - `'predict'`: in prediction -- dropouts and position shifts inactive
"""

import jax._src.ops
import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.assert_shape import assert_shape
from trax.layers.attention import SplitIntoHeads, MergeHeads
from trax.fastmath.ops import index_update

#TODO assert_shape

# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name

# TODO Dopisz testy tutaj
def PositionalEmbeddings(d_feature, separate_cls, total_pooling):
    """Returns a layer that based of queries and keys and a position
     in the funnel transformer computes postional embeddings for
     relative attention calculations.

    Args:
        d_feature: Depth/dimensionality of feature embedding.
        separate_cls: If `True`, pooling in funnel blocks is not applied to
            embeddings of the first token (`cls` from BERT paper) and only final
            embedding of this token is used for categorization - the rest are
            discarded. If `False`, each token from the beginning is pooled and
            all embeddings are averaged and mapped to output categories like in
            original `TransformerEncoder` model.
        total_pooling: The combined pool size of previously used funnel blocks.
    """
    def CalculatePositionalEmbeddings(queries, keys):
        is_funnel_layer = queries.shape != keys.shape
        keys_len = keys.shape[1]
        positions = np.arange(-keys_len + 1, keys_len, 1.0) * total_pooling

        if is_funnel_layer and separate_cls:
            # We consider cls as first token in the block that precedes
            # the first block of actual tokens;
            # Let's say tokens start at index 1 then cls_id should have id
            # of (1 - total_pooling_ration) in that sequence.
            # Because we are separating cls_token from the other tokens
            # we need to add proper offset so our shift in computing
            # attention later on would work properly
            single_pooling_ratio = keys.shape[1] // queries.shape[1]
            cls_offset = (single_pooling_ratio - 1) * total_pooling
            positions = positions + cls_offset

        return EncodeSequence(positions)

    def EncodeSequence(positions):
        inv_freq = 1 / (10000 ** (np.arange(0.0, d_feature, 2.0) / d_feature))
        sinusoid_freq = np.einsum('i,j->ij', positions, inv_freq)
        pos_emb = np.concatenate([np.sin(sinusoid_freq),
                                  np.cos(sinusoid_freq)], axis=1)
        return pos_emb

    return cb.Fn('Positional embeddings', CalculatePositionalEmbeddings,
                 n_out=1)


def RelativeAttentionLayer(d_feature, context_bias_layer, location_bias_layer,
                           separate_cls, total_pooling,
                           n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (q, k, v, mask) to
                                  (activations, pos_emb, biases, mask).

  See `Attention` above for further context/details.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  return cb.Serial(
      cb.Branch(PositionalEmbeddings(d_feature, separate_cls, total_pooling),
                cb.Select([0]),
                cb.Select([1])),
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      context_bias_layer,
      location_bias_layer,
      RelativeAttention(  # pylint: disable=no-value-for-parameter
          separate_cls=separate_cls, n_heads=n_heads,
          dropout=dropout, mode=mode),
      core.Dense(d_feature),
  )


# 'k' is number of keys/values, while 'l' is number of queries. Typically they
# will be the same, but it is not necessary.
class RelativeAttention(base.Layer):
  """Returns a layer that maps (q, k, v, mask) to (activations, mask).

  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:

    - splits queries, keys, and values into multiple 'heads',
    - computes per-head attention weights from per-head (queries, keys),
    - applies mask to screen out positions that come from padding tokens,
    - [in `'train'` mode] applies dropout to attention weights,
    - uses attention weights to combine per-head values vectors, and
    - merges per-head results into outgoing activations matching original input
      activation vector shapes.
  """

  def __init__(self, separate_cls, n_heads=1, dropout=0.0, mode='train'):
    """Returns a new PureAttention instance.

    Args:
      n_heads: Number of attention heads.
      dropout: Probababilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=7, n_out=2)
    self._separate_cls = separate_cls
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A (queries, keys, values, mask) tuple.
    """
    location_bias, context_bias, pos_emb, q, k, v, mask = inputs

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

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
        rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(n_heads, merged_batch_and_head=False).forward(
        per_head_results)
    return merged_results, mask


def DotProductAttention(queries, keys, values, pos_emb, context_bias,
                        location_bias, mask, separate_cls, dropout, mode, rng):
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
      pos_emb: Per-head activations representing relative positions.
      context_bias: Global context bias from Transformer XL's attention.
      location_bias: Global location bias from Transformer XL's attention.
      mask: Mask that distinguishes positions with real content vs. padding.
      dropout: Probababilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      Per-head activations resulting from masked per-head attention-weighted
      sum of per-head values.
    """
    # queries, keys, values are shape (batch_size, n_heads, seq_len, d_head)
    # position embeddings are shape (2 * keys_len - 1, n_heads, d_head)
    # bias vectors are shape (1, n_heads, 1, d_head)
    # mask is shape (batch_size, 1, 1, keys_len)
    d_feature = queries.shape[-1]
    funnels_shift = keys.shape[-2] // queries.shape[-2]

    # Compute parts of attention score
    # Shapes of AC and BD are (batch_size, n_heads, queries_len, keys_len)
    ac = jnp.einsum('bnid,bnjd->bnij', queries + context_bias, keys)
    bd = jnp.einsum('bnid,jnd->bnij', queries + location_bias, pos_emb)
    bd = _fast_matrix_shift(bd, shift=funnels_shift)

    if separate_cls:
      # Masking out location part of attention for cls token
      bd = bd.at[:, :, :, 0].set(0)
      # bd = index_update(bd, jax.ops.index[:, :, :, 0], 0)
      # bd = index_update(bd, jax.ops.index[:, :, 0, :], 0)

    dots = (ac + bd) / jnp.sqrt(d_feature)
    if mask is not None:
        dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
    # Softmax.
    dots = jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True))
    if dropout >= 1.0:
        raise ValueError('Dropout rates must be lower than 1.')
    if dropout is not None and dropout > 0.0 and mode == 'train':
        keep = fastmath.random.bernoulli(rng, 1.0 - dropout, dots.shape)
        dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))
    out = jnp.matmul(dots, values)
    out = out.astype(jnp.float32)
    dots = dots.astype(jnp.float32)
    return out, dots


def _fast_matrix_shift(x, shift):
  # This function shifts i-th row by i * shift elements
  # to the left. It implemenets necessary shift for
  # relative positional attention calculation
  # decribed in Transformer XL paper

  bsz, n_head = x.shape[0], x.shape[1]
  qlen, klen = x.shape[2], (x.shape[3] + 1) // 2

  zero_pad = jnp.zeros((bsz, n_head, qlen, shift))
  x = jnp.concatenate([zero_pad, x], axis=3)
  x = x.reshape(bsz, n_head, 2 * klen - 1 + shift, qlen)
  x = x[:, :, shift:, :]
  x = x.reshape(bsz, n_head, qlen, klen * 2 - 1)
  x = x[:, :, :, shift - 1: shift - 1 + klen]
  return x


@assert_shape('...d->...d')
def ShiftRightCls(cls_id):
  """Returns a layer that can insert padding to shift the input sequence.
  Args:
    cls_id: id of cls token in embedding
  """
  def f(x):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)  # Padding on axis.
    padded = jnp.pad(x, pad_widths, mode='constant',
                     constant_values=x.dtype.type(cls_id))
    return padded[:, :-1]
  return cb.Fn('ShiftRightCls()', f)