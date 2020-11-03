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

import jax
import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn
from trax.layers.attention import SplitIntoHeads, MergeHeads


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name

def RelativeAttentionLayer(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (q, k, v, pos_emb, biases, mask) to
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
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      RelativeAttention(  # pylint: disable=no-value-for-parameter
          n_heads=n_heads, dropout=dropout, mode=mode),
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

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    """Returns a new PureAttention instance.

    Args:
      n_heads: Number of attention heads.
      dropout: Probababilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=6, n_out=4)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A (queries, keys, values, mask) tuple.
    """
    q, k, v, pos_emb, biases, mask = inputs

    # comparing sequence lengths
    if q.shape[-2] != k.shape[-2]:
      stride = 2
    else:
      stride = 1

    # adjusting pos_emb to currently pooled sequence
    while pos_emb.shape[0] != k.shape[-2]:
      pos_emb = pos_emb[1::2, :, :]

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    per_head_results, dots = DotProductRelativeAttention(
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(q),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(k),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(v),
        pos_emb,
        biases[0],
        biases[1],
        stride,
        mask,
        dropout=self._dropout,
        mode=self._mode,
        rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(n_heads, merged_batch_and_head=False).forward(
        per_head_results)
    return (merged_results, pos_emb, biases, mask)


def DotProductRelativeAttention(queries, keys, values, pos_emb, u_bias, v_bias,
                                stride, mask, dropout, mode, rng):
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
    u_bias: Global u-type bias from relative attention
    v_bias: Global v-type bias from relative attention
    stride: Factor by which keys vector is larger than queries vector
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
    # poition embeddings are shape (2 * keys_len - 1, n_heads, d_head)
    # bias vectors are shape (1, n_heads, 1, d_head)
    # mask is shape (batch_size, 1, 1, queries_len)
    d_feature = queries.shape[-1]

    # Compute parts of attention score
    # Output shapes of AC and BD are (batch_size, n_heads, queries_len, keys_len)
    ac = jnp.einsum('bnid,bnjd->bnij', queries + u_bias, keys)
    bd = jnp.einsum('bnid,jnd->bnij', queries + v_bias, pos_emb)
    bd = _fast_matrix_shift(bd, stride=stride)

    dots = ac + bd / jnp.sqrt(d_feature)  # maybe divide by sqrt(4 * d_feature)?
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


def _fast_matrix_shift(x, stride=3):
    # bsz x n_head x qlen x klen
    # along qlen dim keys are in ascending order 0, 1, ..., qlen - 1
    # along klen dim rel positioning is in order -
    # - (klen - 1), - (klen - 2), ..., -1, 0, 1, ..., klen - 2, klen - 1

    assert stride <= 2
    bsz, n_head = x.shape[0], x.shape[1]
    qlen, klen = x.shape[2], (x.shape[3] + 1) // 2

    zero_pad = jnp.zeros((bsz, n_head, qlen, stride))
    x = jnp.concatenate([zero_pad, x], axis=3)
    x = x.reshape(bsz, n_head, 2 * klen - 1 + stride, qlen)
    x = x[:, :, 1:, :] if stride == 1 else x[:, :, 1:-1, :]
    x = x.reshape(bsz, n_head, qlen, klen * 2 - 1)
    x = x[:, :, :, (klen - qlen + stride - 1): (klen - qlen + stride - 1) + klen]
    return x
