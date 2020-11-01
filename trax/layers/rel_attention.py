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


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name

@assert_shape('bSq,blk,blv,b1xl->bSd,b1xl')
def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (q, k, v, mask) to (activations, mask).

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
      ),
      PureAttention(  # pylint: disable=no-value-for-parameter
          n_heads=n_heads, dropout=dropout, mode=mode),
      core.Dense(d_feature),
  )