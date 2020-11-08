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
"""Tests for Funnel-Transformer models."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from trax import layers as tl, shapes
from trax.models.research.funnel_transformer import PoolLayer, \
  _FunnelResidualBlock, FunnelTransformerEncoder, FunnelTransformer, MaskPool


class FunnelTransformerTest(parameterized.TestCase):

  def test_mean_pool(self):
    x = np.ones((1, 4, 1))
    x[0, :3, 0] = [5., 2., 4.]

    pooling = PoolLayer(tl.AvgPool, (2,), (2,))
    y = pooling(x)

    self.assertEqual(y.shape, (1, 2, 1))
    self.assertEqual(y.tolist(), [[[5.], [3.]]])

  def test_mask_pool(self):
    x = np.array([1, 0, 0, 1], dtype=bool).reshape((1, 1, 1, 4))
    pooling_cls = MaskPool((2,), (2,))
    y1 = pooling_cls(x)

    self.assertEqual(y1.shape, (1, 1, 1, 2))
    self.assertEqual(y1.squeeze().tolist(), [True, False])

    pooling_without_cls = MaskPool((2,), (2,), separate_cls=False)
    y2 = pooling_without_cls(x)

    self.assertEqual(y2.shape, (1, 1, 1, 2))
    self.assertEqual(y2.squeeze().tolist(), [True, True])

  def test_funnel_block_forward_shape(self):
    n_even = 4
    d_model = 8

    x = np.ones((1, n_even, d_model), dtype=np.float)
    mask = np.ones((1, n_even), dtype=np.int32)

    masker = tl.PaddingMask()
    mask = masker(mask)

    block = tl.Serial(
        _FunnelResidualBlock(d_model, 8, 2, 0.1, None, 'train', tl.Relu,
                             tl.AvgPool, (2,), (2,)))

    xs = [x, mask]
    _, _ = block.init(shapes.signature(xs))

    y, _ = block(xs)

    self.assertEqual(y.shape, (1, n_even // 2, d_model))

  def test_funnel_transformer_encoder_forward_shape(self):
    n_classes = 5
    model = FunnelTransformerEncoder(2, n_classes=n_classes, d_model=8,
                                     d_ff=8, encoder_segment_lengths=(1, 1),
                                     n_heads=2, max_len=8)

    batch_size = 2
    n_tokens = 4
    x = np.ones((batch_size, n_tokens), dtype=np.int32)
    _ = model.init(shapes.signature(x))
    y = model(x)

    self.assertEqual(y.shape, (batch_size, n_classes))

  def test_funnel_transformer_forward_shape(self):
    d_model = 8
    model = FunnelTransformer(2, d_model=d_model, d_ff=8,
                              encoder_segment_lengths=(1, 1),
                              n_decoder_blocks=1, n_heads=2, max_len=8)

    batch_size = 2
    n_tokens = 4
    x = np.ones((batch_size, n_tokens), dtype=np.int32)
    _ = model.init(shapes.signature(x))
    y = model(x)

    self.assertEqual(y.shape, (batch_size, n_tokens, d_model))


if __name__ == '__main__':
  absltest.main()
