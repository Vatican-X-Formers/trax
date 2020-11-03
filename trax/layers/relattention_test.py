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
"""Tests for trax.layers.relattention."""

from absl.testing import absltest
import numpy as np

from trax import shapes

import trax.layers as tl
import trax.layers.relattention as relattention

class RelAttentionTest(absltest.TestCase):

  def test_simple_call(self):
    layer = tl.CausalAttention(d_feature=4, n_heads=2)
    x = [np.array([[[2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],]]),
         np.array([[[[1, 0, 1]]]])]
    _, _ = layer.init(shapes.signature(x))

    y, mask = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))
    self.assertEqual(mask.shape, (1, 1, 1, 3))

  def test_fast_shift_matrix_stride_1(self):
    layer = relattention._fast_matrix_shift
    x = np.array([[[[-3., -2., -1.,  0.,  1.,  2.,  3.],
                    [-3., -2., -1.,  0.,  1.,  2.,  3.]]]]).astype(np.float32)

    y = layer(x, stride=1)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[[0.,  1.,  2.,  3.],
                                      [-1.,  0.,  1.,  2.]]]])

  def test_fast_shift_matrix_stride_2(self):
    layer = relattention._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.]]]]).astype(np.float32)

    y = layer(x, stride=2)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[[0., 1., 2., 3.],
                                       [-2., -1., 0., 1.]]]])


if __name__ == '__main__':
  absltest.main()
