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
from trax.models.research.funnel_transformer import UFunnel, _UFunnelValley

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

from absl.testing import absltest
from absl.testing import parameterized

from trax import shapes
from trax.models.research.funnel_transformer import UFunnel


class FunnelTransformerTest(parameterized.TestCase):

    def test_ufunnel_forward_shape_flat(self):
        vocab_size = 16
        tokens = 3*2*2*2*2*2
        model = UFunnel(
            vocab_size, d_model=32, d_ff=64,
            n_heads=2, segment_lengths=(2,),
            use_conv=True)
        x = np.ones((3, tokens)).astype(np.int32)
        _, _ = model.init(shapes.signature(x))
        y = model(x)
        self.assertEqual(y.shape, (3, tokens, vocab_size))

    def test_ufunnel_forward_shape_deep(self):
        vocab_size = 16
        tokens = 3*2*2*2*2*2
        model = UFunnel(
            vocab_size, d_model=32, d_ff=64,
            n_heads=2, segment_lengths=(2,2),
            use_conv=True)
        x = np.ones((3, tokens)).astype(np.int32)
        _, _ = model.init(shapes.signature(x))
        y = model(x)
        self.assertEqual(y.shape, (3, tokens, vocab_size))


if __name__ == '__main__':
    absltest.main()
