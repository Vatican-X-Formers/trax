import gin

gin.parse_config("""
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

import trax.layers
import trax.models
import trax.optimizers
import trax.data.tf_inputs
import trax.supervised.trainer_lib

# Parameters for batcher:
# ==============================================================================
batcher.data_streams = @tf_inputs.data_streams
batcher.batch_size_per_device = 1
batcher.eval_batch_size = 8
batcher.max_eval_length = 3072  # 32 * 32 * 3
batcher.variable_shapes = False

# Parameters for data_streams:
# ==============================================================================
data_streams.data_dir = None
data_streams.dataset_name = 'downsampled_imagenet/32x32'
data_streams.input_name = 'image'
data_streams.target_name = 'image'
data_streams.bare_preprocess_fn = @trax.data.tf_inputs.downsampled_imagenet_flatten_bare_preprocess

# Parameters for multifactor:
# ==============================================================================
# 0.0442 ~= 512^-0.5 = d_model^-0.5
multifactor.constant = 0.0442
multifactor.factors = 'constant * linear_warmup * rsqrt_decay'
multifactor.warmup_steps = 0

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate = 0.0
Adam.b1 = 0.9
Adam.b2 = 0.98
Adam.eps = 1e-09

# Parameters for train:
# ==============================================================================
train.eval_frequency = 2
train.eval_steps = 2
train.model = @trax.models.FunnelTransformerLM
train.optimizer = @trax.optimizers.Adam
train.steps = 1
train.checkpoints_at = \
    []
train.permanent_checkpoints_at = []

# Parameters for FunnelTransformerLM:
# ==============================================================================
FunnelTransformerLM.d_ff = 16
FunnelTransformerLM.d_model = 4
FunnelTransformerLM.dropout = 0.04
FunnelTransformerLM.n_funnel_blocks = (2,)
FunnelTransformerLM.n_heads = 4
FunnelTransformerLM.shorten_factors = (3,)
FunnelTransformerLM.vanilla_layers = (1, 1)
FunnelTransformerLM.vocab_size = 8
FunnelTransformerLM.ff_activation = @trax.layers.FastGelu
""")

import os
import trax
output_dir = os.path.expanduser('~/train_dir/')
train = trax.supervised.train(output_dir=output_dir)