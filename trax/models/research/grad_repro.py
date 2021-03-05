from absl import logging
import gin
import trax
import shutil
import os

# from jax.config import config
# config.update('jax_disable_jit', True)

config = '''
import trax.models
import trax.optimizers
import trax.data.inputs
import trax.data.tf_inputs
import trax.supervised.trainer_lib

vocab_size = 13  #  For addition, base = vocab_size - 3.
max_len = 8
twice_max_len = 16

# Parameters for data_streams:
# ==============================================================================
data_streams.data_dir = None
data_streams.dataset_name = 'copy'

# Parameters for sequence_copy_inputs:
# ==============================================================================
sequence_copy_inputs.vocab_size = %vocab_size
sequence_copy_inputs.batch_size = 1
sequence_copy_inputs.train_length = %max_len
sequence_copy_inputs.eval_min_length = 4
sequence_copy_inputs.eval_max_length = %max_len
sequence_copy_inputs.reverse = False
sequence_copy_inputs.pad_to_multiple = %max_len

# Parameters for multifactor:
# ==============================================================================
multifactor.constant = 0.05
multifactor.factors = 'constant * linear_warmup * rsqrt_decay'
multifactor.warmup_steps = 4000

# Parameters for FunnelTransformerLM:
# ==============================================================================
FunnelTransformerLM.d_model = 4
FunnelTransformerLM.d_ff = 1
FunnelTransformerLM.dropout = 0.0
FunnelTransformerLM.n_heads = 1
FunnelTransformerLM.vanilla_layers=(0, 0)
FunnelTransformerLM.shorten_factors=(2, 2)
FunnelTransformerLM.n_funnel_blocks=(0, 0)
FunnelTransformerLM.vocab_size = %vocab_size

# Parameters for train:
# ==============================================================================
train.inputs = @trax.data.inputs.sequence_copy_inputs
# train.inputs = @trax.data.inputs.addition_inputs
train.eval_frequency = 5000
train.eval_steps = 4
train.optimizer = @trax.optimizers.Adam
train.steps = 1
train.model = @trax.models.FunnelTransformerLM
'''

if __name__ == '__main__':
  # train_dir = 'TRAIN_DIR'
  #
  # shutil.rmtree(train_dir)
  # os.makedirs(train_dir)

  gin.parse_config(config)
  logging.set_verbosity('debug')

  # logging.set_verbosity('info')
  train = trax.supervised.train(output_dir=train_dir)
