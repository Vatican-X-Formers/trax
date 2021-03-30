import trax
import gin
import os
from trax.models import FunnelTransformerLM

"""
STEPS
1. Load and go with im32 reg
2. Load and go with cf10 reg
3. Load, replace weights and go with cf10 cls
"""

"""
FunnelTransformerLM.d_ff = 256
FunnelTransformerLM.d_model = 64
FunnelTransformerLM.dropout = 0.04
FunnelTransformerLM.ff_activation = @trax.layers.FastGelu
FunnelTransformerLM.n_funnel_blocks = (12,)
FunnelTransformerLM.n_heads = 1
FunnelTransformerLM.shorten_factors = (3,)
FunnelTransformerLM.vanilla_layers = (1, 1)
FunnelTransformerLM.vocab_size = 256
"""

model = FunnelTransformerLM(
    d_ff = 256,
    d_model = 64,
    dropout = 0.04,
    ff_activation = trax.layers.FastGelu,
    n_funnel_blocks = (12,),
    n_heads = 1,
    shorten_factors = (3,),
    vanilla_layers = (1,1),
    vocab_size = 256
)

def vatican_stream():
    streams = trax.data.tf_inputs.data_streams(
        data_dir = None,
        dataset_name = 'downsampled_imagenet/32x32',
        input_name = 'image',
        target_name = 'image',
        bare_preprocess_fn = trax.data.tf_inputs.downsampled_imagenet_flatten_bare_preprocess
    )

    return trax.data.inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=1,
                        eval_batch_size=1)

train_stream = itertools.cycle(vatican_stream().train_stream(1))
eval_stream = itertools.cycle(vatican_stream().eval_stream(1))

reg_task = trax.supervised.training.TrainTask(
    labeled_data=train_stream,
    loss_layer=trax.layers.CategoryCrossEntropy,
    optimizer=trax.optimizers.adam.Adam(0.001),
)
reg_eval_task = trax.supervised.training.EvalTask(
    labeled_data=eval_stream,
    metrics=[trax.layers.CategoryCrossEntropy]
)

training_session = training.Loop(
    model,
    tasks=[reg_task],
    eval_tasks=[reg_eval_task],
    eval_at=lambda step_n: step_n % 20 == 0,
)
