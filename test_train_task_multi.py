import trax.layers as tl
from trax.optimizers import adam
from trax.models.research.funnel_transformer import FunnelTransformerLM
import itertools
from trax.supervised import training
from trax.data import tf_inputs, inputs

def _dataset():
  """Loads (and caches) the standard MNIST data set."""
  streams = tf_inputs.data_streams(dataset_name='cifar10',
                                   input_name='image',
                                   target_name='image',
                                   preprocess_fn=tf_inputs.cifar10_gen_cls)

  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=1,
                        eval_batch_size=1)


model = FunnelTransformerLM(
    vocab_size=256,
    d_model=16,
    d_ff=64
)

#(cls_task, cls_eval_task) = _mnist_tasks(head=tl.Select([0], n_in=2))
reg_task = training.TrainTask(
    _dataset().train_stream(1),
    #itertools.cycle(_mnist_brightness_dataset().train_stream(1)),
    loss_layer=tl.WeightedCategoryCrossEntropy(),#tl.Serial(tl.Select([1]), tl.L2Loss()),
    optimizer=adam.Adam(0.001)
)

#reg_eval_task = training.EvalTask(
#    itertools.cycle(_dataset().train_stream(1)),
#    #itertools.cycle(_mnist_brightness_dataset().eval_stream(1)),
#    tl.CategoryCrossEntropy(),#[tl.Serial(tl.Select([1]), tl.CategoryCrossEntropy())],
#    n_eval_batches=1,
#    metric_names=['L2'],
#)

training_session = training.Loop(
    model,
    tasks=[reg_task], #[cls_task, reg_task],
    eval_tasks=None,#[cls_eval_task, reg_eval_task],
    eval_at=lambda step_n: step_n % 20 == 0,
    #which_task=lambda step_n: step_n % 2,
)

training_session.run(n_steps=1)
