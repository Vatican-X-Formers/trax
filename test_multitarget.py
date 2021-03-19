from trax.supervised.training import MultiTargetTrainTaskTFDS, TargetHandler, Loop
import trax.data as td
import trax.layers as tl
from trax import optimizers
from trax.models.research.funnel_transformer import FunnelTransformerLM

gen_handler = TargetHandler(
    input_name='image',
    target_name='image',
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=optimizers.Adam(),
    preprocess_fn=td.tf_inputs.cifar10_gen_cls
)

cls_handler = TargetHandler(
    input_name='image',
    target_name='label',
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=optimizers.Adam(),
    preprocess_fn=td.tf_inputs.cifar10_gen_cls
)

mt_train_task = MultiTargetTrainTaskTFDS(
    'cifar10',
    [gen_handler, cls_handler]
)

model = FunnelTransformerLM(
    vocab_size=256,
    d_model=16,
    d_ff=64
)

training_loop = Loop(model,
                     mt_train_task.tasks,
                     eval_tasks=None,
                     output_dir='./tmp')

training_loop.run(5000)
