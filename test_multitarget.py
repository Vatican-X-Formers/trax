from trax.supervised.training import MultiTargetTrainTaskTFDS, TargetHandler, Loop
import trax.data as td
import trax.layers as tl
from trax import optimizers
from trax.models.research.funnel_transformer import FunnelTransformerLM
import tensorflow as tf


def cifar10_gen(dataset, training):
  del training

  def cast_image(arg):
    fst, snd = arg
    fst = tf.cast(fst, tf.float32) / 255.0
    snd = tf.cast(snd, tf.float32) / 255.0
    fst = tf.cast(fst, tf.float32) / 255.0
    snd = tf.cast(snd, tf.float32) / 255.0
    return fst, snd

  def flat_image(arg):
    fst, snd = arg
    return tf.cast(tf.reshape(fst, [-1]), tf.int64), tf.cast(tf.reshape(snd, [-1]), tf.int64)

  #print('debug', next(dataset))
  dataset = map(cast_image, dataset)
  dataset = map(flat_image, dataset)

  return dataset

gen_handler = TargetHandler(
    input_name='image',
    target_name='image',
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=optimizers.Adam(),
    preprocess_fn=cifar10_gen
)

cls_handler = TargetHandler(
    input_name='image',
    target_name='label',
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=optimizers.Adam(),
    preprocess_fn=cifar10_gen
)

mt_train_task = MultiTargetTrainTaskTFDS(
    'cifar10',
    [gen_handler]
)

model = FunnelTransformerLM(
    vocab_size=256,
    d_model=16,
    d_ff=64
)

training_loop = Loop(model,
                     mt_train_task.train_tasks,
                     eval_tasks=mt_train_task.eval_tasks,
                     output_dir='./tmp')

training_loop.run(5000)
