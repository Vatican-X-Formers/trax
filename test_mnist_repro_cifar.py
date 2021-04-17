import trax.layers as tl
from trax.optimizers import adam
from trax.models.research.funnel_transformer import FunnelTransformerLM
import itertools
from trax.supervised import training
from trax.data import tf_inputs, inputs
import tensorflow as tf

def cast_image(img):
    return tf.cast(img, tf.float32) / 255.0

def flat_image(img):
    return tf.cast(tf.reshape(img, [-1]), tf.int64)

def _cifar_reg_preprocess():
  def preprocess_stream(stream):
    def new_stream():
      for (image, _) in stream():
        _img = flat_image(cast_image(image))
        yield _img, _img
    return new_stream

  streams = tuple(map(preprocess_stream, tf_inputs.data_streams('cifar10', input_name='image',
                                   target_name='image')))

  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=1,
                        eval_batch_size=1)

def _mnist_brightness_dataset():
  """Loads (and caches) a MNIST mean brightness data set."""
  def preprocess_stream(stream):
    def new_stream():
      for (image, _) in stream():
        yield (image, (image / 255).mean()[None])
    return new_stream

  streams = tuple(map(preprocess_stream, tf_inputs.data_streams('mnist')))
  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=256,
                        eval_batch_size=256)

def _build_model(two_heads):
  cls_head = tl.Dense(10)
  if two_heads:
    reg_head = tl.Dense(1)
    heads = tl.Branch(cls_head, reg_head)
  else:
    heads = cls_head
  return tl.Serial(
      tl.Fn('ScaleInput', lambda x: x / 255),
      tl.Flatten(),
      tl.Dense(512),
      tl.Relu(),
      tl.Dense(512),
      tl.Relu(),
      heads,
  )

mnist_model = _build_model(False)

model = FunnelTransformerLM(
    vocab_size=256,
    d_model=16,
    d_ff=64
)

reg_task = training.TrainTask(
    _mnist_brightness_dataset().train_stream(1),
    tl.L2Loss(), #tl.Serial(tl.Select([1]), tl.L2Loss()),
    adam.Adam(0.001),
)

training_session = training.Loop(
    mnist_model,
    tasks=[reg_task],
)

training_session.run(n_steps=1)
