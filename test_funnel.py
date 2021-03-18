import trax
import gin
import os

gin.parse_config_file('funnel_cls_gen.gin')
output_dir = os.path.expanduser('~/train_dir/')
#!rm -f ~/train_dir/model.pkl.gz  # Remove old model
train = trax.supervised.train(output_dir=output_dir)
