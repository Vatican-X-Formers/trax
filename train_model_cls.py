import trax
import gin
import os


gin.parse_config_file('funnel_imagenet32.gin')
output_dir = os.path.expanduser('~/train_dir/')
train = trax.supervised.train(output_dir=output_dir)