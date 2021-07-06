# coding=utf-8
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

"""Write Summaries from JAX for use with Tensorboard.

See jaxboard_demo.py for example usage.
"""
import os
import warnings

import gin
import matplotlib as mpl

# Necessary to prevent attempted Tk import:
from trax.jaxboard import SummaryWriter

with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  mpl.use('Agg')
# pylint: disable=g-import-not-at-top
import neptune.new as neptune


class NeptuneRunWrapper:
  def __init__(self):
    neptune_project = os.environ['NEPTUNE_PROJECT']
    self._run = neptune.init(project=neptune_project,
                             api_token=os.environ['NEPTUNE_TOKEN'])

    self._run['TRAX_BRANCH'] = os.environ['TRAX_BRANCH']
    self._run['gin_config'] = gin.operative_config_str()
    self._run['parameters'] = gin.config._CONFIG

  def log_value(self, tag, value, step):
    self._run[tag].log(value)


class SummaryWriterWithNeptune(SummaryWriter):
  """Saves data in event and summary protos for tensorboard."""

  def __init__(self, log_dir, enable=True,
               neptune_run: NeptuneRunWrapper = None):
    """Create a new SummaryWriter.

    Args:
      log_dir: path to record tfevents files in.
      enable: bool: if False don't actually write or flush data.  Used in
        multihost training.
    """
    super().__init__(log_dir, enable)
    self._neptune_run = neptune_run

  def scalar(self, tag, value, step=None):
    """Saves scalar value.

    Args:
      tag: str: label for this data
      value: int/float: number to log
      step: int: training step
    """
    super().scalar(tag, value, step)
    self._neptune_run.log_value(tag, value, step)
