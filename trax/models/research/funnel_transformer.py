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

# Lint as: python3
"""Relformer model.
"""
import functools
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers.research.rel_attention import RelativeAttentionWrapper
from trax.models.reformer.reformer import DecoderBlock
from trax.models.research.configurable_transformer import PositionalEncoder


def _UpsamplerLM(shorten_factor, d_model):
  return tl.Serial(
      tl.Dense(shorten_factor * d_model),
      tl.Fn(
          'ProlongBack',
          lambda x: jnp.reshape(  # Prolong back.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] * shorten_factor, -1)),
          n_out=1),
  )


def _DownsamplerLM(shorten_factor, d_model):
  return tl.Serial(
      tl.Fn(
          'Shorten',
          lambda x: jnp.reshape(  # Shorten -- move to depth.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] // shorten_factor, -1)),
          n_out=1),
      tl.Dense(d_model))


def RelformerLM(vocab_size,
                d_model=512,
                d_ff=2048,
                vanilla_layers=(1, 1),
                shorten_factor=3,
                n_rel_layers=6,
                rel_chunk_len=None,
                vanilla_chunk_len=None,
                last_full_layers=0,
                n_heads=8,
                dropout=0.1,
                dropout_shared_axes=None,
                vanilla_attn_type=tl.SelfAttention,
                relative_in_vanilla=False,
                pos_type=None,
                max_len=3072,
                n_raw_tokens_generated=1,
                mode='train',
                ff_activation=tl.FastGelu):

  token_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)]

  if vanilla_chunk_len is None and vanilla_attn_type == tl.SelfAttention:
    positional_encoder = PositionalEncoder(mode, dropout, max_len, pos_type)
  else:
    positional_encoder = []

  n_pre_decoder_blocks, n_post_decoder_blocks = vanilla_layers

  def create_reformer_blocks(  # pylint: disable=invalid-name
      n_layers,
      total_kv_pooling=1,
      layer_chunk_len=None,
      force_relative=False,
      dense=True):
    if n_layers == 0:
      return [tl.LayerNorm()]

    def determine_attn_type(layer_number):  # pylint: disable=invalid-name
      if layer_chunk_len is None and not force_relative:
        return vanilla_attn_type

      if layer_chunk_len is not None:
        chunk_offset = (layer_number % 2) * (layer_chunk_len // 2)
      else:
        chunk_offset = None

      if force_relative and n_layers - layer_number <= last_full_layers:
        chunk_len = None
        chunk_offset = None
      else:
        chunk_len = layer_chunk_len

      return functools.partial(
          RelativeAttentionWrapper,
          n_raw_tokens_generated=n_raw_tokens_generated,
          max_inference_length=max_len,
          total_kv_pooling=total_kv_pooling,
          chunk_len=chunk_len,
          chunk_offset=chunk_offset)

    d_per_head = d_model // n_heads

    decoder_blocks = []
    for i in range(n_layers):
      layer_attn_type = determine_attn_type(i)

      decoder_blocks.append(
          DecoderBlock(
              d_model,
              d_ff,
              d_per_head,
              d_per_head,
              n_heads,
              layer_attn_type,
              dropout,
              ff_activation,
              dropout,
              ff_use_sru=0,
              ff_chunk_size=0,
              ff_sparsity=0,
              attention_chunk_size=0,
              mode=mode))

    return [
        tl.Dup(),
        tl.ReversibleSerial(decoder_blocks),
        tl.Concatenate(),
        tl.LayerNorm(),
        tl.Dense(d_model) if dense else [],
    ]

  pre_decoder_blocks = create_reformer_blocks(
      n_pre_decoder_blocks, layer_chunk_len=vanilla_chunk_len,
      force_relative=relative_in_vanilla)

  relative_decoder_blocks = create_reformer_blocks(
      n_rel_layers,
      total_kv_pooling=shorten_factor,
      layer_chunk_len=rel_chunk_len,
      force_relative=True)

  conv_layer = tl.Serial(
      tl.CausalConv(d_model, shorten_factor),
      ff_activation()
  )

  post_decoder_blocks = create_reformer_blocks(
      n_post_decoder_blocks, layer_chunk_len=vanilla_chunk_len, dense=False,
      force_relative=relative_in_vanilla)

  if n_rel_layers > 0:
    funnel_part_of_model = [
        tl.Dup(),
        tl.ShiftRight(n_positions=shorten_factor - 1, mode=mode),
        _DownsamplerLM(shorten_factor, d_model),
        relative_decoder_blocks,
        tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
        _UpsamplerLM(shorten_factor, d_model),
        tl.LayerNorm(),
        tl.Concatenate(),
        conv_layer,
    ]
  else:
    funnel_part_of_model = []

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      token_encoder,
      positional_encoder,
      pre_decoder_blocks,
      funnel_part_of_model,
      post_decoder_blocks,
      tl.Dense(vocab_size),
  )
