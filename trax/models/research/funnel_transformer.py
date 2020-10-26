"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
https://arxiv.org/abs/2006.03236
"""
from trax import layers as tl
from typing import List
from trax.models.transformer import _EncoderBlock

def _FunnelBlock(d_model=512, d_ff=2048, n_heads=8,
                 dropout=0.1, dropout_shared_axes=None, 
                 mode='train', ff_activation=tl.Relu,
                 pool_size=(2,),
                 strides=(2,),
                 padding='VALID',
                 pool_layer=tl.AvgPool
                 ):
    """Internal funnel block. On input it takes (activations, masks).

    Args:
        d_model: Final dimension of tensors at most points in the model, including
            the initial embedding output.
        d_ff: Size of special dense layer in the feed-forward part of each block.
        n_heads: Number of attention heads.
        dropout: Stochastic rate (probability) for dropping an activation value
            when applying dropout within a block.
        dropout_shared_axes: Tensor axes on which to share a dropout mask.
            Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
            a useful way to save memory and apply consistent masks to activation
            vectors at different sequence positions.
        mode: If `'train'`, each block will include dropout; else, it will
            pass all values through unaltered.
        ff_activation: Type of activation function at the end of each block; must
            be an activation-type subclass of `Layer`.
        pool_size: Shape of window that gets reduced to a single vector value.
            If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
            must be a tuple of length :math:`n-2`.
        strides: Offsets from the location of one window to the locations of
            neighboring windows along each axis. If specified, must be a tuple of
            the same length as `pool_size`. If None, then offsets of 1 along each
            window axis, :math:`(1, ..., 1)`, will be used.
        padding: 'VALID' or 'SAME'. If 'VALID', no padding is done, and only
            full windows get reduced; partial windows are discarded. If 'SAME',
            padding is added at array edges as needed but is not counted in the
            computation of averages.

    Returns:
        A list of layers that maps (activations, mask) to (activations', mask).
    """
    attention = tl.AttentionQKV(
        d_feature=d_model, n_heads=n_heads, dropout=dropout, mode=mode)
    
    return tl.Serial(
        tl.Dup(), # h, h, mask
        tl.Dup(), # h, h, h, mask
        pool_layer(pool_size=pool_size, 
                strides=strides,
                padding=padding),# q,k,v,masks=h',h,h,mask
        tl.Dup(), # h', h', h, h, m
        tl.Parallel(
            None,
            attention
        ), # h', attention(...), mask
        tl.Add(), # h'+attention(...), mask    
        tl.LayerNorm() # funnel_activations, mask
        #TODO(mvxxx) fc
    )


def _FunnelDecoder():
    pass

def _FunnelEncoder(input_vocab_size,
                encoder_segment_lenghts,
                output_vocab_size=None,
                d_model=512, #start
                d_ff=2048,
                n_heads=8,
                max_len=2048,
                dropout=0.1,
                dropout_shared_axes=None,
                mode='train',
                ff_activation=tl.Relu,
                pool_layer=tl.AvgPool,
                pool_size=(2,),
                strides=(2,)):

  """Returns a Funnel Encoder.
  """
  def Embedder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
    ]
  segments = len(encoder_segment_lenghts)
  funnels = segments-1
  f,s = pool_size[0], strides[0]
  assert(funnels>=0)

  def funnel_size_generator(init, f, s, n):
      def generator():
        _val = init
        for _ in range(n):
            yield _val
            _val = int((_val - f)/s + 1)
      return generator

  dim_generator = funnel_size_generator(d_model, f, s, segments)
  funnel_dims = list(dim_generator())
  in_embedder = Embedder(input_vocab_size)

  # Positional encodings are not shared between encoder and decoder.
  # Since encoder doesn't run stepwise, we do not use predict mode there.
  encoder_mode = 'eval' if mode == 'predict' else mode
  in_encoder = in_embedder + [
      tl.PositionalEncoding(max_len=max_len, mode=encoder_mode)
  ]


  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  encoder_blocks = []
  n_encoder_segments = len(encoder_segment_lenghts)

  for i in range(n_encoder_segments):
      # Building i'th segment
      for _ in range(encoder_segment_lenghts[i]):
        # segment_size encoder blocks
        encoder_blocks.append(_EncoderBlock(funnel_dims[i], d_ff, n_heads, dropout, dropout_shared_axes,
                        mode, ff_activation))

      # if not last segment, add funnel block
      if i != n_encoder_segments-1:
          encoder_blocks.append(_FunnelBlock(funnel_dims[i], pool_layer=pool_layer))

  encoder = tl.Serial(
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )

def FunnelTransformer():
    pass