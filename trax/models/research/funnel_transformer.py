"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
https://arxiv.org/abs/2006.03236
"""
from trax import layers as tl
from trax.models.transformer import _EncoderBlock, _FeedForwardBlock
import numpy as np


def _InternalMaxPool(arr):
    shape = arr.shape
    arr = arr.reshape((*shape[:-1], int(shape[-1] / 2), 2))
    arr = arr.max(axis=-1, keepdims=False)
    return arr


def _upsample(short, long):
    factor = -(-long.shape(1) // short.shape(1))  # ceil division
    return short.repeat(factor, axis=1)[, :long.shape(1), ]


def _Upsampler():
    return tl.Fn('Upsampler', _upsample)


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
    feed_forward = _FeedForwardBlock(
        d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)
    return tl.Serial(  # h, mask
        tl.LayerNorm(),  # h, mask
        tl.Dup(),  # h, h, mask
        tl.Dup(),  # h, h, h, mask
        pool_layer(pool_size=pool_size,
                   strides=strides,
                   padding=padding),  # q,k,v,masks=h',h,h,mask
        tl.Dup(),  # h', h', h, h, mask
        tl.Parallel(
            None,
            attention
        ),  # h', attention(...), mask
        tl.Add(),  # h'+attention(...), mask
        tl.Parallel(
            None,
            tl.Fn('max pool experiment', lambda x: _InternalMaxPool(x)),
        ),  # h'+attention(...), mask
        feed_forward
    )


def _FunnelEncoder(vocab_size,
                   encoder_segment_lenghts,
                   n_classes=10,
                   d_model=512,  # start
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
    segments = len(encoder_segment_lenghts)
    funnels = segments - 1
    assert (funnels >= 0)

    positional_encoder = [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
        tl.PositionalEncoding(max_len=max_len)]

    encoder_blocks = []
    n_encoder_segments = len(encoder_segment_lenghts)

    for i in range(n_encoder_segments):
        # Building i'th segment
        for _ in range(encoder_segment_lenghts[i]):
            # segment_size encoder blocks
            encoder_blocks.append(_EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                                                mode, ff_activation))

        # if not last segment, add funnel block
        if i != n_encoder_segments - 1:
            encoder_blocks.append(_FunnelBlock(d_model, pool_layer=pool_layer))

    # Assemble and return the model.
    return tl.Serial(  # toks
        # Encode.
        tl.Branch(
            positional_encoder, tl.PaddingMask()),  # vecs masks
        encoder_blocks,  # vecs masks
        tl.Select([0], n_in=2),  # vecs
        tl.LayerNorm(),  # vecs

        # Map to output categories.
        tl.Mean(axis=1),  # vecs
        tl.Dense(n_classes),  # vecs
        tl.LogSoftmax(),  # vecs
    )


def FunnelTransformer(vocab_size,
                      encoder_segment_lenghts,
                      n_decoder_segments,
                      d_model=512,  # start
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
    """Returns a Full Funnel Transformer.
    """
    segments = len(encoder_segment_lenghts)
    funnels = segments - 1
    assert (funnels >= 0)

    positional_encoder = [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
        tl.PositionalEncoding(max_len=max_len)]

    n_encoder_segments = len(encoder_segment_lenghts)

    encoder_blocks_before_first_pooling = [_EncoderBlock(d_model, d_ff, n_heads, dropout,
                                                         dropout_shared_axes, mode, ff_activation)
                                           for _ in range(encoder_segment_lenghts[0])]
    encoder_blocks_from_first_pooling = []

    for i in range(1, n_encoder_segments):
        # Building i'th segment

        # add funnel block between segments
        encoder_blocks_from_first_pooling.append(_FunnelBlock(d_model, pool_layer=pool_layer,
                                                              pool_size=pool_size, strides=strides))

        for _ in range(encoder_segment_lenghts[i]):
            # segment_size encoder blocks
            encoder_blocks_from_first_pooling.append(_EncoderBlock(d_model, d_ff, n_heads, dropout,
                                                                   dropout_shared_axes, mode, ff_activation))

    decoder_blocks = [_EncoderBlock(d_model, d_ff, n_heads, dropout,
                                    dropout_shared_axes, mode, ff_activation)
                      for _ in range(n_decoder_segments)]

    # Assemble and return the model.
    return tl.Serial(  # toks
        tl.Branch(
            positional_encoder, tl.PaddingMask()),  # vecs masks
        encoder_blocks_before_first_pooling,  # vecs masks
        tl.Select([0, 1, 0]),  # vecs masks residual = vecs
        encoder_blocks_from_first_pooling,  # vecs masks residual
        tl.Select([0, 2, 1], n_in=3),  # vecs residual masks
        tl.Parallel(  # residual from first segment is taken before normalization, so apply it now
            None, tl.LayerNorm(), None),  # vecs norm(residual) masks
        _Upsampler(),  # vecs masks
        decoder_blocks,
        tl.Select([0], n_in=2),  # vecs
        tl.LayerNorm(),
    )
