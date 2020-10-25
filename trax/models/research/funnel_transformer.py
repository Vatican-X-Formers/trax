"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
https://arxiv.org/abs/2006.03236
"""
from trax import layers as tl


def _FunnelBlock(d_model=512, d_ff=2048, n_heads=8,
                 dropout=0.1, dropout_shared_axes=None, 
                 mode='train', ff_activation=tl.Relu,
                 pool_size=(2,),
                 strides=(2,),
                 padding='VALID'
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
        tl.Branch(
            tl.Dup(), tl.Dup()
        ) # => h, h, mask
        tl.AvgPool(pool_size=pool_size, 
                strides=strides,
                padding=padding),# q,k,v,masks=h',h,h,mask
        tl.Dup(), # h', h', h, h, m
        tl.Parallel(
            None,
            attention
        ), # h', attention(...), mask
        tl.Add(), # h'+attention(...), mask    
        tl.LayerNorm() # funnel_activations, mask
    )