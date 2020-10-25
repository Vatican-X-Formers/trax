"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
https://arxiv.org/abs/2006.03236
"""
from trax import layers as tl

# h' <- Pooling(h)
# na inpucie jest tupla (activations B x L x D, mask B x 1 x 1 x L)
# na outpucie jest (activations' , mask') tyle, że mniejsze
def _FunnelBlock(d_model=512, d_ff=2048, n_heads=8,
                 dropout=0.1, dropout_shared_axes=None, 
                 mode='train', ff_activation=tl.Relu,
                 pool_size=(2,),
                 strides=(2,),
                 padding='VALID'
                 ):
    attention = tl.AttentionQKV(
        d_features=d_model, n_heads=n_heads, dropout=dropout, mode=mode)
    
    # pierwszego zioma skopiowac 3 razy x,y,z = activ, activ, activations
    # 
    return tl.Serial(
        tl.Dup(), # => activ, activ, mask
        tl.Dup(), # => activ, activ, activ, mask
        tl.MaxPool(pool_size=pool_size, 
                strides=strides,
                padding=padding),# => Q := efekt max poolingu, K, V = stary ziom == activations, MASK    
        tl.Dup(), # h', h', h, h, m
        tl.Parallel(
            None,
            attention # stack semantic => tutaj wejdzie q,k,v, mask
                      # czyli po tym mamy activations, mask
        ), # po nim mamy h', atencja(...), mask
        tl.Add(), # h'+atencja(...), mask    
        tl.LayerNorm() # funnel_activations, mask
    )