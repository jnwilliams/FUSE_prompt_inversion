import numpy as np

import torch
import torch.nn.functional as F

try:
    from .circulant_utils import t_circulant
except ImportError as e:
    from circulant_utils import t_circulant
except Exception as e:
    raise e

def t_prod_slow( x, y ):
    dims = x.shape
    if x.ndim <= 2:
        return x @ y
    
    x_unfold = torch.movedim( x, -1, 0 )
    y_unfold = torch.movedim( y, -1, 0 )
    
    x_circulant = t_circulant( x_unfold, dim = 0 )
    col = [[0]] * dims[-1]
    
    for i, batch in enumerate( x_circulant ):
        r = [[0]] * dims[-1]
        for j, ( x_, y_ ) in enumerate( zip( batch, y_unfold ) ):
            r[j] = t_prod( x_, y_ ).tolist()
            
        col[i] = torch.tensor( r ).sum( axis=0 )
    
    tensor = torch.stack( col )
    tensor = torch.movedim( tensor, 0, -1 )
    return tensor

def t_prod_fast( x, y ): 
    device = x.device
    dtype = x.dtype
    x, y = x.cpu().float(), y.cpu().float()
    outer = torch.einsum( 'imk, mjl -> ijkl', x, y )
    
    outer = torch.flip( outer, dims=(-1,) )
    outer = np.pad( outer, ( ( 0, 0 ), ( 0, 0, ), ( 0, 0 ), ( x.shape[-1], 0 ) ), mode='wrap' )
    outer = torch.tensor( outer )

    prod = torch.zeros( x.shape[0], y.shape[1], y.shape[-1] )
    for i in range( y.shape[-1] ):
        prod[ ..., i ] += torch.diagonal( outer, dim1=-2, dim2=-1, offset=i ).sum( axis=-1 )

    return torch.flip( prod, dims=(-1,) ).to( device ).to( dtype )

def t_prod( x, y, fast=True ):
    if fast:
        return t_prod_fast( x, y )
    else:
        return t_prod_slow( x, y )