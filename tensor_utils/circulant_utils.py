import numpy as np

import torch
import torch.nn.functional as F

from scipy.linalg import circulant

import time

def flatten( x ):
    x = torch.hstack( torch.unbind( x, 0 ) )
    return x

def flat_circulant( x ):
    if x.ndim <= 2:
        return x
    
    x_ = t_circulant( x, -1 )
    x_ = flatten( flatten( x_ ) )
    
    return flat_circulant( x_ )

def t_circulant( x, dim = 0 ):
    dims = torch.arange( len( x.shape ) )
    dim = dims[ dim ]
    
    S = x.shape[dim]
    
    tensor = torch.stack( [
        torch.index_select( x, dim, torch.tensor( idxs, device = x.device ) ) \
                for idxs in circulant( range( S ) )
    ] )
    
    dim = dim + 1
    shape_dims = [ 
        torch.arange( 1, dim ), 
        torch.arange( dim + 1, len( x.shape ) + 1 )
    ]
    shape_dims = torch.hstack( shape_dims ).tolist()
    
    return tensor.permute( 0, dim, *shape_dims )