import numpy as np

import torch
import torch.nn.functional as F

try:
    from .circulant_utils import flat_circulant
except ImportError as e:
    from circulant_utils import flat_circulant
except Exception as e:
    raise e

def dft_mat( n ):
    return torch.fft.fft( torch.eye( n ) )

def idft_mat( n ):
    return torch.fft.ifft( torch.eye( n ) )

def block_dft( x, dims ):
    n1, n2, m = dims[0], dims[1], torch.tensor( dims[2:] ).prod()

    x = np.fft.fft2( x.reshape( -1, m, n2 ).numpy() )
    x = x.reshape( n1 * m, n2 * m )

    x = np.fft.ifft2( x.reshape( m, n1, -1 ), axes=(0,1) )
    x = x.reshape( n1 * m, n2 * m )

    return torch.tensor( x ).cfloat()

def block_idft( x, dims ):
    n1, n2, m = dims[0], dims[1], torch.tensor( dims[2:] ).prod()

    x = np.fft.ifft2( x.reshape( -1, m, n1 ).numpy() )
    x = x.reshape( n2 * m, n1 * m )

    x = np.fft.fft2( x.reshape( m, n2, -1 ), axes=(0,1) )
    x = x.reshape( n2 * m, n1 * m )

    return torch.tensor( x ).cfloat()

def pinv_clear_zero_dim( x, eps=1e-6 ):
    zero_dims = torch.norm( x, dim=[0,1], p=1 )
    z = x[:,:, zero_dims < eps ]
    x = x[:,:, zero_dims >= eps ]

    z_shape, x_shape = z.shape, x.shape
    squeeze = torch.where( torch.tensor( x_shape ) == 1 )[0]
    
    if any( squeeze ):
        x_inv = pinv( x.squeeze() )
        for dim in squeeze:
            x_inv.unsqueeze(dim)
    else:
        x_inv = pinv( x )

    x_inv = x_inv.reshape( *x_inv.shape[:2], *x.shape[2:] )
    z_inv = z.permute( 1, 0, *torch.arange( len( z.shape ) )[2:] )

    diff = [ x_size != y_size for x_size, y_size in zip( x_shape, z_shape ) ]
    diff =  torch.where( torch.tensor( diff ) )[0]
    if any( diff ):
        return torch.concat( [ x_inv, z_inv ], dim=diff[0] )
    else:
        return torch.concat( [ x_inv, z_inv ], dim=-1 )

def pinv( x, eps = 1e-6 ):
    dims = x.shape        
    if len( dims ) <= 2:
        return torch.linalg.pinv( x )
    
    if ( torch.norm( x, dim=[0,1], p=1 ) < eps ).any():
        return pinv_clear_zero_dim( x, eps )
    
    x = flat_circulant( x )

    G = block_dft( x, dims )
    
    G_stack = torch.split( G, dims[1], dim=1 )
    G_stack = torch.stack( G_stack, dim=0 )
    G_stack = torch.split( G_stack, dims[0], dim=1 )
    G_stack = torch.stack( G_stack, dim=0 )

    G_block_diag = torch.diagonal( G_stack, dim1=0, dim2=1 ).permute( 2, 0, 1 )

    G_block_diag = torch.stack( [ torch.linalg.pinv( block ) for block in G_block_diag ], dim=0 )
    
    G_inv = torch.block_diag( *G_block_diag )
    x_inv = block_idft( G_inv, dims )

    x_inv = torch.split( x_inv, dims[0], dim=1 )[0]
    x_inv = torch.stack( torch.split( x_inv, dims[1], dim=0 ), 0 )
    
    for dim_ in dims[2:-1]:
        x_inv = torch.stack( torch.split( x_inv, dim_, dim=0 ), -1 )
        
    x_inv = torch.movedim( x_inv, 0, -1 )
    return x_inv.real


