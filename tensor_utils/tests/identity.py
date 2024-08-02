import time
import torch

try:
    from tensor_utils import tensor_product, tensor_inverse
except ModuleNotFoundError as e:
    import tensor_product, tensor_inverse
except Exception as e:
    raise e

def identity( x_shape, eps=1e-2 ):
    x = torch.randn( *x_shape )

    x_inv = tensor_inverse.pinv( x )
    z = tensor_product.t_prod( x_inv, x )
    x_hat = tensor_product.t_prod( x, z )
    return torch.norm( x_hat - x, p=2 ) < eps


def main( ):
    print( "\nTesting: x @ ( x @ inv( x ) ) == x \n" )
     
    try:
        shape = ( 4, 4, 1 )
        assert identity( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 

    try:
        shape = ( 2, 2, 2 )
        assert identity( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape = ( 3, 3, 3 )
        assert identity( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape = ( 6, 3, 4 )
        assert identity( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 
    
if __name__ == "__main__":
    main()