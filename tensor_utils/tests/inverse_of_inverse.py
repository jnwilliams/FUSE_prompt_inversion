import torch

try:
    from tensor_utils import tensor_product, tensor_inverse
except ModuleNotFoundError as e:
    import tensor_product, tensor_inverse
except Exception as e:
    raise e

def inverse_of_inverse( x_shape, eps=1e-2 ):
    x = torch.randn( *x_shape )

    x_inv = tensor_inverse.pinv( x )
    return torch.norm( tensor_inverse.pinv( x_inv ) - x, p=2 ) < eps

def main( ):    
    print( "\nTesting: inv( inv( x ) ) == x \n" )
    
    try:
        shape = ( 2, 2, 2 )
        assert inverse_of_inverse( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 

    try:
        shape = ( 4, 3, 2 )
        assert inverse_of_inverse( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 

    try:
        shape = ( 4, 5, 3 )
        assert inverse_of_inverse( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape = ( 1, 512, 6 )
        assert inverse_of_inverse( shape )
        print( f"\tShape {shape}  \t: success")
    except AssertionError as e:
        print( f"\tShape {shape}  \t: failed")
    except Exception as e:
        raise e 
    
    if __name__ == "__main__":
        main()