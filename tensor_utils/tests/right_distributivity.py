import time
import torch

try:
    from tensor_utils import tensor_product, tensor_inverse
except ModuleNotFoundError as e:
    import tensor_product, tensor_inverse
except Exception as e:
    raise e

def r_distributivity( shape_a, shape_b, shape_c, eps = 1e-2 ):
    a = torch.randn( shape_a )
    b = torch.randn( shape_b )
    c = torch.randn( shape_c )

    x = tensor_product.t_prod( a + b, c )
    y = tensor_product.t_prod( a, c ) + tensor_product.t_prod( b, c )

    return torch.norm( x - y, p=2 ) < eps

def main( ):
    print( "\nTesting Right Distributivity\n")

    try:
        shape_a = ( 2, 2, 2 )
        shape_b = ( 2, 2, 2 )
        shape_c = ( 2, 2, 2 )

        assert r_distributivity( shape_a, shape_b, shape_c )
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: success")
    except AssertionError as e:
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape_a = ( 4, 2, 3 )
        shape_b = ( 4, 2, 3 )
        shape_c = ( 2, 6, 3 )

        assert r_distributivity( shape_a, shape_b, shape_c )
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: success")
    except AssertionError as e:
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape_a = ( 4, 512, 5 )
        shape_b = ( 4, 512, 5 )
        shape_c = ( 512, 1, 5 )

        assert r_distributivity( shape_a, shape_b, shape_c )

        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: success")
    except AssertionError as e:
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}     \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape_a = ( 4, 2, 3 )
        shape_b = ( 4, 2, 3 )
        shape_c = ( 2, 9, 6 )

        assert r_distributivity( shape_a, shape_b, shape_c )

        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}      \t: success")
    except AssertionError as e:
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}      \t: failed")
    except Exception as e:
        raise e 
    
    try:
        shape_a = ( 4, 8, 6 )
        shape_b = ( 4, 8, 6 )
        shape_c = ( 8, 2, 3 )

        assert r_distributivity( shape_a, shape_b, shape_c )

        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}      \t: success")
    except AssertionError as e:
        print( f"\tShape ( {shape_a} + {shape_b} ) x {shape_c}      \t: failed")
    except Exception as e:
        raise e 
    
if __name__ == "__main__":
    main()