import re
import os
import time
import random
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from prompt_inversion.tensor_utils import tensor_product, tensor_inverse
except ModuleNotFoundError as e:
    from tensor_utils import tensor_product, tensor_inverse
except Exception as e:
    raise e

from datasets import load_dataset

class TranslatorUtils( ):
    def __init__( self, from_model, to_model, max_subtokens = 4, max_examples_per_case = 1024 ):
        self.from_model = from_model
        self.to_model = to_model
        self.max_subtokens = max_subtokens
        self.max_examples_per_case = max_examples_per_case

    def set_grid( self, fname=None ):
        if self.from_model.config['_name_or_path'] == self.to_model.config['_name_or_path']:
            self.grid = None
            return

        if fname is not None:
            (self.grid,) = torch.load( fname, map_location=torch.device('cpu') )
            return
        
        self.set_dataset()
        self.grid = torch.zeros( self.max_subtokens, self.max_subtokens ).tolist()
        self.inv_grid = torch.zeros( self.max_subtokens, self.max_subtokens ).tolist()
        print( "\nInitializing Gradients From {} to {}...\n".format(
            self.from_model.config['_name_or_path'], self.to_model.config['_name_or_path']
        ) )
        for i in range( self.max_subtokens ):
            gradient, n_elements = self.set_gradient( max_subtokens=i+1, max_n=self.max_examples_per_case )
            self.grid[ i ] = gradient
            print( f"\t[{n_elements} examples w/ {i+1} tokens]\t", end = '', flush=True )

        print( "\n" )

        del self.idx_A, self.idx_B, self.text_list

    def tokenize( self, model, text, **kwargs ):
        return model.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt", **kwargs ).input_ids
    
    def set_dataset( self, ):     
        print( "\nLoading Initial Corpus to Precompute the Gradients...\n")         
        data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", download_mode="reuse_cache_if_exists" )
        pattern = re.compile( f"\s+" )

        text_list = pattern.split( ( " ".join( data['text'] ) ).lower() )
        text_list = list( sorted( set( text_list ) ) )

        random.shuffle( text_list )
        self.text_list = text_list

        self.idx_A = [ self.tokenize( self.from_model, text )[0] for text in text_list ] 
        self.idx_B = [ self.tokenize( self.to_model, text )[0] for text in text_list ] 

    def set_gradient( self, max_subtokens = 1, max_n = int( 2**14 ) ):
        if ( self.idx_A is None ) or ( self.idx_B is None ):
            self.set_dataset()

        choose_A = list( range( len( self.idx_A ) ) )
        choose_B = [ i for i, id_ in enumerate( self.idx_B ) if len( id_ ) == ( max_subtokens ) ]

        choose = set( choose_A ).intersection( set( choose_B ) )
        choose = list( choose )[ :max_n ]

        if len( choose ) == 0:
            gradient = torch.eye( self.to_model.dims )
            gradient = F.pad( gradient, ( 0, 0, 0, self.from_model.dims - self.to_model.dims ) )
            gradient = torch.stack( 
                [ gradient ] + [ torch.zeros_like( gradient ) for _ in range( max_subtokens - 1 ) ], 
                dim=-1 
            )
            return gradient, 0 
        
        options = torch.zeros( len( self.text_list ) )
        options[ choose ] += 1
        use_text = list( itertools.compress( self.text_list, options ) )

        sub_idx_A = self.tokenize( 
            self.from_model, 
            use_text,
            padding="max_length",
            truncation=True,
            max_length = max_subtokens ).cpu()    

        sub_idx_B = self.tokenize( self.to_model, use_text ).cpu()    

        x_A = self.from_model.vocab[ sub_idx_A ]
        rows_to_replace = torch.all(
            x_A == self.from_model.vocab[ self.from_model.tokenizer.eos_token_id ], 
            dim=-1
        )
        x_A[ rows_to_replace ] = 0
        x_A = x_A.cpu()

        x_B = self.to_model.vocab[ sub_idx_B ].cpu()

        x_A = x_A.permute( 0, 2, 1 )
        x_B = x_B.permute( 0, 2, 1 )

        inv = tensor_inverse.pinv( x_A )
        gradient = tensor_product.t_prod( inv, x_B )

        torch.cuda.empty_cache()
        return gradient, len( choose )

class InverseTranslate( nn.Module ):
    def __init__( self, from_model, to_model, max_subtokens=5, fname = None ):
        super( InverseTranslate, self ).__init__(  )
        self.device = from_model.device

        self.translate_utils = TranslatorUtils( to_model, from_model, max_subtokens=max_subtokens )
        if os.path.exists( fname ):
            self.translate_utils.set_grid( fname )
        else:
            self.translate_utils.set_grid()
            torch.save( (self.translate_utils.grid,), fname )
        
        for i in range( self.translate_utils.max_subtokens ):
            for j in range( self.translate_utils.max_subtokens ):
                self.translate_utils.grid[ i ][ j ] = tensor_inverse.pinv( self.translate_utils.grid[ i ][ j ] )

        self.fn = Translate_Fn.apply

    def forward( self, x ):
        return self.fn( x, self.translate_utils )
    
class Translate( nn.Module ):
    def __init__( self, from_model, to_model, fname = None, **kwargs ):
        super( Translate, self ).__init__(  )
        self.device = from_model.device
        self.max_subtokens = kwargs.get( "max_subtokens", 4 )
        self.max_examples_per_case = kwargs.get( "max_examples_per_case", 1024 )
        
        self.translate_utils = TranslatorUtils( from_model, to_model, 
            max_subtokens=self.max_subtokens,
            max_examples_per_case=self.max_examples_per_case  
        )

        if os.path.exists( fname ):
            self.translate_utils.set_grid( fname )
        else:
            parent_dir = os.path.dirname(fname)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            self.translate_utils.set_grid()
            torch.save( (self.translate_utils.grid,), fname )
            print( f"\nSaving gradients to: {fname}\n" )

        self.fn = Translate_Fn.apply

    def forward( self, x, prepare_for_backward=True ):
        return self.fn( x, self.translate_utils, prepare_for_backward )
    
class Translate_Fn( torch.autograd.Function ):
    @staticmethod
    def forward( ctx, x_A, translate_utils, prepare_for_backward ):
        text = translate_utils.from_model.to_text( x_A )
        
        is_identity = False
        if translate_utils.from_model.config['_name_or_path'] == translate_utils.to_model.config['_name_or_path']:
            is_identity = True
            x_B = x_A
        else:
            x_B = translate_utils.to_model.remove_special_tokens( translate_utils.to_model.from_text( text ) )

        if prepare_for_backward:
            word_tensor = translate_utils.from_model.split_embedding_by_whitespace( text, x_A )
            word_shape_A = ( word_tensor.norm( p=1, dim=-1 ) > 0 ).sum( dim=-1 )

            word_tensor = translate_utils.to_model.split_embedding_by_whitespace( x = x_B )
            word_shape_B = ( word_tensor.norm( p=1, dim=-1 ) > 0 ).sum( dim=-1 )

            ctx.save_for_backward( x_A, word_shape_A, word_shape_B )
            ctx.constant = ( 
                text,
                translate_utils.max_subtokens,
                translate_utils.grid,
                translate_utils.from_model.dims, 
                translate_utils.to_model.dims,
                is_identity
            )
        return x_B
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward( ctx, grad_output ):
        ( x_A, word_shape_A, word_shape_B ) = ctx.saved_tensors
        text, max_subtokens, gradient_grid, model_A_dims, model_B_dims, is_identity = ctx.constant

        if is_identity:
            return grad_output, None, None, None

        seq_len = x_A.shape[1]
        gradient = []
        for g, shape_A_full, shape_B_full in zip( grad_output, word_shape_A, word_shape_B ):
            grad = []
            g_ = g.reshape( 1, -1, model_B_dims )
            for shape_A in shape_A_full:
                word_grad, grad_output = torch.tensor_split( g_, (shape_A,), dim=1 )

                if shape_A > max_subtokens:
                    chain_gradient = torch.randn( model_A_dims, model_B_dims, shape_A )
                else:
                    chain_gradient = gradient_grid[ shape_A - 1 ]

                chain_gradient = chain_gradient.to( "cuda" )
                grad.append( 
                    tensor_product.t_prod( word_grad.permute( 0, 2, 1 ), chain_gradient.permute( 1, 0, 2 ) ) 
                )

            grad = torch.concat( grad, dim=-1 ).permute( 0, 2, 1 ) 
            gradient.append( grad[ :, :seq_len ] )
            
        gradient = torch.concat( gradient, dim=0 )
        return gradient, None, None, None