import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

try:
    from prompt_inversion.generation_methods.catalog import ABC
except ModuleNotFoundError as e:
    from generation_methods.catalog import ABC
except Exception as e:
    raise e
    
class Random( ABC.BaseClass ):
    """
        Apply the GCG algorithm (https://arxiv.org/pdf/2307.15043.pdf)
        to the inverse problem for image generation
    """
 
    def __init__( self, search_strategy, config,  batch_size = 20, search_ball = 'l2', **kwargs ):
        """
            Initialize PEZ

            Inputs:
                model_name : string
                    - the name of the model that we want to use. 
                    Currently implemented models are gpt2, clip, bert, and bart
                    As gpt2, bert, and bart, do not have image embeddings,
                    this should always be set to clip.
                
                image_path : string
                    - the location of the image that you would like
                    to use when solving the inverse problem
                    
                **kwargs : (See models/base_model.py)
        """
        super( Random, self ).__init__(
            search_strategy,
            config,
            **kwargs
        )
        search_strategy.set_model( self._model )
        if self._model.uses_diffusers:
            raise NotImplementedError( "\nRandom is not configured to work with stablediffusion prompt optimization yet. Try PEZ.\n" )
        

        self.x = self._model.primary_model.x.data.clone()

        self.prompt_len = self.x.shape[1]
        self.k = 256
        self.batch_size = batch_size
        self.candidate_size = self.batch_size * self.prompt_len
        if search_ball.lower() in [ 'l2', 'l0' ]:
            self.search_ball = search_ball.lower()
        else:
            raise ValueError( "Search ball for random search is {search_ball}. Must be L2 or L0" )

        record_interval = kwargs.get( "record_interval", 10 )
        store_intermediate = kwargs.get( "store_intermediate", True )
        self.set_store_intermediate( store = store_intermediate, recording_interval = record_interval )


    def l2_candidate_ball( self, x, percentile=0.05 ):
        b, s, d = x.shape
        n_elements = max( self.k, math.floor( self._model.primary_model.tokenizer.vocab_size * percentile ) )

        distances = self._model.primary_model.vocab.view( -1, 1, d ) - x.view( -1, d )
        distances = distances.reshape( b, -1, s, d )
        distance_norm = distances.norm( dim=-1 ).permute( 0, 2, 1 )

        idxs = torch.topk( distance_norm, n_elements, largest=False ).indices
        return idxs
    
    def l0_candidate_ball( self, x ):
        b, s, d = x.shape
        p = torch.ones( self._model.primary_model.vocab.shape[0], device=x.device ).repeat( s, 1 )
        
        idxs = torch.stack( [ p.multinomial( self.k ) for _ in range( b ) ], dim=0 )
        return idxs

    def generate_candidates( self, x ):
        # ToDO: ensure that idxs has enough elements of for each batch/sequence so
        # that we will not hit an error if idxs < self.batch_size
        """
            Generate a set of candidates by computing the gradients of
            a one-hot encoding and returning the top k for each token in
            the sequence.

            Inputs:
                x : torch.Tensor
                    - Input tensor of size ( Batch x Seq Len x Dim )

            Outputs:
                topk : torch.Tensor.Int32
                    - ( Batch x Seq Len x K ) indices that are the maximum 
                    elements of the negative of the gradient

                idxs ; torch.Tensor.Int32
                    - The indices for the input tensor in the model/tokenizer's 
                    vocab
        """
        b, s, d = x.shape

        if self.search_ball == "l2":
            idxs = self.l2_candidate_ball( x )
        elif self.search_ball == "l0":
            idxs = self.l0_candidate_ball( x )

        seq_p = torch.ones( b, s, device=x.device ) / ( s )
        seq_choice = seq_p.multinomial( self.batch_size, replacement = True )

        x_candidates = x.tile( self.batch_size, 1, 1, 1  )
        x_candidates = x_candidates.permute( 1, 0, 2, 3 )
        x_candidates = x_candidates.to( self._model.primary_model.dtype )

        batch_indices = torch.arange(b, device=x.device).repeat( s * idxs.shape[-1], 1 ).t()
        row_indices = torch.arange(s, device=x.device).repeat( b, idxs.shape[-1] )
        values = idxs.reshape( b, -1 )

        to_sample =  torch.concat(
            (batch_indices, row_indices, values), 
            dim=0
        )
 
        token_p = torch.ones( to_sample.shape[-1] ) / to_sample.shape[-1]
        token_choice = token_p.multinomial( self.batch_size, replacement = True )
        token_choice = to_sample[ ..., token_choice ]

        x_candidates[ token_choice[0], torch.arange( self.batch_size ), token_choice[1] ] = \
            self._model.primary_model.vocab[ token_choice[2] ]

        x_candidates = torch.concat( [ 
            x_candidates, 
            self.x.data.unsqueeze(1).to( self._model.primary_model.dtype )
            ], dim=1 
        )
        return x_candidates.contiguous()
    
    def step( self ):
        """
            Randomly Sample 'Batch_Size' elements from candidate_search
            and compute the losses. replace the token in the input with the
            candidate token that minimizes the loss
        """
        b, s, d = self.x.shape
        x = self.x.data.clone()
        x_candidates = self.generate_candidates( x )

        self.x.data, loss = self._search_strategy.search( x_candidates )
        return self.x.data.clone(), loss

    def search( self, steps = 50 ):
        """
            Inputs:
                steps : int
                    - Number of steps to take
        """

        self.intermediate_results = []
        best = ( float('inf'), self.x.data.clone(), 0. )
        tic = time.time()
        
        for i in range( steps ):
            x, _ = self.step( )
            
            x_proj , _= self._model.primary_model.proj_to_vocab( 
                x.data.to( self._model.primary_model.dtype )
            )
            loss = self._model.loss( x_proj )
            
            if self.store_intermediate:
                try:
                    record_list = iter( self.intermediate_record_interval )
                    if i in record_list:
                        self.intermediate_results.append( { "step": i, "prompt": self._model.primary_model.to_text( x )[0], "loss": loss.item() } )
                    
                except TypeError:
                    if ( i % self.intermediate_record_interval ) == 0:
                        self.intermediate_results.append( { "step": i, "prompt": self._model.primary_model.to_text( x )[0], "loss": loss.item() } )
                    
            if loss < best[0]:
                print( f"\nNew Best \t Loss: {loss} - {self._model.primary_model.to_text( x_proj )}\n" )
                best = ( loss.min(), x[ loss.argmin( keepdim=True ) ] , time.time() - tic, i )

            print( f"Iter: {i} - Loss: {loss} - {self._model.primary_model.to_text( x_proj )}")

        print( f"\nBest Found: \t Loss: {best[0]} - {self._model.primary_model.to_text( best[1] )}" )

        best_loss, best_prompt_embed, time_to_best, steps_to_best = best
        if self.store_intermediate:
            self.intermediate_results.append( 
                { 
                    "step": steps_to_best, 
                    "prompt": self._model.primary_model.to_text( best_prompt_embed )[0], 
                    "loss": best_loss.item() 
                } 
            )
                    
        return best_prompt_embed.data, time_to_best

