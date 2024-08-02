import time
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
    
class AutoDAN( ABC.BaseClass ):
    """
        Apply the GCG algorithm (https://arxiv.org/pdf/2307.15043.pdf)
        to the inverse problem for image generation
    """

    def __init__( self, 
            search_strategy,
            config : dict, 
            **kwargs ):
        """
            Initialize AutoDAN

            Inputs:
                search_strategy: See config_file search_strategy for parameter info
                **kwargs : See config_file generation_methods for parameter info
        """
        super( AutoDAN, self ).__init__(
            search_strategy,
            config,
            **kwargs
        )
        search_strategy.set_model( self._model )
        if self._model.uses_diffusers:
            raise NotImplementedError( "\nAutoDAN is not configured to work with stablediffusion prompt optimization yet. Try PEZ.\n" )
        
        self.x = self._model.primary_model.x.data.clone()
        self.prompt_len = self.x.shape[1]
        self.batch_size = kwargs.get( "batch_size", 32 )
        self.logit_weight = kwargs.get( "logit_weight", 1. )
        torch.manual_seed( 0 )

    def generate_candidates( self, x ):
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

        # Convert the token embedding to a one-hot encoding based on the nearest neighbor
        b, s, d = x.shape
        _, idxs = self._model.primary_model.proj_to_vocab( x )
        x_cand = F.one_hot( idxs.long(), num_classes = self._model.primary_model.vocab.shape[0] )
        x_cand = x_cand.to( self._model.primary_model.dtype )
        
        # Add a dummy token to the end
        next_x = F.one_hot( 
            torch.zeros(1, device=x.device).long(), 
            num_classes=self._model.primary_model.vocab.shape[0] 
        )
        next_x = next_x.to( self._model.primary_model.dtype )
        next_x = next_x.requires_grad_( True ) # We only care about gradients for the new token
        
        x_cand = torch.concat( [ x_cand, next_x.tile( b, 1, 1 ) ], dim=1 )
        
        # Map back to the embedding space
        x_ = x_cand @ self._model.primary_model.vocab

        # compute the loss
        obj = self._model.loss( x_ ).sum()
        obj.backward()

        # compute the logits
        logits = self._model.primary_model.logits( 
            self._model.primary_model.prepare_embedding( x.data )
        )

        # candidate scores are the weighted sum of negative gradient and logits
        cand_choice = ( - next_x.grad + self.logit_weight * logits[:,-1] )

        # take the top k
        topk = torch.topk( cand_choice, k=self.batch_size, dim=-1, largest=True )

        return topk.indices, idxs
        
    def step( self ):
        """
            Randomly Sample 'Batch_Size' elements from candidate_search
            and compute the losses. replace the token in the input with the
            candidate token that minimizes the loss
        """
        b, s, d = self.x.shape
        x = self.x.data.clone()
        x = x.to( self._model.primary_model.dtype )

        # sample a set of {self.batch_size} candidates
        candidates, _ = self.generate_candidates( x ) 

        # Reformat candidates
        x_candidates = torch.concat( [
            x.tile( self.batch_size, 1, 1, 1 ).permute( 1, 0, 2, 3 ), 
            self._model.primary_model.vocab[ candidates ].reshape( b, -1, 1, d )
        ], dim=2 )

        # use predefined search strategy to choose a next iterate
        self.x.data, loss = self._search_strategy.search( x_candidates )
        return self.x.data.clone(), loss

    def search( self, steps = 50 ):
        """
            Inputs:
                steps : int
                    - Number of steps to take
        """

        tic = time.time()
        best = ( float('inf'), self.x.data.clone() )
        for i in range( steps ):
            x, loss = self.step( )
            text = self._model.primary_model.to_text( x )
            toc = round( float( time.time() - tic ), 4 )
            
            if loss.min() < best[0]:
                best = ( loss.min(), x[ loss.argmin( keepdim=True ) ], time.time() - tic )
        
            print( f"Time: {toc} - Iter: {i} - Loss: {loss} - {text}")
            torch.cuda.empty_cache()

        #x = self._model.primary_model.prepare_embedding( self.x.data.to( self._model.primary_model.dtype ) )
        #print( self._model.primary_model.generate( x ) )
        return best[1], best[2]