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
    

class GCG( ABC.BaseClass ):
    """
        Apply the GCG algorithm (https://arxiv.org/pdf/2307.15043.pdf)
        to find prompts for image generation
    """
 
    def __init__( self, search_strategy, config, **kwargs ):
        """
            Initialize AutoDAN

            Inputs:
                search_strategy: See config_file search_strategy for parameter info
                **kwargs : See config_file generation_methods for parameter info
        """
        super( GCG, self ).__init__(
            search_strategy,
            config,
            **kwargs
        )
        if self._model.uses_diffusers:
            raise NotImplementedError( "\nGCG is not configured to work with stablediffusion prompt optimization yet. Try PEZ.\n" )
        

        search_strategy.set_model( self._model )
        self.x = self._model.primary_model.x.data.clone()

        self.prompt_len = self.x.shape[1]
        self.batch_size = kwargs.get( "batch_size", 32 )
        self.topk = kwargs.get( "top_k", 32)

        record_interval = kwargs.get( "record_interval", 10 )
        store_intermediate = kwargs.get( "store_intermediate", True )
        self.set_store_intermediate( store = store_intermediate, recording_interval = record_interval )


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

        # Map input embed to one-hot encoding to inspect gradient per token
        _, idxs = self._model.primary_model.proj_to_vocab( x )

        one_hot = F.one_hot( idxs, self._model.primary_model.tokenizer.vocab_size )
        one_hot = one_hot.to( self._model.primary_model.dtype )
        one_hot.requires_grad_()

        # map back to embedding space differentiably
        x_ = one_hot @ self._model.primary_model.vocab

        # get gradient
        loss = self._model.loss( x_ ).sum()
        loss.backward()

        # take topk of negative gradient
        topk = torch.topk( - one_hot.grad, k=self.topk, dim=-1, largest=True )
        return topk.indices, topk.values
        
    def get_candidate_subset( self, candidates, candidate_values ):
        """
            Take the top_k candidates for every token. Uniformly sample {self.batch_size}
            to compute the true loss

            Inputs:
                candidates: ( Batch x Seq Len x K ) top k candidates
                candidate_values: corresponding gradients for each candidate

            Outputs:
                x_candidates: ( {self.batch_size} x Seq Len x Embedding Dimension ) candidates
                    reshaped from one-hot to sequence embeddings
        """
        b, s, d = candidates.shape
        token_idx = torch.arange( self.prompt_len, device=self.x.device ).reshape( 1, -1, 1 )
        token_idx = token_idx.tile( b, 1, self.topk )

        candidates = torch.stack( 
            [ 
                token_idx, 
                candidates,
                candidate_values.to( torch.float32 )
            ],
            dim=-1 )

        candidates = candidates.reshape( b, -1, 3 )
        seq_idxs = torch.arange( candidates.shape[1] ).to( torch.float32 ).tile( b, 1 )
        keep_idxs = torch.multinomial( 
            seq_idxs, 
            self.batch_size
        )
        batch_idxs = torch.arange( b ).tile( self.batch_size, 1 ).T

        candidates = candidates[ batch_idxs, keep_idxs ]
        idx, token, _ = candidates.permute( 2, 0, 1 )
        idx, token = idx.long(), token.long()

        x_candidates = self.x.data.unsqueeze(1)
        x_candidates = x_candidates.to( self._model.primary_model.dtype )
        x_candidates = x_candidates.tile( 1, self.batch_size, 1, 1 ) 

        seq_idxs = torch.arange( self.batch_size ).tile( b, 1 )
        x_candidates[ batch_idxs, seq_idxs, idx ] = self._model.primary_model.vocab[ token ]

        return x_candidates
    
    def step( self ):
        """
            Randomly Sample 'Batch_Size' elements from candidate_search
            and compute the losses. replace the token in the input with the
            candidate token that minimizes the loss
        """
        x = self.x.data.clone()

        candidates, candidate_values = self.generate_candidates( x )
        x_candidates = self.get_candidate_subset( candidates, candidate_values )

        # apply search strategy to choose next iterate
        with torch.no_grad():
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
            tic = time.time()
            x, _ = self.step( )
            x_proj , _= self._model.primary_model.proj_to_vocab( 
                x.data.to( self._model.primary_model.dtype )
            )
            loss = self._model.loss( x_proj )
            
            if self.store_intermediate:
                try:
                    record_list = iter( self.intermediate_record_interval )
                    if i in record_list:
                        self.intermediate_results.append( { "step": i, "prompt": self._model.primary_model.to_text( x )[0], "loss": loss } )
                    
                except TypeError:
                    if ( i % self.intermediate_record_interval ) == 0:
                        self.intermediate_results.append( { "step": i, "prompt": self._model.primary_model.to_text( x )[0], "loss": loss } )
                    
            if loss.min() < best[0]:
                min_idx = loss.argmin()
                print( f"\nNew Best \t Loss: {loss[min_idx]} - {self._model.primary_model.to_text( x_proj[min_idx].unsqueeze(0) )}\n" )
                best = ( loss[min_idx], x_proj[ min_idx ].unsqueeze(0) , time.time() - tic, i )

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



