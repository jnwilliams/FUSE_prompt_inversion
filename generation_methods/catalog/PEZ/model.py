import time
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

class PEZ( ABC.BaseClass ):
    """
        Apply the PEZ algorithm (https://arxiv.org/pdf/2302.03668.pdf)
        to find prompts for image generation
    """
    def __init__( self, search_strategy, config, **kwargs ):
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
        super( PEZ, self ).__init__(
            search_strategy,
            config,
            **kwargs
        )

        self.x = self._model.primary_model.x.data.clone().requires_grad_()
        self.lr = kwargs.get( "lr", 1e-1 )
        self.weight_decay = kwargs.get( "weight_decay", 1e-1 )
        self.opt = optim.AdamW( (self.x,), lr = self.lr, weight_decay=self.weight_decay ) 
        
        record_interval = kwargs.get( "record_interval", 10 )
        store_intermediate = kwargs.get( "store_intermediate", True )
        self.set_store_intermediate( store = store_intermediate, recording_interval = record_interval )

    def generate_candidates( self ):
        raise NotImplementedError()
    
    def step( self, idx=None ):
        """
            Take one gradient step

            PEZ computes gradients at the nearest tokens in the vocabulary
            and then applies those gradients to an embedding in a continuous
            space
        """
        self.opt.zero_grad()

        x_proj, _ = self._model.primary_model.proj_to_vocab( 
            self.x.data.to( self._model.primary_model.dtype )
        )

        x_proj.requires_grad_()
        loss = self._model.loss( x_proj )
        loss.backward()

        self.x.grad = x_proj.grad.to( self.x.dtype )
        
        self.opt.step()
        return self.x.data.clone(), loss

    def search( self, steps = 50 ):
        """
            Inputs:
                steps : int
                    - Number of steps to take
        """

        self.intermediate_results = []
        self.opt = optim.AdamW( (self.x,), lr = self.lr, weight_decay=self.weight_decay ) 
        best = ( float('inf'), self.x.data.clone(), 0. )
        
        tic = time.time()
        
        for i in range( steps ):
            x, _ = self.step( idx = None )
            
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
                best = ( loss, x_proj, time.time() - tic, i  )

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



    



