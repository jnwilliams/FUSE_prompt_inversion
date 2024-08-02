import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

try:
    from prompt_inversion.generation_methods.utils.import_model import import_model
    from prompt_inversion.generation_methods.catalog import ABC
except ModuleNotFoundError as e:
    from generation_methods.utils.import_model import import_model
    from generation_methods.catalog import ABC
except Exception as e:
    raise e


class Caption( ABC.BaseClass ):
    """
        Use a captioner to find prompts for image generation
    """
 
    def __init__( self, search_strategy, config, model_id, **kwargs ):
        """
            Initialize Captioner

            Inputs:
                search_strategy : None
                    - the captioner does not have a search strategy but the
                    base class requires it
                    
                **kwargs : (See captioner in config_file/generation_methods)
        """
        super( Caption, self ).__init__(
            search_strategy,
            config,
            **kwargs
        )

        self.device = self._model.primary_model.device
        self.captioner, self.caption_processor, self.caption_tokenizer = \
            import_model( model_id )

        self.x = self._model.primary_model.x.data.clone()
        self.prompt_len = self.x.shape[1]
    
    def generate_candidates( self ):
        raise NotImplementedError()
    
    def step( self ):
        """
            Caption the image saved in primary/secondary models
        """
        prefix = self._model.primary_model.prefix
        image = None
        if self._model.primary_model.config['uses_images'] is True:
            image = self._model.primary_model.image
        else:
            if self._model.secondary_models is not None:
                for model in self._model.secondary_models:
                    if model.config['uses_images'] is True:
                        image = model.image

        if image is None:
            raise RuntimeError( "No image found in primary or secondary models" )

        inputs = self.caption_processor( image, text=prefix, return_tensors='pt' ).to( self.device )
        
        # Can probably let the user select these in the config, but may be too cumbersome
        out = self.captioner.generate(**inputs, 
            do_sample=False, 
            num_beams=5, 
            min_new_tokens = 11,
            max_new_tokens = 32,
            length_penalty = 1.0,
            renormalize_logits = True,
        )
        generated_caption = self.caption_processor.decode(out[0], skip_special_tokens=True)

        x = self._model.primary_model.from_text( generated_caption )
        x = self._model.primary_model.remove_special_tokens( x )
        loss = self._model.loss( x )

        x = self._model.primary_model.prepare_embedding( x )

        self.x.data = x
        
        return self.x.data.clone(), loss


    def search( self, steps = 50 ):
        """
            Inputs:
                steps : int
                    - Number of steps to take
        """

        tic = time.time()
        x, loss = self.step( )
        text = self._model.primary_model.to_text( x )
        toc = time.time() - tic

        print( f"Loss: {loss} - {text}")

        return x.data, toc



    



