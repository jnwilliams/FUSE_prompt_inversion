import os, sys

from abc import ABC, abstractmethod
from typing import Union, Optional

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

try:
    from prompt_inversion.generation_methods.utils.import_model import import_model
    from prompt_inversion.generation_methods.utils.tokenizer_translate import Translate
except ModuleNotFoundError as e:
    from generation_methods.utils.import_model import import_model
    from generation_methods.utils.tokenizer_translate import Translate
except Exception as e:
    raise e
    


class Model( ):
    """
        This class handles computing losses for inputs across
        each model and provides access to the preprocessing/helper
        functions in {toplevel_dir}/models/catalog
    """
    def __init__( self, config, **kwargs ):
        self.uses_diffusers = False
        self.adapter_params = config.get( "adapter_params", None )

        self.primary_model_info = config['primary_model']
        self.primary_model = import_model( 
            **self.primary_model_info,
            **kwargs
        )
        if self.primary_model.config['architectures'][0] == 'StableDiffusionPipeline':
            self.uses_diffusers = True

        if 'secondary_models' not in config.keys():
            self.secondary_models_info = None
            self.secondary_models = None
        else:
            self.secondary_models_info = config['secondary_models']
            self.secondary_models = tuple( 
            import_model(
                **model_info,
                **kwargs
            ) for model_info in self.secondary_models_info )

            for model in self.secondary_models:
                if model.config['architectures'][0] == 'StableDiffusionPipeline':
                    self.uses_diffusers = True
                    
            self.primary_to_secondary_translate = tuple( 
                Translate( 
                    self.primary_model, 
                    secondary_model, 
                    max_subtokens=self.adapter_params['max_subtokens'], 
                    max_examples_per_case=self.adapter_params['max_examples_per_case'],
                    fname = os.path.join( 
                            self.adapter_params['save_gradients_to'],
                            "{}_to_{}_grid.pt".format(
                                self.primary_model.config['_name_or_path'].replace( "/", "_" ),
                                secondary_model.config['_name_or_path'].replace( "/", "_" )
                            ) )
                ) for secondary_model in self.secondary_models
            )

    def loss( self, x, prepare_for_backward=True, **kwargs ):
        b, s, d = x.shape
        
        x_ =  self.primary_model.prepare_embedding( x ) 
        model_loss = self.primary_model.loss( x_, **kwargs )
        model_loss *= self.primary_model_info['parameters']["weight"]

        if self.secondary_models_info is not None:
            secondary_loss = torch.zeros( 
                len( self.primary_to_secondary_translate ), x.shape[0],
                device=self.primary_model.device 
            )
            iterable = enumerate( 
                zip( 
                    self.primary_to_secondary_translate, 
                    self.secondary_models, 
                    self.secondary_models_info 
                ) 
            )
            for i, ( translate, secondary_model, secondary_info ) in iterable:
                x_ = translate( x, prepare_for_backward )
                x_ = secondary_model.prepare_embedding( x_ )
                secondary_loss[i] += secondary_model.loss( x_ )
                secondary_loss[i] *= secondary_info['parameters']["weight"]

            model_loss += secondary_loss.sum( dim=0 )
        return model_loss


    def set_image( self, image_path ):
        if self.primary_model.config['uses_images'] is True:
            self.primary_model.set_image( image_path )

        if self.secondary_models_info is not None:
            for model in self.secondary_models:
                if model.config['uses_images'] is True:
                    model.set_image( image_path )

class BaseClass( ABC ):
    """
        Abstract Base class for generation methods.
        This class also creates and the above 'Model'
        class to handle working with multiple models
    """
    def __init__( self, 
            search_strategy,
            config: dict,
            **kwargs
    ) -> None:  
        self._model = Model( 
            config,
            **kwargs
        )

        self.intermediate_results = []
        self._search_strategy = search_strategy
        self._model.primary_model.set_embedding()

    def set_store_intermediate( self, store = True, recording_interval=1 ):
        self.intermediate_results = []
        
        self.store_intermediate = store
        self.intermediate_record_interval = recording_interval

    @abstractmethod
    def generate_candidates( self, x : torch.Tensor ):
        pass
    
    @abstractmethod
    def step( self ):
        pass

    @abstractmethod
    def search( self, steps : int ):
        pass