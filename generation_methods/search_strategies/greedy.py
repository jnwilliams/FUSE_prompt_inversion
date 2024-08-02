import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class GreedySearch( ):
    def __init__( self, model=None ):
        self._model = None
        if model is not None:
            self._model = model

    def reset( self ):
        pass

    def set_model( self, model ):
        self._model = model

    def search( self, candidates, split_size=512 ):   
        if self._model is None:
            raise EnvironmentError( "Language model is not set. Cannot compute error over candidates. Call set_model to set this attribute.")
        
        b, _, _, _ = candidates.shape

        losses = [ 
            self._model.loss( 
                    split.view( -1, *candidates.shape[-2:] ),
                    prepare_for_backward=False
                ).reshape( split.shape[0], -1 )
            for split in torch.split( candidates, split_size_or_sections=split_size, dim=1 )
        ]
        losses = torch.cat( losses, dim=1 )

        candidates_loss =  torch.softmax(  - losses, dim=1 )
        update_idx = candidates_loss.argmax( dim=-1 )

        batch_idx = torch.arange( b )
        choice = candidates[ batch_idx, update_idx ]
        loss = losses[ batch_idx, update_idx ]

        return choice, loss