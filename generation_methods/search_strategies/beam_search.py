import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import heapq

class BeamSearch():
    def __init__( self, model=None, beam_width=4  ):
        self._model = None
        self.beam_width = beam_width
        if model is not None:
            self._model = model

    def reset( self ):
        pass

    def set_model( self, model ):
        self._model = model

    def search( self, candidates, split_size=16 ):
        if self._model is None:
            raise EnvironmentError( "Language model is not set. Cannot compute error over candidates. Call set_model to set this attribute.")
              
        b, c, s, d = candidates.shape

        candidates = candidates.reshape( -1, s, d )
        candidates = candidates.reshape( b * c, -1 )
        candidates = torch.unique( candidates, dim=0 )

        b_uniq, _ = candidates.shape
        candidates = candidates.reshape( b_uniq, s, d )
        
        losses = [ 
            self._model.loss( 
                    split.view( -1, *candidates.shape[-2:] ),
                    prepare_for_backward=False
                ).reshape( split.shape[0], -1 )
            for split in torch.split( candidates, split_size_or_sections=split_size, dim=0 )
        ]
        losses = torch.cat( losses, dim=0 ).reshape( -1 )
        
        idxs = losses.argsort( descending = False )[:self.beam_width]
        choice, loss = candidates[ idxs ], losses[ idxs ]
        return choice, loss