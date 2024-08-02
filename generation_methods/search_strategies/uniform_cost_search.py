import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import heapq

class UniformCostSearch( ):
    def __init__( self, model=None, frontier_max_size = 128 ):
        self.frontier = [ [] ]
        self.frontier_max_size = frontier_max_size

        self._model = None
        if model is not None:
            self._model = model

    def set_model( self, model ):
        self._model = model

    def reset( self ):
        self.frontier = [ [] ]

    def search( self, candidates, split_size=256 ):   
        if self._model is None:
            raise EnvironmentError( "Language model is not set. Cannot compute error over candidates. Call set_model to set this attribute.")
        
        b, _, _, _ = candidates.shape
        if len( self.frontier ) != b:
            self.frontier = [ [] ] * b

        losses = [ 
            self._model.loss( 
                    split.view( -1, *candidates.shape[-2:] ),
                    prepare_for_backward=False
                ).reshape( split.shape[0], -1 )
            for split in torch.split( candidates, split_size_or_sections=split_size, dim=1 )
        ]

        losses = torch.cat( losses, dim=1 )
        
        for j in range( b ):
            for candidate, loss in zip( candidates[j], losses[j] ):
                cand = candidate.reshape( 1, -1, candidate.size(-1) )
                tie_breaker = torch.randn( 1 ).item()
                heapq.heappush( self.frontier[j], ( loss, tie_breaker, cand ) )

        self.frontier = heapq.nsmallest( 
            self.frontier_max_size + 1, 
            self.frontier, 
            key = lambda x: ( x[0], x[1] ) 
        )

        l, c = [], []
        for frontier in self.frontier:
            loss, _, choice = heapq.heappop( frontier )
            l.append( loss )
            c.append( choice )

        choice = torch.stack( c, dim=0 ).reshape( b, -1, self._model.primary_model.dims )
        loss = torch.stack( l, dim=0 )
        return choice, loss