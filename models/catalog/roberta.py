import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from prompt_inversion.models.language_utils import LanguageUtils
except ModuleNotFoundError as e:
    from models.language_utils import LanguageUtils
except Exception as e:
    raise e

class Roberta( LanguageUtils ):
    def __init__( self, model, tokenizer, vocab, config, label="positive", **kwargs  ):

        super( Roberta, self ).__init__( vocab, tokenizer, **kwargs )
        self.model = model
        self.config = config
        self.parameters = kwargs.get( "parameters", None )
        
        self.set_prefix( self.parameters.get( "prefix", "" ) )
        self.set_suffix( self.parameters.get( "suffix", "" ) )

        print( self.x_prefix.shape )
        print (self.x_suffix.shape )
        try:
            self.label = self.config[ 'label2id' ][ label ]
        except KeyError:
            raise KeyError( f"Label is {label}. Must be one of "
                    f"{', '.join( list( self.config['label2id'].keys() ) )}.")

    def logits( self, x = None, **kwargs ):
        if x is None:
            x = self.x.data.clone()

        return self.model( inputs_embeds = x ).logits

    def loss( self, x = None, **kwargs ):
        loss_fn = nn.CrossEntropyLoss(  reduction = 'none' )
        if x is None:
            x = self.x.data.clone()
        labels = ( self.label * torch.ones( x.shape[0], device=x.device ) ).long()
            
        logits = self.logits( x )
        loss = loss_fn( logits, labels )
        loss = loss.mean( dim=0 )
        return loss