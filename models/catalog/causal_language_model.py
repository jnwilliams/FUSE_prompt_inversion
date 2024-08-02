import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from prompt_inversion.models.language_utils import LanguageUtils
except ModuleNotFoundError as e:
    from models.language_utils import LanguageUtils
except Exception as e:
    raise e

class Causal_LM( LanguageUtils ):
    def __init__( self, model, tokenizer, vocab, config, **kwargs  ):
        super( Causal_LM, self ).__init__( vocab, tokenizer, **kwargs )
        
        self.model = model
        self.config = config
        self.parameters = kwargs.get( "parameters", {} )
        self.loss_function = kwargs.get( "loss_function", None )

        self.set_prefix( self.parameters.get( "prefix", "" ) )
        self.set_suffix( self.parameters.get( "suffix", "" ) )

        if self.loss_function is not None:
            if self.loss_function["type"] == "cross_entropy_loss":
                self.target_x = self.from_text( self.loss_function["label"] )

                _, idxs = self.proj_to_vocab( self.target_x )
                self.target_idxs = idxs

            elif self.loss_function['type'] == 'kl_div_loss':
                self.pinv = torch.linalg.pinv( self.vocab.float() ).to( self.dtype )


    def logits( self, x = None, **kwargs ):
        if x is None:
            x = self.x.data.clone()

        return self.model( inputs_embeds = x ).logits
    
    def cross_entropy_loss( self, x = None, **kwargs ):
        loss_fn = F.cross_entropy
        b, s, d = x.shape
        target_x = self.target_x.repeat( b, 1, 1 )
        target_idxs = self.target_idxs.repeat( b, 1 )

        if x is None:
            x = self.x.data.clone()
        
        x = torch.concat( [
            x, target_x
        ], dim=1 )

        length = target_x.size(1)

        logits = self.logits( x, **kwargs )
        loss = loss_fn( 
            logits[:,- ( 1 + length ):-1].permute( 1, 2, 0 ),
            target_idxs.permute( 1, 0 ),
            reduction = 'none'
        )
        
        loss = loss.mean( dim=0 )
        return loss 

    def kl_div_loss( self, x = None, **kwargs ):
        b, s, d = x.shape
        loss_fn = torch.nn.KLDivLoss( reduction = 'none', log_target=True )

        log_input = torch.log_softmax( x[:,1:] @ self.pinv, dim=-1 )

        with torch.no_grad():
            log_target = torch.log_softmax( self.logits( x, **kwargs )[:,:-1], dim=-1 )

        loss = loss_fn( log_input, log_target )
        loss = loss.sum( dim=-1 ).mean( dim=-1 )
        
        return loss

    def log_prob_loss( self, x=None, **kwargs ):
        b, s, d = x.shape
    
        with torch.no_grad():
            output = self.model( inputs_embeds = x, output_hidden_states=True )
            last_hidden_state = output.hidden_states[-1]

        numer = torch.bmm( x[:,1:], last_hidden_state[:,:-1].permute( 0, 2, 1 ) )
        numer = torch.diagonal( numer, dim1=-1, dim2=-2 )

        denom = self.vocab @ last_hidden_state[:,:-1].permute( 0, 2, 1 )
        loss = numer - torch.logsumexp( denom, dim=1 )
        
        return - loss.sum( dim=-1 )

    def nll_loss( self, x = None, probs = None, **kwargs ):
        if x is None:
            x = self.x.data.clone()

        b, s, d = x.shape
        loss_fn = F.cross_entropy

        if probs is None:
            _, idxs = self.proj_to_vocab( x, labels_only=True )
            probs = F.one_hot( idxs, num_classes = self.tokenizer.vocab_size ).to( x.dtype )
        else:
            prefix = F.one_hot( self.idxs_prefix, num_classes = self.tokenizer.vocab_size )
            
            if self.idxs_suffix.size(-1) > 0:
                suffix = F.one_hot( self.idxs_suffix, num_classes = self.tokenizer.vocab_size )
            else:
                suffix = torch.zeros( b, 0, self.tokenizer.vocab_size, device=self.x.device )

            probs = torch.concat( [ prefix, probs, suffix ], dim=1 )
        
        probs = probs[:,1:]
        with torch.no_grad():
            logits = self.logits( x, **kwargs )
            log_prob = torch.log_softmax( logits, dim=-1 )

        loss = loss_fn( 
            log_prob[:,:-1].permute( 1, 2, 0 ),
            probs.permute( 1, 2, 0 ),
            reduction = 'none',
        )

        loss = loss.mean( dim=0 )
        return loss


    def loss( self, x = None, **kwargs ):
        if self.loss_function["type"] == "cross_entropy_loss":
            return self.cross_entropy_loss( x = x, **kwargs )
        elif self.loss_function["type"] == "nll_loss":
            return self.nll_loss( x = x, **kwargs )
        elif self.loss_function["type"] == "kl_div_loss":
            return self.kl_div_loss( x = x, **kwargs )
        elif self.loss_function["type"] == "log_prob_loss":
            return self.log_prob_loss( x = x, **kwargs )
        else:
            raise AttributeError( 
                "Loss function type must be 'nll_loss', " +
                "'kl_div_loss', 'log_prob_loss', or 'cross_entropy_loss'" +
                "Check config.yml for misspellings."
            )

    def generate( self, x = None, max_new_tokens=100, **kwargs ):
        x, idxs = self.proj_to_vocab( x )

        gen_config = self.model.generation_config
        gen_config.temperature = None
        gen_config.top_p = None
        gen_config.do_sample = False
        gen_config.num_beams = 1

        outputs = self.model.generate(
            inputs_embeds = x,
            max_new_tokens = max_new_tokens,
            generation_config = gen_config
        )

        return self.tokenizer.batch_decode( outputs )