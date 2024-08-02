import re
import json
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast

#Map from float32 to float16
def autocast_wrapper(func):
    def wrapper(*args, **kwargs):
        # Check if any of the arguments are of type float16
        input_is_float16 = any(arg.dtype == torch.float16 for arg in args)

        # If input is float16, perform operations in float32 using autocast
        if input_is_float16:
            with autocast():
                # Convert float16 inputs to float32 except for CONST_MAT
                converted_args = [arg.float() if arg.dtype == torch.float16 else arg for arg in args]
                output = func(*converted_args, **kwargs)
        else:
            output = func(*args, **kwargs)
        
        # Cast the output back to float16 if input was float16
        if input_is_float16:
            output = ( out.half() if out.dtype is torch.float32 else out for out in output )
        
        return output
    
    return wrapper

class LanguageUtils( ):
    """
        Parent Class for models used here. Handles projecting from continuous to discrete
        embedding spaces, converting from embedding to text and vice versa, and adding/removing
        special tokens when computing losses for each model
    """
    def __init__( self, vocab, tokenizer, device = "cuda", prompt_len = 8, ignore_non_ascii = True, init_token = ".", **kwargs ):
        """
        Initialization

            vocab : Torch.tensor
                - The ( |V| x D ) token embedding matrix for the model
            tokenizer : Huggingface Tokenizer Object
                - The tokenizer for a specific model. Each class inheriting from this parent
                should have a load_model function that retrieves this
            device : String
                - (cuda/cpu) whether to run on the gpu or cpu
            prompt_len : int
                - A fixed sequence length for the model. Optimizing the embedding
                for a specific task will always keep the same sequence length

            **kwargs
            
            prefix : String
                - If you would like to include some text to condition a model on that will be
                exempt from the optimization, give it here and zero out gradients like
                    ```
                    self.init( ..., prefix="hello, world." )
                    x = self.x.data.clone().requires_grad_( True )
                    optim = torch.optim.SGD( ( x, ), lr=1e-3 )

                    loss = model.loss( x )
                    loss.backward()

                    x.grad[:prefix] = 0.
                    optim.step()
                    ```
            lr : float
                - Default 1e-1 - Learning rate for the optimizer
            weight_decay : float
                - Default 1e-1 - Weight Decay for the optimizer
            optimizer : torch.optim.[Optimizer]
                - Default AdamW. Optimzier for a method. 
        """
        self.device = device
        self.dtype = vocab.dtype
        self.vocab = vocab
        self.dims = self.vocab.size( -1 )
        self.dtype = self.vocab.dtype
        self.tokenizer = tokenizer
        self.ignore_non_ascii = ignore_non_ascii
        self.is_vision_language_model = False

        if self.ignore_non_ascii:
            self.set_tokens_to_ignore( )

        self.prompt_len = int( prompt_len ) 
        self.x, _ = self.set_embedding( init_token=init_token, dtype=self.dtype )
        self.x.requires_grad_()
        self.x.grad = torch.zeros_like( self.x )
        
    def set_prefix( self, prefix ):
        self.prefix = prefix
        if self.prefix != "":
            x_prefix = self.from_text( self.prefix, remove_special_tokens=False )
        else:
            x_prefix = torch.zeros( 1, 0, self.dims, device=self.device, dtype=self.dtype )

        self.x_prefix, self.idxs_prefix = self.proj_to_vocab( x_prefix )
        self.prefix_len = self.x_prefix.shape[1]

    def set_suffix( self, suffix ):
        self.suffix = suffix
        if self.suffix != "":
            x_suffix = self.from_text( self.suffix, remove_special_tokens=False )
        else:
            x_suffix = torch.zeros( 1, 0, self.dims, device=self.device, dtype=self.dtype )
        
        self.x_suffix, self.idxs_suffix = self.proj_to_vocab( x_suffix )
        self.suffix_len = self.x_suffix.shape[1]

    def set_embedding( self, init_token=".", dtype=torch.float32 ):                
        self.init_id = self.tokenizer( init_token, add_special_tokens=False ).input_ids[0]
        x = self.vocab[ self.init_id ].reshape( 1, 1, -1 )
        x = x.repeat( 1, self.prompt_len, 1 )

        x, idxs = self.proj_to_vocab( x )
        return x, idxs

    def prepare_embedding( self, x = None ):
        if x is None:
            x = self.x.data

        b, _, _ = x.shape
        
        x = torch.concat( [
            self.x_prefix.repeat( b, 1, 1 ).to( x.device ),
            x,
        ], dim=1 )

        padding_x = self.vocab[ self.tokenizer.pad_token_id ]
        move_padding_to_left = (x == padding_x).all(dim=-1)
        move_padding_to_left = move_padding_to_left.to( torch.float32 )

        pad_direction = ( self.tokenizer.padding_side.lower() == "left" )
        sorted_indices = torch.argsort( move_padding_to_left,  descending=True, dim=1, stable=True )

        x = x[torch.arange(b), sorted_indices.T].permute(1, 0, 2)        
        x = torch.concat( [
            x,
            self.x_suffix.repeat( b, 1, 1 ).to( x.device )
        ], dim=1 )
        
        return x

    def set_tokens_to_ignore( self, ignore_tokens = None ):
        """
            If necessary, keep a set of tokens that we may want to ignore under certain circumstances
        """
        if ignore_tokens is None:
            ignore_tokens = []

        tokens = self.tokenizer.batch_decode( torch.arange( self.tokenizer.vocab_size ) )
        for i, tok in enumerate( tokens ):
            try:
                if tok == '':
                    ignore_tokens.append( i )

                tok.encode().decode( "ascii" )
            except UnicodeDecodeError as e:
                ignore_tokens.append( i )
            except Exception as e:
                raise( e )
            
        self.ignore_tokens = ignore_tokens

    def add_special_tokens( self, x, max_len=None ):
        """
            Add special tokens to a sequence. E.g., 
                Beginning of String Tokens, Pad Tokens, End of String Tokens
                

            Inputs:
                x : Torch.Tensor
                    - input sequence of shape ( Batch x Seq Len x Dim ).
                max_len : int
                    - the returned length of a tensor after adding special tokens.

            Outputs:
                x : Torch.Tensor
                    - [ BOS_TOKEN, X, PAD_TOKEN{0:max_len-2}, EOS_TOKEN]
 
        """
        tokenizer = self.tokenizer
        vocab = self.vocab.data.clone()

        if max_len is None:
            max_len = tokenizer.model_max_length

        if x.shape[1] == max_len:
            return x

        start_token = tokenizer.bos_token_id
        start_token = vocab[ start_token ].tile( x.shape[0], 1, 1 )

        end_token = tokenizer.eos_token_id
        end_token = vocab[ end_token ].repeat( 
                1, max_len - x.shape[1] - 1, 1 
        )
        end_token = end_token.tile( x.shape[0], 1, 1 )

        x = torch.cat( [ 
            start_token.to( vocab.device ),
            x.to( vocab.device ),
            end_token.to( vocab.device )
        ], dim=1 )   

        return x
    
    def remove_special_tokens( self, x ):
        """
            Remove special tokens. Inverse of self.add_special_tokens

            Inputs:
                x : Torch.Tensor
                    - input sequence of shape ( Batch x Seq Len x Dim ).
                    - [ BOS_TOKEN, X, PAD_TOKEN{0:max_len-2}, EOS_TOKEN]
                max_len : int
                    - the returned length of a tensor after adding special tokens.

            Outputs:
                x : Torch.Tensor
                    - Return only "X" from Inputs example above.
        """
        tokenizer = self.tokenizer
        vocab = self.vocab.clone()

        if tokenizer.bos_token_id is not None:
            start_token = tokenizer.bos_token_id
            start_token = vocab[ start_token ]

            dist = torch.norm( start_token.to( vocab.device ) - x, p=1, dim=-1 )
            dist = torch.any( dist, dim=0 )
            x = x[ :, dist > 1e-6 ]

        if tokenizer.eos_token_id is not None:
            end_token = tokenizer.eos_token_id
            end_token = vocab[ end_token ]

            dist = torch.norm( end_token.to( vocab.device ) - x, p=1, dim=-1 )
            dist = torch.any( dist, dim=0 )
            x = x[ :, dist > 1e-6 ]

        if tokenizer.pad_token_id is not None:
            pad_token = tokenizer.pad_token_id
            pad_token = vocab[ pad_token ]

            dist = torch.norm( pad_token.to( vocab.device ) - x, p=1, dim=-1 )
            dist = torch.any( dist, dim=0 )
            x = x[ :, dist > 1e-6 ]

        return x.reshape( x.shape[0], -1, self.dims )
    
    def to_text( self, x = None, indices=None ):
        """
            Convert a sequence embedding or set of indices to text

            Inputs:
                x : Torch.Tensor or None
                    - input sequence of shape ( Batch x Seq Len x Dim ).
                    If None, use the stored initialization sequence saved under self.x
                indices : Int
                    - If you already know the indices of the tokens, then just give those
                    to the decoder
            
            Outputs:
                text : String
                    - Text that has been converted from the embedding space

        """
        tokenizer = self.tokenizer 

        if indices is None:
            if x is None:
                x = self.x.data.clone()
            
            _, idxs = self.proj_to_vocab( x )
        else:
            idxs = indices

        text = tokenizer.batch_decode( idxs )

        return text

    def from_text( self, text : str, max_len=None, remove_special_tokens=False ):
        """
            Convert text to the sequence embeddings

            Inputs:
                text : String
                    - String to be converted to an embedding
                max_len : int
                    Either truncates or pads the text so that the sequence
                    is of length max_len

            Outputs:
                x : Torch.Tensor
                    - Sequence of shape ( 1 x Seq Len x Dim ).
        """
        
        tokenizer = self.tokenizer
        vocab = self.vocab.data.clone()
    
        if max_len is None:
            max_len = tokenizer.model_max_length
        max_len = min( max_len, 256 ) 
        
        if type( text ) is list:
            kwargs = { "add_special_tokens" : False, "padding": "max_length", "max_length": max_len, "truncation": True }
        elif type( text ) is str:
            kwargs = { "add_special_tokens": False }

        text_ids = tokenizer( 
                        text,
                        return_tensors="pt",
                        **kwargs
            ).input_ids.to( vocab.device )

        x = vocab[ text_ids ]
        if remove_special_tokens:
            x = self.remove_special_tokens( x )

        return x

    def proj_to_vocab( self, x, **kwargs ):
        if x.shape[1] == 0:
            return x, torch.zeros( x.shape[:2], device=x.device ) 

        autocast_proj = autocast_wrapper( self._proj_to_vocab )
        b, s, d = x.shape
        max_batch_size = kwargs.get( "max_batch_size", 14 ) 

        x, idxs = zip( *[ autocast_proj( x_i, **kwargs ) 
                            for x_i in torch.split( x, max_batch_size ) 
        ] )
        x, idxs = torch.cat( x, dim=0 ), torch.concat( idxs, dim=0 )

        return x, idxs

    def _proj_to_vocab( self, x, labels_only=False, **kwargs ):
        """
            Project from a continuous embedding space to a discrete one.
            Find the nearest token in the vocabulary and replace the token
            with that one. Uses Cosine Similarity as matrix operations
            are faster than computing the Euclidean distance.

            Inputs:
                x : Torch.Tensor or None
                    - input sequence of shape ( Batch x Seq Len x Dim ).

        """
        b, s, d = x.shape
        max_seq_len = kwargs.get( "max_seq_len", 8 ) 
        if s > max_seq_len:
            x, idxs = zip( *[ self._proj_to_vocab( x_i, max_seq_len=max_seq_len ) 
                             for x_i in torch.split( x, max_seq_len, dim=1 ) 
            ] )
            x, idxs = torch.cat( x, dim=1 ), torch.concat( idxs, dim=1 )
            return x, idxs

        vocab = self.vocab.data.to( x.dtype ).clone()
        vocab = vocab.to( x.device )
        if self.ignore_non_ascii is True:
            vocab[ list( self.ignore_tokens ) ] = - x.mean( dim=[0,1] )

        dot_ = torch.einsum( 'mjk,ilk->ijl', vocab.unsqueeze(0), x )
        vocab_norm = torch.norm( vocab, dim=-1, p=2 ).tile( b, 1 )
        vocab_norm = vocab_norm.unsqueeze(-1)
        emb_norm = torch.norm( x, dim=-1, p=2 ).unsqueeze(1)

        dot_ = dot_ / torch.bmm( vocab_norm, emb_norm )
        dot_ = torch.nan_to_num( dot_ )
        idxs = dot_.argmax( dim=1 )

        if labels_only:
            return torch.zeros_like( x ), idxs 

        x = self.vocab[ idxs ]
        return x, idxs

    def split_embedding_by_whitespace( self, text = None, x = None ):
        if x is None:
            x = self.x.data.clone()

        if text is None:
            text = self.to_text( x )
            text = [ t.replace( self.tokenizer.pad_token, f" {self.tokenizer.pad_token}" ) for t in text ]
            
        b, s, d = x.shape
        vocab = self.vocab.to( x.device )

        batch_text_list = [ list( filter( None, re.split( '\s+', t ) ) ) for t in text ]
        max_len = max( [ len( t ) for t in batch_text_list ] )
        padding = [ max_len - len( t ) for t in batch_text_list ]

        batch_text_list = [ t + [ self.tokenizer.pad_token ] * x.shape[1] for t in batch_text_list ]
        batch_text_list = [ t[:max_len] for t in batch_text_list ]

        _, idxs = self.proj_to_vocab( x )
        batch_idxs = idxs.tolist()
        batch_idxs = [ 
            t + [ self.tokenizer.pad_token_id ] * pad for t, pad in zip( batch_idxs, padding ) 
        ]

        word_tensor = []
        for batch, ( text_list, idx, pad ) in enumerate( zip( batch_text_list, batch_idxs, padding ) ):
            whitespace_separated = []
            token_idx = 0
            
            x_ = F.pad( x[[batch]], ( 0, 0, 0, pad ) )
            x_[:,-pad:] += vocab[ self.tokenizer.eos_token_id ]

            for i, word in enumerate( text_list ):
                word_embedding = torch.zeros((0,d), device=x.device )
                ids = []
                for j, token in enumerate( idx[ token_idx: ] ):                       
                    ids.append( token )

                    text_ = self.to_text( indices=[ids] )[0]
                    text_ = text_.replace( self.tokenizer.pad_token, f" {self.tokenizer.pad_token}" )

                    if re.match(f'^.+[ \t\r\f\v]', text_ ):
                        whitespace_separated.append( x_[0,token_idx:token_idx+j,:] )
                        token_idx += j
                        break

            whitespace_separated.append( x_[0,token_idx:,:] )
            whitespace_separated = torch.nn.utils.rnn.pad_sequence( whitespace_separated, batch_first=True )
            word_tensor.append( whitespace_separated )

        max_seq_len = max( [ seq.shape[1] for seq in word_tensor ] )
        word_tensor = [ F.pad( seq, ( 0, 0, 0, max_seq_len - seq.shape[1] ) ) for seq in word_tensor ]
        word_tensor =  torch.nn.utils.rnn.pad_sequence( word_tensor, batch_first=True )

        return word_tensor