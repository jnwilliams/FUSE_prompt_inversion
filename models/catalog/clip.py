import re 
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import CLIPProcessor, CLIPModel

try:
    from prompt_inversion.models.language_utils import LanguageUtils
except ModuleNotFoundError as e:
    from models.language_utils import LanguageUtils
except Exception as e:
    raise e

from PIL import Image

class Clip( LanguageUtils ):
    """
    Driver for working with CLIP.
    """

    def __init__( self, model, processor, tokenizer, vocab, config, **kwargs  ):
        """
            Initialization

            device : String
                - (cuda/cpu) whether to run on the gpu or cpu
            prompt_len : int
                - A fixed sequence length for the model. Optimizing the embedding
                for a specific task will always keep the same sequence length
            image_path : str or None
                - The path to the image in order to compare against an image embedding.
                If set to none, the image embedding is set to 0's of the appropriate dimension
        """

        #Load the model and tokenizer
        super( Clip, self ).__init__( vocab, tokenizer, **kwargs )

        self.model = model
        self.processor = processor
        self.config = config
        self.parameters = kwargs.get( "parameters", {} )
        self.loss_function = kwargs.get( "loss_function", None )

        self.set_image( kwargs.get( 'image_path', None ) )

        self.set_prefix( self.parameters.get( "prefix", "" ) )
        self.set_suffix( self.parameters.get( "suffix", "" ) )

        self.attn_mask = self._make_causal_mask( 
            ( self.x.shape[0], self.tokenizer.model_max_length ), 
            self.dtype, 
            device=self.device 
        )

        # Clip's positional embeddings. Added separately when computing losses/logits
        position_embedding = \
            self.model.text_model.embeddings.position_embedding.weight
        self.position_embedding = position_embedding.unsqueeze(0).to( self.device )

        self.eos_embedding = self.vocab[ self.tokenizer.eos_token_id ]
        
    def set_image( self, image_path = None ):
        if image_path is None:
            self.image_embed = torch.randn( 1, self.dims, device=self.device  )
        elif type( image_path ) is list:
                self.image = []
                self.image_embed = torch.zeros( 
                    len( image_path ), 
                    self.dims, 
                    device=self.device 
                )

                for i, path in enumerate( image_path ):
                    image = Image.open( path )
                    self.image.append( image )
                    self.image_embed[i] = self.PIL_to_embed( image ).to( self.device )
        else:
            self.image = Image.open( image_path ).convert('RGB')  
            self.image_embed = self.PIL_to_embed( self.image ).to( self.device )

    
    def PIL_to_embed( self, image  ):
        """
            Get Clip image embeddings from a PIL image

            Inputs:
                image : PIL.Image
                - PIL image
            Outputs:
                im_embedding : torch.tensor
                - clip image embedding
        """
        inputs = self.processor.image_processor( image, return_tensors="pt", do_normalize=True ).to( self.device )
        vision_outputs = self.model.vision_model( inputs["pixel_values"].to( self.dtype ) )[1]
        im_embedding = self.model.visual_projection( vision_outputs )
        
        return im_embedding

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self, input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def logits( self, x=None, **kwargs ):
        if x is None:
            x = self.x.data.clone()

        logits = torch.ones(
            x.size(0), x.size(1), self.tokenizer.vocab_size,
            device = self.device )
        logits *= 1 / self.tokenizer.vocab_size

        return torch.log( logits ).to( x.dtype )

    def loss( self, x=None, **kwargs ):
        """
            Cosine Similarity between text and image embedding

            Inputs:
                x : Torch.Tensor or None
                    - input sequence of shape ( Batch x Seq Len x Dim ).
                    If None, use the stored initialization sequence saved under self.x

            Outputs:
                loss : Torch.Tensor
                    - output of shape ( Batch x 1 ) cosine distance between each input
                    tensor and the saved image embedding
        """

        if x is None:
            x = self.x.data.clone()

        b, _, d = x.shape
        x = self.add_special_tokens( x )
        with torch.no_grad():
            idxs = torch.cdist( self.eos_embedding.repeat( b, 1, 1 ), x )
            idxs = idxs.argmin( dim = -1 ).reshape( -1 )

        x = self.model.text_model.embeddings( inputs_embeds=x )
        x = self.model.text_model.encoder( 
            x, 
            causal_attention_mask = self.attn_mask.repeat( x.shape[0], 1, 1, 1 )
        ).last_hidden_state

        x = self.model.text_model.final_layer_norm( x )
        x = x[ :, idxs ]
        x = self.model.text_projection( x )
        
        x = torch.diagonal( x, dim1=0, dim2=1 ).T
        x = x / x.norm( p=2, dim=-1, keepdim=True )
        
        y = self.image_embed
        y /= y.norm( p=2, dim=-1, keepdim=True )
        y = y.to( x.dtype )

        cosine_sim = torch.matmul(x, y.t())
        loss = 1 - cosine_sim

        return loss.mean( dim=-1 )
