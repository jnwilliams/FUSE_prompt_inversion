import os
import re 
import inspect

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import AutoProcessor

import torchvision.transforms as tfms 
from contextlib import contextmanager, nullcontext

try:
    from prompt_inversion.models.language_utils import LanguageUtils
except ModuleNotFoundError as e:
    from models.language_utils import LanguageUtils
except Exception as e:
    raise e
    
from PIL import Image

import shutil

import matplotlib.pyplot as plt
from contextlib import contextmanager

class Seed:
    def __init__( self, seed ):
        self.seed = seed
        self.rng_state = None
        self.cuda_rng_state = None

    def __enter__( self ):
        self.rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_rng_state = torch.cuda.get_rng_state_all()

        torch.manual_seed( self.seed )
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng_state)


class BackwardPassInfo( torch.autograd.Function ):
    stored_gradients = []

    @staticmethod
    def forward( ctx, x ):
        return x

    @staticmethod
    def backward( ctx, grad_out ):
        BackwardPassInfo.stored_gradients.append( grad_out.flatten().clone() )

        return grad_out

class StableDiffusion( LanguageUtils ):
    """
    Driver for working with CLIP.
    """

    def __init__( self, pipeline, text_model, tokenizer, vocab, config, **kwargs  ):
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
        super( StableDiffusion, self ).__init__( vocab, tokenizer, **kwargs )

        self.pipeline = pipeline
        self.config = config
        self.text_model = text_model      
        self.processor = AutoProcessor.from_pretrained( "openai/clip-vit-base-patch32" )
          
        self.set_image( kwargs.get( 'image_path', None ) )

        self.parameters = kwargs.get( "parameters", {} )
        self.loss_function = kwargs.get( "loss_function", None )

        self.set_prefix( self.parameters.get( "prefix", "" ) )
        self.set_suffix( self.parameters.get( "suffix", "" ) )

        self.num_timesteps = 20
        self.pipeline.scheduler.set_timesteps( self.num_timesteps )
        self.timesteps = self.pipeline.scheduler.timesteps.to( self.pipeline.vae.device )
        
        self.alphas_cumprod = self.pipeline.scheduler.alphas_cumprod.to( self.pipeline.vae.device )
        self.alphas_cumprod = self.alphas_cumprod.to( self.pipeline.vae.dtype )

        self.seed = 3

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

    def set_image( self, image_path = None ):
        device = self.pipeline.vae.device
        dtype = self.pipeline.vae.dtype

        if image_path is None:
            return

        self.image_path = image_path
        image = Image.open( image_path ).convert( 'RGB' )

        image = image.resize( ( 512, 512 ) )
        image = tfms.functional.to_tensor(image).to( dtype ).unsqueeze(0).to(device) * 2 - 1
        latent = self.pipeline.vae.encode( image )

        self.latent_dist = latent.latent_dist
        self.sample_scaling = 0.18215
        self.pipeline.enable_vae_slicing()
        
    def text_embedding( self, x=None, **kwargs ):
        """
            Get Clip text embedding from a sequence embedding

            Inputs:
                x : Torch.Tensor or None
                    - input sequence of shape ( Batch x Seq Len x Dim ).
                    If None, use the stored initialization sequence saved under self.x

            Outputs:
                text_embed : Torch.Tensor
                    - clip text embedding

        """
        if x is None:
            x = self.x.data.clone()
        
        x = self.add_special_tokens( x )
        
        x = self.text_model.embeddings( inputs_embeds=x )
        x = self.text_model.encoder( 
            x, 
            causal_attention_mask = self._make_causal_mask( 
                x.shape[:2], 
                dtype=x.dtype, 
                device=x.device 
            )
        )
        x = x.last_hidden_state
        x = self.text_model.final_layer_norm( x )

        return x

    def logits( self, x=None, **kwargs ):
        if x is None:
            x = self.x.data.clone()

        logits = torch.ones(
            x.size(0), x.size(1), self.tokenizer.vocab_size,
            device = self.device )
        logits *= 1 / self.tokenizer.vocab_size

        return torch.log( logits ).to( x.dtype )

    def loss( self, x=None, batch_size=1, guidance_scale=3.5, **kwargs ):
        BackwardPassInfo.stored_gradients.clear()
        
        if x is None:
            x = self.x.data.clone()

        text_embedding = self.text_embedding( x ).repeat( batch_size, 1, 1 )

        latent = self.latent_dist.sample().repeat( batch_size, 1, 1, 1 )
        latent *= self.sample_scaling
        loss = 0
        
        samples = torch.linspace( 0, self.num_timesteps, 3 ).long()
        for i, t in enumerate( self.timesteps ):
            context = torch.no_grad() if i not in samples else nullcontext()
            noise = torch.randn_like( latent )
            noisy_latent = self.pipeline.scheduler.add_noise( latent.clone(), noise, t )

            with context:
                _text_embedding = BackwardPassInfo.apply( text_embedding.clone() )
                _, noise_pred = self.diffusion_sample( 
                    noisy_latent, t, _text_embedding, guidance_scale=guidance_scale 
                )

            loss += F.mse_loss( noise_pred, noise )

        with torch.no_grad():
            with Seed( self.seed ):
                image = self.sample( self.to_text( x )[0], num_inference_steps=self.num_timesteps )
                image = np.array( image[0] )

            with Seed( self.seed ):
                recovered_image = self.sample( self.to_text( x )[0], num_inference_steps=50, do_classifier_free_guidance=True )
                recovered_image = np.array(recovered_image[0])

            latent_image = self.pipeline.decode_latents( latent )
            latent_image = self.pipeline.numpy_to_pil(latent_image)
            latent_image = np.array(latent_image[0])

            fig, ax = plt.subplots( 1, 3, figsize=(10,10) )
            ax[0].imshow( image )
            ax[1].imshow( recovered_image )
            ax[2].imshow( latent_image )

            plt.draw()
            plt.pause(1.)
            plt.close()

        return loss
    
    def scheduler_step( self, latents, noise_pred, t ):
        prev_t = max(
            1, t.item() - (1000 // self.pipeline.scheduler.num_inference_steps)
        ) # t-1
        alpha_t = self.alphas_cumprod[t.item()].reshape( -1, 1, 1, 1 )
        alpha_t_prev = self.alphas_cumprod[prev_t].reshape( -1, 1, 1, 1 )
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

        return latents
    
    def diffusion_sample( self, latents, timestep, text_embedding, guidance_scale=1. ):
        do_classifier_free_guidance = guidance_scale > 1.
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if do_classifier_free_guidance:
            uncond_text_embedding = self.text_embedding( self.from_text( "<|endoftext|>" ) )
            text_embedding = torch.concat( [ uncond_text_embedding, text_embedding ] )

        noise_pred = self.pipeline.unet( 
            latent_model_input,
            timestep, 
            encoder_hidden_states = text_embedding ).sample
        
        if do_classifier_free_guidance:
            # Predict the noise residual
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = self.scheduler_step(
            latents,
            noise_pred, 
            timestep,
        )

        return latents, noise_pred
    
    def sample(
        self,
        prompt,
        start_step=0,
        start_latents=None,
        guidance_scale=3.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt="",
        device="cuda",
    ):
        # Encode prompt
        text_embeddings = self.pipeline._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Set num inference steps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

        # Create a random starting point if we don't have one already
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=device).to( self.pipeline.vae.dtype )
            start_latents *= self.pipeline.scheduler.init_noise_sigma

        latents = start_latents.clone()

        for i in range(start_step, num_inference_steps):

            t = self.pipeline.scheduler.timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Normally we'd rely on the scheduler to handle the update step:
            # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000 // self.pipeline.scheduler.num_inference_steps))  # t-1
            alpha_t = self.alphas_cumprod[t.item()]
            alpha_t_prev = self.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
            latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

        # Post-processing
        images = self.pipeline.decode_latents(latents)
        images = self.pipeline.numpy_to_pil(images)

        return images
