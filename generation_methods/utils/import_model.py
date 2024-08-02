from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pydantic import BaseModel

import torch
import requests
import json

import transformers
from transformers import AutoConfig
from diffusers import DiffusionPipeline
from operator import attrgetter

try:
    from prompt_inversion import models
except ModuleNotFoundError as e:
    import models
except Exception as e:
    raise e

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ArchitectureInfo(ABC):
    @abstractmethod
    def embed_weights(self) -> Union[dict,str]:
        pass

    @abstractmethod
    def get_uses_images(self) -> bool:
        pass

class StaticTensorNames(ArchitectureInfo, BaseModel, frozen=True):
    name: str

    embed_weight_names: Union[dict,str]  # weights for embed/lm_head
    uses_images: bool 

    def embed_weights(self) -> Union[dict,str]:
        return self.embed_weight_names

    def get_uses_images(self) -> bool:
        return self.uses_images 

GPT2_INFO = StaticTensorNames(
    name="GPT2LMHeadModel",
    embed_weight_names="transformer.wte.weight",
    uses_images=False,
)

CLIP_INFO = StaticTensorNames(
    name="CLIPModel",
    embed_weight_names="text_model.embeddings.token_embedding.weight",
    uses_images=True,
)

BLIP_INFO = StaticTensorNames(
    name="BlipForConditionalGeneration",
    embed_weight_names="text_decoder.bert.embeddings.word_embeddings.weight",
    uses_images=True
)

BLIP2_INFO = StaticTensorNames(
    name="Blip2ForConditionalGeneration",
    embed_weight_names={
        "OPTForCausalLM": "language_model.model.decoder.embed_tokens.weight",
        "T5ForConditionalGeneration": "language_model.encoder.embed_tokens.weight"
    },
    uses_images=True
)

LLAMA_INFO = StaticTensorNames(
    name="LlamaForCausalLM",
    embed_weight_names="model.embed_tokens.weight",
    uses_images=False
)

LLAMA_CHAT_INFO = StaticTensorNames(
    name="LlamaForCausalLM",
    embed_weight_names="model.embed_tokens.weight",
    uses_images=False
)

FALCON_INFO = StaticTensorNames(
    name="RWForCausalLM",
    embed_weight_names="transformer.word_embeddings.weight",
    uses_images=False
)

ROBERTA_INFO = StaticTensorNames(
    name="RobertaForSequenceClassification",
    embed_weight_names="roberta.embeddings.word_embeddings.weight",
    uses_images=False
)

MISTRAL_INFO = StaticTensorNames(
    name="MistralForCausalLM",
    embed_weight_names = "model.embed_tokens.weight",
    uses_images=False
)

STABLEDIFFUSION_INFO = StaticTensorNames(
    name="StableDiffusionPipeline",
    embed_weight_names="text_encoder.text_model.embeddings.token_embeddings.weight",
    uses_images=True
)


def check_model_supported( config ):
    arch_name = config.architectures[0]
    supported = [
        GPT2_INFO,
        CLIP_INFO,
        BLIP_INFO,
        BLIP2_INFO,
        LLAMA_INFO,
        FALCON_INFO,
        ROBERTA_INFO,
        MISTRAL_INFO,
        STABLEDIFFUSION_INFO
    ]
    for arch in supported:
        if arch.name == arch_name:
            return arch

    raise RuntimeError(f"Unsupported architecture {arch_name}")

def validate_tokenizer( tokenizer ):
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    return tokenizer

def import_model( model_id, **kwargs ):
    """
        Import a model for use with one of the methods in the 
        generation_methods directory
    """

    device = kwargs.get( "device", DEVICE )
    dtype = kwargs.get( "dtype", "torch.float16" )
    dtype = eval( dtype )
    
    try:
        config = AutoConfig.from_pretrained( model_id, trust_remote_code=True )
    except OSError as e:
        r = requests.get( f"https://huggingface.co/{model_id}/raw/main/model_index.json")
        if r.status_code == 404:
            raise( e )

        config = json.loads( r.text )
        config.update( {"_name_or_path": model_id } )
        config[ "architectures" ] = [ config.pop( "_class_name" ) ]       
        config = transformers.PretrainedConfig.from_dict( config )

    architecture = check_model_supported( config ) 
    
    if architecture.name == "GPT2LMHeadModel":
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        model = GPT2LMHeadModel.from_pretrained( 
            model_id,
            torch_dtype = dtype
        ).to( device )
        
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        tokenizer = GPT2TokenizerFast.from_pretrained( model_id )
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.causal_language_model.Causal_LM( model, tokenizer, vocab, config, **kwargs )

    if architecture.name == "CLIPModel":
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained( 
            model_id,
            torch_dtype = dtype
        ).to( device )
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        processor = CLIPProcessor.from_pretrained( model_id, torch_dtype = torch.float16 )
        tokenizer = processor.tokenizer
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.clip.Clip( model, processor, tokenizer, vocab, config, **kwargs )

    if architecture.name == "BlipForConditionalGeneration":
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model = BlipForConditionalGeneration.from_pretrained( model_id ).to( device )
        
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        processor = BlipProcessor.from_pretrained( model_id )
        tokenizer = processor.tokenizer
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return model, processor, tokenizer

    if architecture.name == "Blip2ForConditionalGeneration":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        model = Blip2ForConditionalGeneration.from_pretrained( model_id ).to( device )
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        processor = Blip2Processor.from_pretrained( model_id )
        tokenizer = processor.tokenizer
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names[model.language_model._get_name()])(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return model, processor, tokenizer

    if architecture.name == "LlamaForCausalLM":
        from transformers import pipeline, LlamaTokenizer

        generator = pipeline(            
            "text-generation",                   
            model=model_id,                      
            torch_dtype=torch.float16,           
            device_map="auto"                    
        )                 
        model = generator.model
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        tokenizer = LlamaTokenizer.from_pretrained( model_id )
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.causal_language_model.Causal_LM( model, tokenizer, vocab, config, **kwargs )

    if architecture.name == "RWForCausalLM":
        from transformers import pipeline, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained( model_id )
        tokenizer = validate_tokenizer( tokenizer )

        pipeline =  pipeline( 
            "text-generation", 
            model=model_id, 
            tokenizer=tokenizer, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            device_map="auto" 
        )

        model = pipeline.model.to( torch.float32 )
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.causal_language_model.Causal_LM( model, tokenizer, vocab, config, **kwargs )

    if architecture.name == "RobertaForSequenceClassification":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained( model_id ).to( device )
        
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained( model_id )
        tokenizer = validate_tokenizer( tokenizer )
        
        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.roberta.Roberta( model, tokenizer, vocab, config, **kwargs )

    if architecture.name == "MistralForCausalLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained( model_id )
        tokenizer = validate_tokenizer( tokenizer )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype
            ).to( device )

        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        vocab = attrgetter(architecture.embed_weight_names)(model)
        config = config.to_dict()
        config.update( architecture.dict() )

        return models.catalog.causal_language_model.Causal_LM( model, tokenizer, vocab, config, **kwargs )

    if architecture.name == "StableDiffusionPipeline":
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
        ).to( device )
        pipeline.scheduler = DDIMScheduler.from_config( pipeline.scheduler.config )
        
        for key in pipeline.config.keys():
            if hasattr( pipeline, key ):
                attr = getattr( pipeline, key )
                if hasattr( attr, "requires_grad_" ):
                    attr.requires_grad_( False )

        tokenizer = pipeline.tokenizer
        text_model = pipeline.text_encoder.text_model
        vocab = text_model.embeddings.token_embedding.weight
        config = config.to_dict()
        config.update( architecture.dict())

        return models.catalog.stablediffusion.StableDiffusion( pipeline, text_model, tokenizer, vocab, config, **kwargs )