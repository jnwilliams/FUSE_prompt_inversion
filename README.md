# FUSE-ing Language Models Zero-Shot Adapter Discovery for Prompt Optimization Across Tokenizers

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Minimal Example](#minimal-example)
4. [Configuration](#configuration)
    - [Runtime Parameters](#runtime-parameters)
    - [Model Parameters](#model-parameters)
    - [Search Strategies](#search-strategies)
    - [Optimizers](#optimizers)
5. [References](#references)

## Overview

Welcome to the official GitHub repository for FUSE-ing Language Models: Zero-Shot Adapter Discovery for Prompt Optimization Across Tokenizers (arXiv: [arXiv ID]).

This repository provides an implementation of a simple adapter, which we call FUSE (Flexible Unification of Semantic Embeddings), that acts as an inexpensive approximation to an adapter layer that maps from one model’s textual embedding space to another, even across different tokenizers. In this repository, we provide several discrete optimizers for prompt optimization on image generation models that can be used to test the efficacy of the adapter or for a variety of prompt optimization related tasks in image generation.

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:jnwilliams/FUSE_prompt_inversion.git
   cd FUSE_prompt_inversion
   ```

2. Create a new environment using conda (use `spec-file-windows.txt` if on windows machines):
   ```
   conda create -n myenv python=3.9 --file spec-file.txt
   conda activate myenv
   ```

## Minimal Example

Once the environment is set up, the following is a minimal example to caption an image by optimizing a prompt over GPT2-medium and CLIP-VIT-B32.

`python find_prompt.py --image_path images/house.png --config_file configs/model_config.yml`

## Configuration

There is an example configuration file in `configs/config.yaml` that users should use to specify runtime parameters, models, discrete optimizers (i.e., generation methods), and search strategies (e.g., greedy search, beam search, uniform cost search) for the optimizers.  

Below is a breakdown of the configuration options:

### Runtime Parameters

```yaml
runtime_parameters:
  generation_method: AutoDAN  # generation method
  search_strategy: beam_search  # search strategy
  prompt_length: 0  # number of free tokens to optimize
  steps: 16  # number of steps for the generation method to run
```

### Model Parameters

```yaml
models:
  adapter_params:
    save_gradients_to: ./gradients
    max_examples_per_case: 8192
    max_subtokens: 4

  primary_model:
    model_id: gpt2-medium      # Hugging Face model ID
    loss_function:             # only necessary for causal language models
      type: nll_loss           # log_prob_loss, kl_div_loss, nll_loss, or cross_entropy_loss
      label: "Sure, here's"    # target label (only used with cross_entropy)
    parameters:
      weight: 1.               # weight for this model's loss in the objective
      prefix: "An image of"    # prefix for free tokens
      suffix: ""               # suffix for free tokens

  secondary_models:
    - model_id: openai/clip-vit-base-patch32
      parameters:
        weight: 22.            # weight for this model's loss in the objective
        prefix: "An image of" # prefix for free tokens
        suffix: ""             # suffix for free tokens
```

In this section, we define the models and relevant parameters that we use to compute losses and gradients. The above case defines the objective:

$\alpha_{0} \cdot L_{NLL}(f_{\theta}(E), E) + \alpha_{1} \cdot CLIP_{\theta}( T_{f:CLIP}(E), I) $,
  
Where $L_{NLL}$ is the negative log likelihood loss for predicted next tokens with GPT2-medium, $CLIP_{\theta}(T_{f:CLIP}(E), I)$ is 1 minus cosine similarity between the clip embedding of the prompt and a given image, $\alpha_{0}, \alpha_{1}$ are weights for each model.

#### Parameters:

- **adapter_params:** The FUSE Adapter precomputes gradients for words in which a model requires up to {max_subtokens} tokens to represent. The gradient that it computes is saved in {save_gradients_to} and uses {max_examples_per_case} words in the wikitext corpus to fit each precomputed gradient. 

- **loss_function:** Causal Language Models (e.g., GPT, Mistral, Llama) can use a variety of loss functions. For these models, we implement 4 loss functions:
  - *log_prob_loss:* See (https://arxiv.org/abs/2205.12558), the log probability of the current token in embedding space given all previous tokens
  - *nll_loss:* The negative log likelihood of the current token, given all previous tokens
  - *cross_entropy_loss:* For targeted generation. The likelihood of the text defined in *label* below given the current embedding.
  - *kl_div_loss:* An approximation of log_prob_loss, based on the KL Divergence between the model's predicted token distribution and the current embedding.

- **weight:** How much to weight this model in the objective
- **prefix:** Add a prefix to the free tokens during optimization. Each model can have a different prefix.  
- **suffix:** Add a suffix to the free tokens during optimization. Each model can have a different suffix.  

### Search Strategies

```yaml
strategies:
  beam_search:
    beam_width: 4              # beam width for beam search

  greedy_search:
  
  uniform_cost_search:
    frontier_max_size: 32      # max size for uniform cost search frontier
```

#### Parameters:

- **beam_search:** When using GCG, AutoDAN, or Random Search, use a beam search instead of a greedy search. Each method will create {beam_width} beams when solving.
- **uniform_cost_search:** When using GCG, AutoDAN, or Random Search, use a uniform cost search. The frontier of the search is limited to the top {frontier_max_size} sequences at every step. 
- **greedy_search:** When using GCG, AutoDAN, or Random Search, use a greedy search.

### Optimizers

```yaml
generation_methods:
  gcg:
    batch_size: 512            # batch size for computing losses
    top_k: 256                 # choose candidates from top_k gradients for each free token

  autodan:
    batch_size: 512            # batch size for computing losses
    logit_weight: 1.           # how much to weight readability vs gradient

  random:
    batch_size: 128            # batch size for computing losses
    search_ball: L0            # kind of ball to sample candidates from (L2 or L0)

  pez:
    lr: 0.1                    # step size
    weight_decay: 0.1          # weight decay

  caption:
    model_id: Salesforce/blip2-opt-2.7b # Hugging Face ID for captioner
```

#### Parameters and Brief Description of the Optimizers:

- **gcg:** An implementation of Greedy Coordinate Descent (https://arxiv.org/pdf/2307.15043). For every token in a sequence, use the gradient of the current iterate to choose the {top_k} potential replacement candidates for every token in the sequence. Then randomly select {batch_size} of these candidates choosing the best option as the next iterate.

- **autodan:** An implementation of AutoDAN (https://arxiv.org/pdf/2310.15140). Apply GCG's candidate selection with an additional readability prior on a dummy token that is appended to the end of the current iterate. Take {batch_size} top candidates and choose the best option to place at the end of the sequence and continue for the next iterate.

- **random:** An implementation of Random Search. The user can select either an $l_{0}$ or $l_{2}$ ball for the search. The model then samples {batch_size} candidates and computes the objective for each, choosing the best performing candidate as the next iterate.

- **pez:** An implementation of PEZ (https://arxiv.org/pdf/2302.03668). The algorithm performs projected gradient descent, optimizing all tokens in the sequence at once. At every step, project a continuous embedding to its nearest neighbor in the embedding table, and apply the gradient at the discrete point to the continous embedding for the optimizer.

- **caption:** A captioner given by: {model_id}. This method outputs the caption using this model for a given image.
