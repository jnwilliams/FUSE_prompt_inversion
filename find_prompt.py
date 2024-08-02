import torch
import os, sys
import pandas as pd
import yaml

import argparse

import time

try:
    from prompt_inversion import generation_methods
    from prompt_inversion.generation_methods import search_strategies
except ModuleNotFoundError as e:
    import generation_methods
    from generation_methods import search_strategies
except Exception as e:
    raise e

CONFIG_YML = "configs/model_config.yml"

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def load_config( config_file = None ):
    if config_file is None:
        config_file = CONFIG_YML
        
    with open( config_file, 'r' ) as f:
        config = yaml.safe_load( f )

    return config
    
def set_search_strategy( runtime_config, search_strategy_config ):
    if runtime_config['search_strategy'] == "greedy_search":
        search_strategy = search_strategies.GreedySearch( )
        search_strategy.config = { 
            'greedy_search' : search_strategy_config['greedy_search']
        }
        return search_strategy

    elif runtime_config['search_strategy'] == "beam_search":
        search_strategy = search_strategies.BeamSearch( 
            **search_strategy_config['beam_search']
        )
        search_strategy.config = { 
            'beam_search' : search_strategy_config['beam_search']
        }
        return search_strategy

    elif runtime_config['search_strategy'] == "uniform_cost_search":
        search_strategy = search_strategies.UniformCostSearch( 
            **search_strategy_config['uniform_cost_search']
        )
        search_strategy.config = { 
            'uniform_cost_search' : search_strategy_config['uniform_cost_search']
        }
        return search_strategy

    raise RuntimeError(f"Unsupported search strategy {runtime_config['search_strategy']}" )
   
def set_generation_method( runtime_config, generation_methods_config, search_strategy_config ):
    if runtime_config['generation_method'] == "GCG":
        generation_method = generation_methods.GCG(
            set_search_strategy( runtime_config, search_strategy_config ),
            runtime_config['models'],
            prompt_len = runtime_config['prompt_length'],
            **generation_methods_config['gcg'],
        )
        generation_method.config = {
            'gcg': generation_methods_config['gcg']
        }
        return generation_method

    elif runtime_config['generation_method'] == "AutoDAN":
        generation_method = generation_methods.AutoDAN(
            set_search_strategy( runtime_config, search_strategy_config ),
            runtime_config['models'],
            prompt_len = runtime_config['prompt_length'],
            **generation_methods_config['autodan'],
        )
        generation_method.config = {
            'autodan': generation_methods_config['autodan']
        }
        return generation_method

    elif runtime_config['generation_method']  == "Random":
        generation_method = generation_methods.Random(
            set_search_strategy( runtime_config, search_strategy_config ),
            runtime_config['models'],
            prompt_len = runtime_config['prompt_length'],
            **generation_methods_config['random'],
        )
        generation_method.config = {
            'random': generation_methods_config['random']
        }
        return generation_method

    elif runtime_config['generation_method'] == "PEZ":
        generation_method = generation_methods.PEZ(
            set_search_strategy( runtime_config, search_strategy_config ),
            runtime_config['models'],
            prompt_len = runtime_config['prompt_length'],
            **generation_methods_config['pez'],
        )
        generation_method.config = {
            'pez': generation_methods_config['pez']
        }
        return generation_method

    elif runtime_config['generation_method'] == "Caption":
        generation_method = generation_methods.Caption(
            set_search_strategy( runtime_config, search_strategy_config ),
            runtime_config['models'],
            prompt_len = runtime_config['prompt_length'],
            **generation_methods_config['caption'],
        )  
        generation_method.config = {
            'caption': generation_methods_config['caption']
        }
        return generation_method
        
    raise RuntimeError(f"Unsupported generation method {runtime_config['generation_method']}" )

def generate_prompt( generation_method, image_path, steps=50 ):
    generation_method._model.primary_model.set_embedding()
    generation_method._model.image_path = image_path
    
    generation_method.x = generation_method._model.primary_model.x.data.clone()
    generation_method.x = generation_method.x.to( torch.float32 )

    if generation_method._search_strategy is not None:
        generation_method._search_strategy.reset()

    generation_method._model.set_image( image_path )

    tic = time.time()
    x, toc = generation_method.search( steps )
    total_time = time.time() - tic

    loss = generation_method._model.loss( x )
    text = generation_method._model.primary_model.to_text( x )
    
    return text[0], loss.item(), toc, total_time

def parse_arguments():
    default_dir = os.path.join(os.getcwd(), 'images', 'test_images', 'house.png' )
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--image_path', type=str, default=default_dir,
                        help='Path to the image file (default: images/test_images/house.png)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to the config file (default: None)')
    args = parser.parse_args()
    return args.image_path, args.config_file
    
def main( ):
    image_path, config_file = parse_arguments()

    config = load_config( config_file )
    generation_method = set_generation_method( 
        config['runtime_parameters'], 
        config['generation_methods'], 
        config['strategies']  
    )
    generation_method._model.set_image( image_path )

    if os.path.isdir( image_path ):
        dirname = image_path
        prompts_found = pd.DataFrame( [] )
        for root, dirs, files in os.walk( image_path ):
            for fil in files:
                if os.path.splitext( fil )[1] == ".txt":
                    continue

                image_path = os.path.join( root, fil )
                
                blockPrint()
                text, loss, toc, _ = generate_prompt( 
                    generation_method, 
                    image_path, steps=config['runtime_parameters']['steps'] )

                prompts_found = pd.concat( 
                    [
                        prompts_found,
                        pd.Series( { 
                        'Image': image_path,
                        'Loss': loss, 
                        'Prompt': text 
                        } )
                    ], axis=1
                )
                enablePrint()

                print( { loss : text } )
            
            prompts_found = prompts_found.T.reset_index( drop=True )
            out_csv_name = (
                f"{dirname.replace( os.sep, '_' )}_"
                f"{config['runtime_parameters']['generation_method']}_"
                f"length_{config['runtime_parameters']['prompt_length']}_"
                "results.csv"
            )
            prompts_found.to_csv( out_csv_name, index=False )
    else:
        text, loss, toc, _ = generate_prompt( 
            generation_method, 
            image_path, 
            steps=config['runtime_parameters']['steps']
        )
        print( ( round( float( toc ), 4 ), { loss : text } ) )

if __name__ == "__main__":
    main()
