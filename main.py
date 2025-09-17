import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset

from client import generate_client_fn
from typing import Dict

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    trainloaders,validateloaders,testloader = prepare_dataset(cfg.num_clients,cfg.batch_size)

    
    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validateloaders, num_classes=10)
    ## 4. Define your strategy
    ## 5. Start Simulation
    ## 6. Save your results

if __name__ == "__main__":
    main()
