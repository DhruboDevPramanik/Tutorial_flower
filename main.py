import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    trainloaders,validateloaders,testloader = prepare_dataset(cfg.num_clients,cfg.batch_size)

    print(len(trainloaders),len(validateloaders),len(testloader),len(trainloaders[0]),len(validateloaders[0]),len(testloader))
    ## 3. Define your clients
    ## 4. Define your strategy
    ## 5. Start Simulation
    ## 6. Save your results

if __name__ == "__main__":
    main()
