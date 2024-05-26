import yaml
from models import *
from experiment import VAEXperiment

config_path = "./configs/cvae.yaml"
checkpoint_path = "./logs/ConditionalVAE/version_38/checkpoints/last.ckpt"

with open(config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path, vae_model=model, params=config['exp_params'])
experiment.eval()

experiment.sample_specific_class(torch.FloatTensor([1, 0, 0]))
