import wandb
import random
import os

class Logger:
    def __init__(self, id, name, config):
        # Run "wandb login" in your console firsy
        wandb_api = os.environ['WANDB_API_KEY']

        wandb.login(key=wandb_api)
        wandb.init(
            project="resp-nav",
            name=name,
            config=config, 
            id=id, 
            resume="allow")
        
    def log(self, reward, fitness, loss):
        wandb.log({"reward": reward, "fitness": fitness, "loss": loss})

