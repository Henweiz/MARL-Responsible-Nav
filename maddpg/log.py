import wandb
import random
import os

class Logger:
    # Log into wandb first "wandb login" and paste ur api key there
    def __init__(self, id, name, config):
        wandb.init(
            project="resp-nav",
            name=name,
            config=config, 
            id=id, 
            resume="allow")
    
    def __init__(self, name, config):
        wandb.init(
            project="resp-nav",
            name=name,
            config=config, 
            resume="allow")
        
    def log(self, reward, fitness, loss):
        wandb.log({"reward": reward, "fitness": fitness, "loss": loss})

    def log(self, reward, loss):
        wandb.log({"reward": reward, "loss": loss})    

