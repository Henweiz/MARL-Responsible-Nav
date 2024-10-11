import wandb
import random
import os

class Logger:
    # Log into wandb first "wandb login" and paste ur api key there
    def __init__(self, name, config, id=None):
        wandb.init(
            project="resp-nav",
            name=name,
            config=config, 
            id=id, 
            resume="allow")
    
        
    def log(self, reward, fitness, loss, steps):
        wandb.log({"reward": reward, "fitness": fitness, "loss": loss, "steps": steps})

    def log(self, reward, loss, steps):
        wandb.log({"reward": reward, "loss": loss, "steps": steps})    

