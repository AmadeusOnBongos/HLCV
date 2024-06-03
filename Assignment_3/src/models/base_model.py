import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()+ '\n'
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################
        ret_str += "Number of trainable parameters per layer:\n"
        
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                ret_str += f"{name}: {num_params}\n"

        ret_str += f"Total number of trainable parameters: {total_params}\n"
        
        return ret_str