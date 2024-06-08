import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        in_channels = self.input_size
        current_size = 32
        for out_channels in self.hidden_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if self.norm_layer:
                layers.append(self.norm_layer(out_channels))
            layers.append(self.activation())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2,padding=1))
            in_channels = out_channels
            current_size = (current_size + 2 * 1 - 1 * 2) // 2 + 1  
            
            if current_size <= 0:
                raise ValueError("Invalid output size after conv/pooling layers. Adjust the network architecture.")

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(self.hidden_layers[-1] * current_size * current_size, self.num_classes)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Extract the weights from the first convolutional layer
        filters = self.conv_layers[0].weight.data.cpu().numpy()
        num_filters = filters.shape[0]
    
        grid_rows = 8
        grid_cols = 16
    
        # Initialize a large image to hold all the filter images
        filter_height, filter_width = filters.shape[2], filters.shape[3]
        big_image = np.zeros((grid_rows * filter_height, grid_cols * filter_width, 3))
    
        # Normalize and place each filter image in the big image
        for idx in range(num_filters):
            filter_img = filters[idx, :, :, :]
            filter_img = self._normalize(filter_img.transpose(1, 2, 0))
            row = idx // grid_cols
            col = idx % grid_cols
    
            if row < grid_rows and col < grid_cols:
                big_image[row * filter_height: (row + 1) * filter_height,
                          col * filter_width: (col + 1) * filter_width, :] = filter_img
    
        plt.figure(figsize=(10, 10))
        plt.imshow(big_image)
        plt.axis('off')
        plt.show()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        #out = None
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        out = self.fc(x)
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     print(f"Shape after {layer}: {x.shape}")
        # x = x.view(x.size(0), -1)
        # out = self.fc(x)
        return out
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
