"""
=====================================================================

neural_energy.py

Standalone Neural Helmholtz Free Energy

Physics-Informed Constitutive Neural Network

Author : Rahul Kumar
Institute : IIT Madras

=====================================================================
"""

import torch
import torch.nn as nn

import config


#####################################################################
# Activation Function
#####################################################################

def activation():

    """
    Activation function used throughout the network.
    """

    if config.ACTIVATION.lower() == "softplus":

        return nn.Softplus(beta=config.SOFTPLUS_BETA)

    elif config.ACTIVATION.lower() == "relu":

        return nn.ReLU()

    elif config.ACTIVATION.lower() == "tanh":

        return nn.Tanh()

    else:

        raise ValueError("Unknown activation function")


#####################################################################
# Weight Initialization
#####################################################################

def initialize_weights(model):

    """
    Xavier initialization.
    """

    for m in model.modules():

        if isinstance(m, nn.Linear):

            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:

                nn.init.zeros_(m.bias)


#####################################################################
# Residual Block
#####################################################################

class ResidualBlock(nn.Module):

    """
    Residual block

    x -> Linear -> Softplus -> Linear -> +x

    """

    def __init__(self, width):

        super().__init__()

        self.linear1 = nn.Linear(width, width)

        self.linear2 = nn.Linear(width, width)

        self.activation = activation()

    #############################################################

    def forward(self, x):

        identity = x

        out = self.linear1(x)

        out = self.activation(out)

        out = self.linear2(out)

        out = out + identity

        out = self.activation(out)

        return out


#####################################################################
# Neural Helmholtz Energy
#####################################################################

class NeuralHelmholtz(nn.Module):

    """
    Standalone Neural Helmholtz Potential

    Inputs

    I1
    I2
    J
    I41
    I42

    Output

    Psi
    """

    def __init__(self):

        super().__init__()

        self.input_layer = nn.Linear(

            config.INPUT_DIM,

            config.HIDDEN_LAYERS[0]

        )

        self.activation = activation()

        self.block1 = ResidualBlock(

            config.HIDDEN_LAYERS[0]

        )

        self.block2 = ResidualBlock(

            config.HIDDEN_LAYERS[0]

        )

        self.block3 = ResidualBlock(

            config.HIDDEN_LAYERS[0]

        )

        self.hidden = nn.Sequential(

            nn.Linear(

                config.HIDDEN_LAYERS[0],

                config.HIDDEN_LAYERS[1]

            ),

            activation(),

            nn.Linear(

                config.HIDDEN_LAYERS[1],

                config.HIDDEN_LAYERS[2]

            ),

            activation(),

            nn.Linear(

                config.HIDDEN_LAYERS[2],

                config.HIDDEN_LAYERS[3]

            ),

            activation(),

            nn.Linear(

                config.HIDDEN_LAYERS[3],

                config.HIDDEN_LAYERS[4]

            ),

            activation()

        )

        self.output_layer = nn.Linear(

            config.HIDDEN_LAYERS[4],

            config.OUTPUT_DIM

        )

        initialize_weights(self)

    #############################################################

    def forward(

            self,

            I1,

            I2,

            J,

            I41,

            I42

    ):

        """
        Forward pass

        Returns

        Helmholtz free energy
        """

        x = torch.cat(

            [

                I1,

                I2,

                J,

                I41,

                I42

            ],

            dim=1

        )

        x = self.input_layer(x)

        x = self.activation(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.hidden(x)

        psi = self.output_layer(x)

        return psi


#####################################################################
# Model Builder
#####################################################################

def build_model():

    model = NeuralHelmholtz()

    model = model.to(config.DEVICE)

    return model


#####################################################################
# Unit Test
#####################################################################

if __name__ == "__main__":

    model = build_model()

    batch = 8

    I1 = torch.rand(batch,1,device=config.DEVICE)

    I2 = torch.rand(batch,1,device=config.DEVICE)

    J = torch.ones(batch,1,device=config.DEVICE)

    I41 = torch.rand(batch,1,device=config.DEVICE)

    I42 = torch.rand(batch,1,device=config.DEVICE)

    psi = model(

        I1,

        I2,

        J,

        I41,

        I42

    )

    print("="*60)

    print("Neural Helmholtz Model")

    print("="*60)

    print(model)

    print()

    print("Output Shape :", psi.shape)

    print()

    print("Output")

    print(psi)
    