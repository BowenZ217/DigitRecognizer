import numpy as np

import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # raise NotImplementedError("You need to write this part!")
        
        # Calculate the height and width of the images
        self.image_size = int(np.sqrt(in_size / 3))
        
        # Define the activation function
        activation = nn.LeakyReLU()        # Accuracy: 0.85067

        momentum_num = 0.819
        lambda_reg=1e-4
        dampening_num = 0.0
        dropout_rate = 0.2

        conv_kernel_size_num = 5
        conv_padding_num = 1
        pool_kernel_size_num = 2
        pool_stride_num = 2

        self.dropout = nn.Dropout(dropout_rate)

        # Define the CNN architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=conv_kernel_size_num, padding=conv_padding_num),
            nn.BatchNorm2d(32), # can improve accuracy
            activation,
            nn.MaxPool2d(kernel_size=pool_kernel_size_num, stride=pool_stride_num),
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size_num, padding=conv_padding_num),
            nn.BatchNorm2d(64), # can improve accuracy
            activation,
            nn.MaxPool2d(kernel_size=pool_kernel_size_num, stride=pool_stride_num),
            nn.Conv2d(64, 128, kernel_size=conv_kernel_size_num, padding=conv_padding_num),
            # nn.BatchNorm2d(128), # can not improve accuracy
            activation,
            nn.MaxPool2d(kernel_size=pool_kernel_size_num, stride=pool_stride_num)
        )

        # calc image size after convolutions
        conv1_size = (self.image_size - conv_kernel_size_num + 2 * conv_padding_num) + 1
        pool1_size = (conv1_size - pool_kernel_size_num) // pool_stride_num + 1
        conv2_size = (pool1_size - conv_kernel_size_num + 2 * conv_padding_num) + 1
        pool2_size = (conv2_size - pool_kernel_size_num) // pool_stride_num + 1
        conv3_size = (pool2_size - conv_kernel_size_num + 2 * conv_padding_num) + 1
        pool3_size = (conv3_size - pool_kernel_size_num) // pool_stride_num + 1

        flattened_size = 128 * pool3_size * pool3_size

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            activation,
            # self.dropout,
            nn.Linear(128, out_size)
        )

        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=momentum_num, dampening=dampening_num, weight_decay=lambda_reg)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)
        
        x = x.view(x.size(0), 3, self.image_size, self.image_size)  # Reshape the input tensor

        x = self.feature_extractor(x)  # Pass through the feature extraction layers

        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers

        x = self.classifier(x)  # Pass through the classifier to get the final output

        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # raise NotImplementedError("You need to write this part!")
        # return 0.0
        
        self.optimizer.zero_grad()
        y_pred = self.forward(x)

        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu().numpy())
