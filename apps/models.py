import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBNBlock(a, b, k, s, device=None, dtype="float32"):
    #a为输入通道，b为输出通道， k为kernel_size, s为stride
    module = nn.Sequential(nn.Conv(a, b, k, s, device=device, dtype=dtype), 
                           nn.BatchNorm2d(b, device=device, dtype=dtype),
                           nn.ReLU())
    return module
class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.ConvBN1 = ConvBNBlock(3, 16, 7, 4, device=device, dtype=dtype)
        self.ConvBN2 = ConvBNBlock(16, 32, 3, 2, device=device, dtype=dtype)
        self.ConvBN3 = nn.Residual(nn.Sequential(ConvBNBlock(32, 32, 3, 1, device=device, dtype=dtype)
                                   , ConvBNBlock(32, 32, 3, 1, device=device, dtype=dtype)))
        self.ConvBN4 = ConvBNBlock(32, 64, 3, 2, device=device, dtype=dtype)
        self.ConvBN5 = ConvBNBlock(64, 128, 3, 2, device=device, dtype=dtype)
        self.ConvBN6 = nn.Residual(nn.Sequential(ConvBNBlock(128, 128, 3, 1, device=device, dtype=dtype),
                                   ConvBNBlock(128, 128, 3, 1, device=device, dtype=dtype)))
        self.Linear0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype), 
            nn.ReLU(), 
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.ConvBN1(x)
        x = self.ConvBN2(x)
        x = self.ConvBN3(x)
        x = self.ConvBN4(x)
        x = self.ConvBN5(x)
        x = self.ConvBN6(x)
        x = self.Linear0(x)
        return x
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)