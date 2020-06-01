import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, device, num_layers):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out, h


    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return hidden