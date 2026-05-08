import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, n_hidden, depth, init=0.1):
        super().__init__()

        # Corrigido para 2 entradas (x, y)
        self.input   = nn.Linear(2, n_hidden) 
        self.layers  = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(depth)])
        self.predict = nn.Linear(n_hidden, 1)

        self.activation = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # Xavier Normal ou Uniform é ótimo para SiLU/Tanh
        nn.init.xavier_normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_normal_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
    
    def forward(self, x):
        # x deve ser (N, 2)
        x = self.activation(self.input(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Saída direta da temperatura
        return self.predict(x)
