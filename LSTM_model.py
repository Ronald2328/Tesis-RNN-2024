import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Capa LSTM
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=layer_dim, 
                            batch_first=True,
                            dropout=dropout_prob)
        
        # Capa densa de salida
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # Inicializar el estado oculto y de la celda
        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(X.device)
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(X.device)

        # Pasar los datos a través de la capa LSTM
        out, (hn, cn) = self.lstm(X, (h0, c0))
        
        # Tomar la salida del último paso de la secuencia (h(T))
        out = self.fc(out[:, -1, :])  # Seleccionamos el último estado oculto
        return out