import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Implementación de una Red Neuronal Recurrente LSTM (Long Short-Term Memory)
    
    Esta clase implementa una red LSTM multicapa que puede ser utilizada para tareas
    de procesamiento de secuencias como predicción de series temporales o 
    clasificación de secuencias.

    Atributos:
        input_dim (int): Dimensión de entrada para cada elemento de la secuencia
        hidden_dim (int): Número de unidades en las capas ocultas LSTM
        layer_dim (int): Número de capas LSTM apiladas
        output_dim (int): Dimensión del vector de salida
        dropout_prob (float): Probabilidad de dropout entre las capas LSTM

    Arquitectura:
        1. Capas LSTM apiladas con dropout entre ellas
        2. Capa lineal final que proyecta a la dimensión de salida deseada
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """
        Inicializa la red LSTM.

        :param input_dim: Dimensión de entrada para cada elemento de la secuencia
        :param hidden_dim: Número de unidades en las capas ocultas LSTM
        :param layer_dim: Número de capas LSTM apiladas
        :param output_dim: Dimensión del vector de salida
        :param dropout_prob: Probabilidad de dropout entre las capas LSTM

        Ejemplo:
            >>> modelo = LSTM(input_dim=10, hidden_dim=64, layer_dim=2, output_dim=1)
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,       # Dimensión de entrada
            hidden_size=hidden_dim,     # Dimensión del estado oculto
            num_layers=layer_dim,       # Número de capas LSTM
            batch_first=True,           # Espera entrada en formato (batch, seq_len, input_dim)
            dropout=dropout_prob        # Dropout entre capas LSTM
        )
        
        # Capa densa final
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        """
        Realiza la pasada hacia adelante de la red.

        :param X: Tensor de entrada con dimensiones (batch_size, seq_length, input_dim)
        :return: Tensor de salida con dimensiones (batch_size, output_dim)

        Ejemplo:
            >>> modelo = LSTM(input_dim=10, hidden_dim=64, layer_dim=2, output_dim=1)
            >>> X = torch.randn(32, 10, 10)
            >>> out = modelo(X) 
        """
        # Inicializar estados con ceros
        # Dimensiones: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(X.device)
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(X.device)

        # Procesar secuencia
        # out shape: (batch_size, seq_length, hidden_dim)
        # hn, cn shape: (num_layers, batch_size, hidden_dim)
        out, (hn, cn) = self.lstm(X, (h0, c0))
        
        # Usar último estado oculto para predicción
        # Salida final shape: (batch_size, output_dim)
        out = self.fc(out[:, -1, :])
        
        return out