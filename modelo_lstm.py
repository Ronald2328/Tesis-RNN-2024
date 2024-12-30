"""
Este script define una implementación de una red neuronal LSTM (Long Short-Term Memory) 
utilizando PyTorch, diseñada para modelar series temporales o datos secuenciales.

Importaciones:
- torch y torch.nn: Bibliotecas de PyTorch para construir y entrenar modelos de aprendizaje profundo.
- device: Variable que define si se utiliza la GPU (si está disponible) o la CPU para ejecutar el modelo.

*Uso típico:
Este modelo puede usarse en tareas como predicción de series temporales, modelado de datos 
secuenciales y aprendizaje de dependencias temporales en conjuntos de datos estructurados.
"""

import torch
import torch.nn as nn

# Configuración del dispositivo: usa GPU si está disponible, de lo contrario usa CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    """
    Implementación de una red neuronal LSTM para modelar datos secuenciales o series temporales.
    
    :param input_dim: Dimensión de los datos de entrada (número de características)
    :param hidden_dim: Dimensión de los estados ocultos de la LSTM
    :param layer_dim: Número de capas en la LSTM
    :param output_dim: Dimensión de los datos de salida (número de clases o características a predecir)
    :return: Modelo LSTM
    """
    super(LSTM, self).__init__()
    self.M = hidden_dim     # Dimensión de las unidades ocultas
    self.L = layer_dim      # Número de capas LSTM

    # Capa LSTM para procesar datos secuenciales
    self.rnn = nn.LSTM(
        input_size=input_dim,      # Dimensionalidad de las características de entrada
        hidden_size=hidden_dim,    # Número de unidades ocultas
        num_layers=layer_dim,      # Número de capas LSTM
        batch_first=True)          # Formato de entrada: (batch, secuencia, características)
    
    # Capa completamente conectada para la salida final
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, X):
    """
    Realiza la propagación hacia adelante a través de la red LSTM.

    :param X: Datos de entrada en formato (batch, secuencia, características)
    :return: Predicciones de la red LSTM
    """
    ## Inicializa el estado oculto (h0) y el estado de celda (c0) en ceros
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    # Procesa la entrada a través de la capa LSTM
    out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))

    # Obtiene la salida del último paso de tiempo y pasa por la capa fc
    out = self.fc(out[:, -1, :])
    
    return out