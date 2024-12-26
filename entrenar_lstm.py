import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class EarlyStoppingConfig:
    """
    Configuración para early stopping.
    
    :param paciencia: Número de épocas sin mejora antes de detener el entrenamiento.
    :param min_delta: Umbral mínimo para considerar una mejora.
    :param monitor: Métrica a monitorear para early stopping.
    """
    paciencia: int = 10
    min_delta: float = 1e-4
    monitor: str = 'val_loss'

def entrenar(
    modelo: torch.nn.Module,
    tasa_aprendizaje: float,
    X_entrenar: torch.Tensor,
    y_entrenar: torch.Tensor,
    X_prueba: torch.Tensor,
    y_prueba: torch.Tensor,
    epocas: int = 200,
    etiqueta: str = 'TOTAL',
    intervalo: int = 50,
    batch_size: Optional[int] = None,
    early_stopping: Optional[EarlyStoppingConfig] = None,
    clip_grad_norm: Optional[float] = 1.0,
    dispositivo: Optional[str] = None
) -> Dict[str, list]:
    """
    Entrena un modelo LSTM con características avanzadas como early stopping,
    gradient clipping y entrenamiento por lotes.

    Args:
        modelo: Modelo PyTorch a entrenar
        tasa_aprendizaje: Learning rate para el optimizador
        X_entrenar: Tensor de datos de entrenamiento de entrada
        y_entrenar: Tensor de datos de entrenamiento objetivo
        X_prueba: Tensor de datos de validación de entrada
        y_prueba: Tensor de datos de validación objetivo
        epocas: Número total de épocas de entrenamiento
        etiqueta: Etiqueta para la barra de progreso
        intervalo: Frecuencia para mostrar métricas
        batch_size: Tamaño del lote para entrenamiento por mini-batches
        early_stopping: Configuración para early stopping
        clip_grad_norm: Valor máximo para gradient clipping
        dispositivo: Dispositivo para entrenamiento ('cuda' o 'cpu')

    Returns:
        Dict[str, list]: Diccionario con el historial de métricas:
            - loss: Pérdida de entrenamiento
            - val_loss: Pérdida de validación
            - mae: Error absoluto medio de entrenamiento
            - val_mae: Error absoluto medio de validación
            - mse: Error cuadrático medio de entrenamiento
            - val_mse: Error cuadrático medio de validación
            
    Examples:
        >>> modelo = LSTM(input_dim=10, hidden_dim=64, layer_dim=2, output_dim=1)
        >>> historia = entrenar(
        ...     modelo=modelo,
        ...     tasa_aprendizaje=0.01,
        ...     X_entrenar=X_train,
        ...     y_entrenar=y_train,
        ...     X_prueba=X_val,
        ...     y_prueba=y_val,
        ...     epocas=100,
        ...     batch_size=32,
        ...     early_stopping=EarlyStoppingConfig(paciencia=5)
        ... )
    """
    # Configurar dispositivo
    if dispositivo is None:
        dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = modelo.to(dispositivo)
    
    # Mover datos al dispositivo
    X_entrenar = X_entrenar.to(dispositivo)
    y_entrenar = y_entrenar.to(dispositivo)
    X_prueba = X_prueba.to(dispositivo)
    y_prueba = y_prueba.to(dispositivo)

    # Configurar optimizador y criterio
    optimizador = torch.optim.SGD(
        modelo.parameters(),
        lr=tasa_aprendizaje,
        momentum=0.9,
        weight_decay=1e-4
    )
    criterio = torch.nn.MSELoss()
    
    # Variables para early stopping
    mejor_valor = float('inf')
    epocas_sin_mejora = 0
    
    # Preparar DataLoader si se especifica batch_size
    if batch_size is not None:
        train_dataset = torch.utils.data.TensorDataset(X_entrenar, y_entrenar)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    historia = {
        'loss': [], 'val_loss': [],
        'mae': [], 'val_mae': [],
        'mse': [], 'val_mse': []
    }

    # Barra de progreso
    with tqdm(total=epocas, desc=f"\033[32mEntrenando Modelo LSTM {etiqueta}\033[0m",
              ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epocas") as pbar:
        
        for epoca in range(epocas):
            modelo.train()
            perdidas_epoca = []
            mae_epoca = []
            mse_epoca = []

            # Entrenamiento por lotes o full batch
            if batch_size is not None:
                for X_batch, y_batch in train_loader:
                    optimizador.zero_grad()
                    salidas = modelo(X_batch)
                    loss = criterio(salidas, y_batch)
                    loss.backward()
                    
                    # Gradient clipping
                    if clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            modelo.parameters(),
                            clip_grad_norm
                        )
                    
                    optimizador.step()
                    
                    # Calcular métricas por lote
                    perdidas_epoca.append(loss.item())
                    mae_epoca.append(torch.mean(torch.abs(salidas - y_batch)).item())
                    mse_epoca.append(torch.mean((salidas - y_batch) ** 2).item())
                
                # Promediar métricas de los lotes
                train_loss = np.mean(perdidas_epoca)
                train_mae = np.mean(mae_epoca)
                train_mse = np.mean(mse_epoca)
            
            else:
                # Entrenamiento full batch
                optimizador.zero_grad()
                salidas = modelo(X_entrenar)
                train_loss = criterio(salidas, y_entrenar)
                train_loss.backward()
                
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        modelo.parameters(),
                        clip_grad_norm
                    )
                
                optimizador.step()
                
                train_mae = torch.mean(torch.abs(salidas - y_entrenar)).item()
                train_mse = torch.mean((salidas - y_entrenar) ** 2).item()
                train_loss = train_loss.item()

            # Evaluación
            modelo.eval()
            with torch.no_grad():
                salidas_prueba = modelo(X_prueba)
                test_loss = criterio(salidas_prueba, y_prueba).item()
                test_mae = torch.mean(torch.abs(salidas_prueba - y_prueba)).item()
                test_mse = torch.mean((salidas_prueba - y_prueba) ** 2).item()

            # Actualizar historia
            for key, value in zip(
                ['loss', 'val_loss', 'mae', 'val_mae', 'mse', 'val_mse'],
                [train_loss, test_loss, train_mae, test_mae, train_mse, test_mse]
            ):
                historia[key].append(value)

            # Early stopping
            if early_stopping:
                valor_actual = historia[early_stopping.monitor][-1]
                if valor_actual < mejor_valor - early_stopping.min_delta:
                    mejor_valor = valor_actual
                    epocas_sin_mejora = 0
                else:
                    epocas_sin_mejora += 1
                
                if epocas_sin_mejora >= early_stopping.paciencia:
                    print(f'\nEarly stopping en época {epoca+1}')
                    break

            # Actualizar barra de progreso
            if (epoca + 1) % intervalo == 0:
                print(f'\nÉpoca {epoca+1}/{epocas}:')
                print(f'Train Loss: {train_loss:.3f}, Val Loss: {test_loss:.3f}')
                print(f'Train MAE: {train_mae:.3f}, Val MAE: {test_mae:.3f}')

            pbar.update(1)
            pbar.set_postfix(
                loss=train_loss,
                val_loss=test_loss,
                train_mae=train_mae,
                val_mae=test_mae
            )

    return historia