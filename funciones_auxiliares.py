"""
Este módulo, `funciones_auxiliares`, incluye funciones útiles para el 
análisis y evaluación de modelos de predicción de temperatura. 

Cada función está diseñada para ser modular y reutilizable en diferentes 
etapas del pipeline de análisis y predicción.
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn as nn
import torch
import seaborn as sns

def train(model, learning_rate, X_train, y_train, X_test, y_test, epochs=200):
    """
    Entrena un modelo PyTorch utilizando SGD y calcula métricas de rendimiento.

    :param model: Modelo a entrenar (instancia de nn.Module).
    :param learning_rate: Tasa de aprendizaje para el optimizador.
    :param X_train: Tensor con los datos de entrada de entrenamiento.
    :param y_train: Tensor con las etiquetas de entrenamiento.
    :param X_test: Tensor con los datos de entrada de prueba.
    :param y_test: Tensor con las etiquetas de prueba.
    :param epochs: Número de épocas para el entrenamiento (por defecto 200).
    :return: Diccionario con el historial de métricas del entrenamiento.
    """
    # Optimizador y función de pérdida
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # Inicializar el historial de entrenamiento
    history = {
        "loss": np.zeros(epochs),
        "val_loss": np.zeros(epochs),
        "mae": np.zeros(epochs),
        "val_mae": np.zeros(epochs),
        "rmse": np.zeros(epochs),
        "val_rmse": np.zeros(epochs),
        "r2": np.zeros(epochs),
        "val_r2": np.zeros(epochs)
    }

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Calcular métricas de entrenamiento
        history["loss"][epoch] = loss.item()
        history["mae"][epoch] = mean_absolute_error(y_train.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        history["rmse"][epoch] = np.sqrt(mean_squared_error(y_train.detach().cpu().numpy(), outputs.detach().cpu().numpy()))
        history["r2"][epoch] = r2_score(y_train.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        # Calcular métricas de validación
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            history["val_loss"][epoch] = test_loss.item()
            history["val_mae"][epoch] = mean_absolute_error(y_test.detach().cpu().numpy(), test_outputs.detach().cpu().numpy())
            history["val_rmse"][epoch] = np.sqrt(mean_squared_error(y_test.detach().cpu().numpy(), test_outputs.detach().cpu().numpy()))
            history["val_r2"][epoch] = r2_score(y_test.detach().cpu().numpy(), test_outputs.detach().cpu().numpy())

        # Mostrar progreso cada 50 épocas
        if (epoch + 1) % 50 == 0:
            print(f'At epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {loss.item():.3f}, '
                  f'Val Loss: {history["val_loss"][epoch]:.3f}, '
                  f'MAE: {history["mae"][epoch]:.3f}, '
                  f'Val MAE: {history["val_mae"][epoch]:.3f}, '
                  f'RMSE: {history["rmse"][epoch]:.3f}, '
                  f'Val RMSE: {history["val_rmse"][epoch]:.3f}, '
                  f'R^2: {history["r2"][epoch]:.3f}, '
                  f'Val R^2: {history["val_r2"][epoch]:.3f}')

    return history

def asignar_estacion(row):
    """
    Determina la estación del año basada en el mes y el día proporcionados.

    :param row: Fila de un DataFrame con las columnas 'mes' y 'dia'.
    :return: La estación del año correspondiente.
    """
    if (row['mes'] == 12 and row['dia'] >= 1) or (row['mes'] == 1) or (row['mes'] == 2):
        return 'Verano'
    elif (row['mes'] == 3) or (row['mes'] == 4) or (row['mes'] == 5):
        return 'Otoño'
    elif (row['mes'] == 6) or (row['mes'] == 7) or (row['mes'] == 8):
        return 'Invierno'
    elif (row['mes'] == 9) or (row['mes'] == 10) or (row['mes'] == 11):
        return 'Primavera'
    return None

# Función para graficar la evolución del entrenamiento
def plot_training_history(history, epoch_start=0, epoch_end=None, tamaño=0.01):
    """
    Genera gráficos que muestran la evolución de la pérdida, MAE y RMSE durante el entrenamiento.

    :param history: Diccionario con métricas de entrenamiento y validación.
    :param epoch_start: Inicio del rango de épocas para graficar.
    :param epoch_end: Fin del rango de épocas para graficar (por defecto, hasta el final).
    :param tamaño: Tamaño de ajuste para las anotaciones en el gráfico.
    """
    if epoch_end is None:
        epoch_end = len(history['loss'])
    
    plt.figure(figsize=(18, 5))
    
    # Gráfico de la pérdida
    plt.subplot(1, 3, 1)
    plt.plot(range(epoch_start, epoch_end), history['loss'][epoch_start:epoch_end], label='Pérdida de Entrenamiento')
    plt.plot(range(epoch_start, epoch_end), history['val_loss'][epoch_start:epoch_end], label='Pérdida de Validación')
    plt.title('Evolución de la Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Inicial: {history["loss"][epoch_start]:.4f}', 
                 xy=(epoch_start, history["loss"][epoch_start]), 
                 xytext=(epoch_start, history["loss"][epoch_start] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))
    plt.annotate(f'Final: {history["loss"][epoch_end - 1]:.4f}', 
                 xy=(epoch_end - 1, history["loss"][epoch_end - 1]), 
                 xytext=(epoch_end - 1, history["loss"][epoch_end - 1] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Gráfico del MAE
    plt.subplot(1, 3, 2)
    plt.plot(range(epoch_start, epoch_end), history['mae'][epoch_start:epoch_end], label='MAE de Entrenamiento')
    plt.plot(range(epoch_start, epoch_end), history['val_mae'][epoch_start:epoch_end], label='MAE de Validación')
    plt.title('Evolución del MAE')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Inicial: {history["mae"][epoch_start]:.4f}', 
                 xy=(epoch_start, history["mae"][epoch_start]), 
                 xytext=(epoch_start, history["mae"][epoch_start] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))
    plt.annotate(f'Final: {history["mae"][epoch_end - 1]:.4f}', 
                 xy=(epoch_end - 1, history["mae"][epoch_end - 1]), 
                 xytext=(epoch_end - 1, history["mae"][epoch_end - 1] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Gráfico del MSE
    plt.subplot(1, 3, 3)
    plt.plot(range(epoch_start, epoch_end), history['rmse'][epoch_start:epoch_end], label='RMSE de Entrenamiento')
    plt.plot(range(epoch_start, epoch_end), history['val_rmse'][epoch_start:epoch_end], label='RMSE de Validación')
    plt.title('Evolución del MSE')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Inicial: {history["rmse"][epoch_start]:.4f}', 
                 xy=(epoch_start, history["rmse"][epoch_start]), 
                 xytext=(epoch_start, history["rmse"][epoch_start] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))
    plt.annotate(f'Final: {history["rmse"][epoch_end - 1]:.4f}', 
                 xy=(epoch_end - 1, history["rmse"][epoch_end - 1]), 
                 xytext=(epoch_end - 1, history["rmse"][epoch_end - 1] + tamaño),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.show()

def plot_temperature_errors(y_train, y_pred):
    """
    Genera gráficos de dispersión, distribución de errores y valores reales vs predicciones para las temperaturas.

    :param y_train: Valores reales de las temperaturas (máxima, mínima y media).
    :param y_pred: Valores predichos de las temperaturas (máxima, mínima y media).
    """
    # Calcular los errores para cada temperatura en los datos de entrenamiento
    error_max = y_train[:, 0] - y_pred[:, 0]
    error_min = y_train[:, 1] - y_pred[:, 1]
    error_media = y_train[:, 2] - y_pred[:, 2]

    # Calcular las métricas de error para cada temperatura en los datos de entrenamiento
    mae_max = mean_absolute_error(y_train[:, 0], y_pred[:, 0])
    rmse_max = np.sqrt(mean_squared_error(y_train[:, 0], y_pred[:, 0]))
    r2_max = r2_score(y_train[:, 0], y_pred[:, 0])
    r_corr_max = np.corrcoef(y_train[:, 0], y_pred[:, 0])[0, 1]

    mae_min = mean_absolute_error(y_train[:, 1], y_pred[:, 1])
    rmse_min = np.sqrt(mean_squared_error(y_train[:, 1], y_pred[:, 1]))
    r2_min = r2_score(y_train[:, 1], y_pred[:, 1])
    r_corr_min = np.corrcoef(y_train[:, 1], y_pred[:, 1])[0, 1]

    mae_media = mean_absolute_error(y_train[:, 2], y_pred[:, 2])
    rmse_media = np.sqrt(mean_squared_error(y_train[:, 2], y_pred[:, 2]))
    r2_media = r2_score(y_train[:, 2], y_pred[:, 2])
    r_corr_media = np.corrcoef(y_train[:, 2], y_pred[:, 2])[0, 1]

    # Crear el gráfico
    fig, axs = plt.subplots(3, 3, figsize=(18, 20))

    # Primera fila: Dispersión con línea de ajuste
    # Gráfico para Temperatura Máxima
    sns.scatterplot(ax=axs[0, 0], x=y_train[:, 0], y=y_pred[:, 0], alpha=0.70)
    sns.regplot(ax=axs[0, 0], x=y_train[:, 0], y=y_pred[:, 0], scatter=False, color='red', line_kws={'linewidth': 2})
    axs[0, 0].plot([min(y_train[:, 0]), max(y_train[:, 0])], [min(y_train[:, 0]), max(y_train[:, 0])], color='black', linestyle='--')
    axs[0, 0].set_title(f'Temperatura Máxima:\nMAE={mae_max:.4f}, RMSE={rmse_max:.4f}, R={r_corr_max:.4f}, R2={r2_max:.4f}')
    axs[0, 0].set_xlabel('Real (°C)')
    axs[0, 0].set_ylabel('Predicción (°C)')

    # Gráfico para Temperatura Mínima
    sns.scatterplot(ax=axs[0, 1], x=y_train[:, 1], y=y_pred[:, 1], alpha=0.70)
    sns.regplot(ax=axs[0, 1], x=y_train[:, 1], y=y_pred[:, 1], scatter=False, color='red', line_kws={'linewidth': 2})
    axs[0, 1].plot([min(y_train[:, 1]), max(y_train[:, 1])], [min(y_train[:, 1]), max(y_train[:, 1])], color='black', linestyle='--')
    axs[0, 1].set_title(f'Temperatura Mínima:\nMAE={mae_min:.4f}, RMSE={rmse_min:.4f}, R={r_corr_min:.4f}, R2={r2_min:.4f}')
    axs[0, 1].set_xlabel('Real (°C)')
    axs[0, 1].set_ylabel('Predicción (°C)')

    # Gráfico para Temperatura Media
    sns.scatterplot(ax=axs[0, 2], x=y_train[:, 2], y=y_pred[:, 2], alpha=0.70)
    sns.regplot(ax=axs[0, 2], x=y_train[:, 2], y=y_pred[:, 2], scatter=False, color='red', line_kws={'linewidth': 2})
    axs[0, 2].plot([min(y_train[:, 2]), max(y_train[:, 2])], [min(y_train[:, 2]), max(y_train[:, 2])], color='black', linestyle='--')
    axs[0, 2].set_title(f'Temperatura Media:\nMAE={mae_media:.4f}, RMSE={rmse_media:.4f}, R={r_corr_media:.4f}, R2={r2_media:.4f}')
    axs[0, 2].set_xlabel('Real (°C)')
    axs[0, 2].set_ylabel('Predicción (°C)')

    # Segunda fila: Distribución de errores
    # Distribución de error en temperatura máxima
    sns.histplot(error_max, ax=axs[1, 0], kde=True, color='red', bins=20, fill=True, alpha=0.15)
    axs[1, 0].set_title(f'Distribución de Error (Temperatura Máxima)\nMAX: {error_max.max():.2f} MIN: {error_max.min():.2f}')
    axs[1, 0].set_xlabel('Error (°C)')
    axs[1, 0].set_ylabel('Frecuencia')

    # Distribución de error en temperatura mínima
    sns.histplot(error_min, ax=axs[1, 1], kde=True, color='red', bins=20, fill=True, alpha=0.15)
    axs[1, 1].set_title(f'Distribución de Error (Temperatura Mínima)\nMAX: {error_min.max():.2f} MIN: {error_min.min():.2f}')
    axs[1, 1].set_xlabel('Error (°C)')
    axs[1, 1].set_ylabel('Frecuencia')

    # Distribución de error en temperatura media
    sns.histplot(error_media, ax=axs[1, 2], kde=True, color='red', bins=20, fill=True, alpha=0.15)
    axs[1, 2].set_title(f'Distribución de Error (Temperatura Media)\nMAX: {error_media.max():.2f} MIN: {error_media.min():.2f}')
    axs[1, 2].set_xlabel('Error (°C)')
    axs[1, 2].set_ylabel('Frecuencia')

    # Tercera fila: Gráficos de línea de los valores reales y predicción
    # Temperatura máxima
    axs[2, 0].plot(y_train[:, 0], label='Real (°C)', color='blue')
    axs[2, 0].plot(y_pred[:, 0], label='Predicción (°C)', color='red')
    axs[2, 0].set_title('Temperatura Máxima (Real vs Predicción)')
    axs[2, 0].set_xlabel('Día')
    axs[2, 0].set_ylabel('Temperatura (°C)')
    axs[2, 0].legend()

    # Temperatura mínima
    axs[2, 1].plot(y_train[:, 1], label='Real (°C)', color='blue')
    axs[2, 1].plot(y_pred[:, 1], label='Predicción (°C)', color='red')
    axs[2, 1].set_title('Temperatura Mínima (Real vs Predicción)')
    axs[2, 1].set_xlabel('Día')
    axs[2, 1].set_ylabel('Temperatura (°C)')
    axs[2, 1].legend()

    # Temperatura media
    axs[2, 2].plot(y_train[:, 2], label='Real (°C)', color='blue')
    axs[2, 2].plot(y_pred[:, 2], label='Predicción (°C)', color='red')
    axs[2, 2].set_title('Temperatura Media (Real vs Predicción)')
    axs[2, 2].set_xlabel('Día')
    axs[2, 2].set_ylabel('Temperatura (°C)')
    axs[2, 2].legend()

    plt.tight_layout()
    plt.show()