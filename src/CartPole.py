import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Definir el entorno
env = gym.make('CartPole-v1')


#Red neuronal
class DQL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQL, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Capa oculta 1
        self.fc2 = nn.Linear(128, 128)        # Capa oculta 2
        self.fc3 = nn.Linear(128, output_dim) # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activación ReLU en la primera capa
        x = torch.relu(self.fc2(x))  # Activación ReLU en la segunda capa
        return self.fc3(x)           # Salida (no activacion debido a la busqueda de valor Q)


#Hiperparámetros
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Factor de exploración inicial
epsilon_decay = 0.995  # Decaimiento del epsilon por episodio
epsilon_min = 0.01  # Mínimo valor de epsilon
learning_rate = 0.001  # Tasa de aprendizaje
batch_size = 64  # Tamaño de lote para la reproducción de experiencia
memory_size = 10000  # Capacidad de la memoria de experiencia
target_update_freq = 100  # Frecuencia de actualización de la red objetivo
num_episodes = 500


state = env.reset()
print(state)  # Verifica el estado inicial del entorno